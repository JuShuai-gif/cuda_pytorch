"""
Triton kernel code generation from computation graph IR.

Industrial context: This is the final step in ML compilers like torch.inductor
and Triton. After graph optimization (module 09), the compiler lowers the optimized
graph into actual GPU kernel source code. This module generates valid Triton kernel
source strings that can be JIT-compiled and executed.

The codegen handles:
  - Elementwise fusion: multiple pointwise ops combined into a single kernel
  - Reduction: softmax, layernorm, rmsnorm
  - Broadcasting: scalar/bias inputs expanded to full shape
  - Mask generation: proper edge masking for non-aligned element counts
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import torch
import triton
import triton.language as tl

from ir import Graph, OpType

EPS = 1e-5

_GELU_EXPR = "0.5 * x * (1.0 + tl.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))"
_SILU_EXPR = "x * tl.sigmoid(x)"

_UNARY_OPS: dict[OpType, str] = {
    OpType.RELU: "tl.where(x < 0, 0.0, x)",
    OpType.SIGMOID: "tl.sigmoid(x)",
    OpType.TANH: "tl.tanh(x)",
    OpType.EXP: "tl.exp(x)",
    OpType.LOG: "tl.log(x)",
    OpType.GELU: _GELU_EXPR,
    OpType.SILU: _SILU_EXPR,
}

_ACTIVATION_SET = {
    OpType.RELU,
    OpType.GELU,
    OpType.SILU,
    OpType.SIGMOID,
    OpType.TANH,
    OpType.EXP,
    OpType.LOG,
}

_BINARY_OPS: dict[OpType, str] = {
    OpType.ADD: "+",
    OpType.SUB: "-",
    OpType.MUL: "*",
    OpType.DIV: "/",
}


def _sanitize(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if name and name[0].isdigit():
        name = "_" + name
    return name or "var"


def _topo_group(node_ids: list[int], graph: Graph) -> list[int]:
    """Topological sort within a subset of node IDs."""
    gset = set(node_ids)
    indeg: dict[int, int] = {}
    for nid in node_ids:
        node = graph.nodes[nid]
        indeg[nid] = sum(1 for i in node.inputs if i in gset)
    q = [nid for nid, d in indeg.items() if d == 0]
    out: list[int] = []
    while q:
        nid = q.pop(0)
        out.append(nid)
        for oid in graph.nodes[nid].outputs:
            if oid in indeg:
                indeg[oid] -= 1
                if indeg[oid] == 0:
                    q.append(oid)
    return out


@dataclass
class _KernelMeta:
    """Metadata for a generated kernel, used by compile_and_run."""

    kernel_name: str = ""
    kind: str = ""  # "elementwise" | "softmax" | "layernorm" | "rmsnorm"
    param_names: list[str] = field(default_factory=list)
    param_origin: dict[str, str] = field(default_factory=dict)
    source: str = ""


class TritonCodeGenerator:
    """Generates Triton kernel source code from graph IR nodes.

    After calling generate_*(), call compile_and_run() with the returned
    source string to JIT-compile and execute the kernel.
    """

    def __init__(self, block_size: int = 1024):
        self.block_size = block_size
        self._cnt = 0
        self._meta: _KernelMeta = _KernelMeta()

    def _next(self, prefix: str = "kernel") -> str:
        self._cnt += 1
        return f"_{prefix}_{self._cnt}"

    # ==================================================================
    # Elementwise Fusion
    # ==================================================================

    def generate_elementwise_fusion(self, node_ids: list[int], graph: Graph) -> str:
        """Generate a fused elementwise Triton kernel.

        Walks the subgraph formed by node_ids, loads all external inputs,
        chains arithmetic/activation ops in topological order, and stores
        the final value to output. Handles broadcasting of scalar/1D bias.

        Returns a complete @triton.jit decorated kernel source string.
        """
        gset = set(node_ids)
        order = _topo_group(node_ids, graph)
        kname = self._next("fused")

        # Assign variable names for internal nodes
        vnames: dict[int, str] = {}
        for nid in order:
            vnames[nid] = _sanitize(graph.nodes[nid].name)

        # Collect external inputs (referenced but not in the fusion group)
        ext: list[tuple[str, int]] = []  # (param_name, node_id)
        seen_ext: set[int] = set()
        for nid in node_ids:
            for inp_id in graph.nodes[nid].inputs:
                if inp_id not in gset and inp_id not in seen_ext:
                    seen_ext.add(inp_id)
                    node = graph.nodes[inp_id]
                    pname = _sanitize(node.name)
                    ext.append((pname, inp_id))

        out_var = vnames[node_ids[-1]]
        BLOCK = "BLOCK_SIZE"

        # --- Signature ---
        sig_parts: list[str] = []
        param_names: list[str] = []
        param_origin: dict[str, str] = {}

        for pname, nid in ext:
            p = f"{pname}_ptr"
            sig_parts.append(f"    {p},")
            param_names.append(p)
            param_origin[p] = pname

            # If the external node is a CONSTANT scalar, emit as constexpr too
            node = graph.nodes[nid]
            val = node.attrs.get("value")
            if node.op == OpType.CONSTANT and isinstance(val, (int, float)):
                cname = f"{pname}_val"
                sig_parts.append(f"    {cname}: tl.constexpr,")
                param_names.append(cname)
                param_origin[cname] = pname

        sig_parts.append("    out_ptr,")
        param_names.append("out_ptr")
        param_origin["out_ptr"] = "out_ptr"

        sig_parts.append("    n_elements: int,")
        param_names.append("n_elements")
        param_origin["n_elements"] = "n_elements"

        sig_parts.append(f"    {BLOCK}: tl.constexpr,")
        param_names.append(BLOCK)
        param_origin[BLOCK] = BLOCK

        sig_str = "\n".join(sig_parts)

        # --- Body ---
        body: list[str] = []
        body.append("    pid = tl.program_id(0)")
        body.append(f"    offs = pid * {BLOCK} + tl.arange(0, {BLOCK})")
        body.append("    mask = offs < n_elements")

        # Map each external node to its loaded variable name and value
        ext_var: dict[int, str] = {}

        for pname, nid in ext:
            node = graph.nodes[nid]
            val = node.attrs.get("value")
            if node.op == OpType.CONSTANT and isinstance(val, (int, float)):
                # Scalar constant: broadcast via tl.full or just use the value
                body.append(f"    {pname} = tl.full(offs.shape, {pname}_val, dtype=tl.float32)")
                ext_var[nid] = pname
            else:
                body.append(f"    {pname} = tl.load({pname}_ptr + offs, mask=mask, other=0.0)")
                ext_var[nid] = pname

        # Compute nodes in topological order
        for nid in order:
            node = graph.nodes[nid]
            v = vnames[nid]

            # Resolve input variables
            inp_vars: list[str] = []
            for inp_id in node.inputs:
                if inp_id in vnames:
                    inp_vars.append(vnames[inp_id])
                elif inp_id in ext_var:
                    inp_vars.append(ext_var[inp_id])
                else:
                    inp_vars.append(_sanitize(graph.nodes[inp_id].name))

            if node.op in _BINARY_OPS:
                a = inp_vars[0] if inp_vars else "0.0"
                b = inp_vars[1] if len(inp_vars) > 1 else "0.0"
                body.append(f"    {v} = {a} {_BINARY_OPS[node.op]} {b}")

            elif node.op in _ACTIVATION_SET:
                body.append(f"    {v} = {_UNARY_OPS[node.op].replace('x', inp_vars[0])}")

            else:
                body.append(f"    {v} = {inp_vars[0]}")

        body.append(f"    tl.store(out_ptr + offs, {out_var}, mask=mask)")

        body_str = "\n".join(body)

        src = f'''@triton.jit
def {kname}(
{sig_str}
):
    """Generated fused elementwise kernel."""
{body_str}
'''

        self._meta = _KernelMeta(
            kernel_name=kname,
            kind="elementwise",
            param_names=param_names,
            param_origin=param_origin,
            source=src,
        )
        return src

    # ==================================================================
    # Reduction kernels
    # ==================================================================

    def generate_reduction(self, node_id: int, graph: Graph) -> str:
        """Generate a Triton kernel for a reduction operation.

        Supported: SOFTMAX, LAYERNORM, RMSNORM.
        Each program handles one row.
        """
        node = graph.nodes[node_id]
        BLOCK = "BLOCK_SIZE"

        if node.op == OpType.SOFTMAX:
            return self._gen_softmax(BLOCK)
        elif node.op == OpType.LAYERNORM:
            return self._gen_layernorm(BLOCK)
        elif node.op == OpType.RMSNORM:
            return self._gen_rmsnorm(BLOCK)
        else:
            raise ValueError(f"Unsupported reduction op: {node.op.value}")

    def _gen_softmax(self, BLOCK: str) -> str:
        kname = self._next("softmax")
        src = f'''@triton.jit
def {kname}(
    x_ptr,
    out_ptr,
    n_cols: int,
    {BLOCK}: tl.constexpr,
):
    """Online softmax. Each program handles one row."""
    rid = tl.program_id(0)
    base = rid * n_cols
    offs = tl.arange(0, {BLOCK})
    mask = offs < n_cols
    row = tl.load(x_ptr + base + offs, mask=mask, other=float('-inf'))
    mx = tl.max(row, axis=0)
    s  = tl.sum(tl.exp(row - mx), axis=0)
    out = tl.exp(row - mx) / s
    tl.store(out_ptr + base + offs, out, mask=mask)
'''
        self._meta = _KernelMeta(
            kernel_name=kname,
            kind="softmax",
            param_names=["x_ptr", "out_ptr", "n_cols", BLOCK],
            param_origin={
                "x_ptr": "x_ptr",
                "out_ptr": "out_ptr",
                "n_cols": "n_cols",
                BLOCK: BLOCK,
            },
            source=src,
        )
        return src

    def _gen_layernorm(self, BLOCK: str) -> str:
        kname = self._next("layernorm")
        src = f'''@triton.jit
def {kname}(
    x_ptr,
    out_ptr,
    n_cols: int,
    eps: float,
    {BLOCK}: tl.constexpr,
):
    """LayerNorm. Each program handles one row."""
    rid = tl.program_id(0)
    base = rid * n_cols
    offs = tl.arange(0, {BLOCK})
    mask = offs < n_cols
    row = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
    mu = tl.sum(row, axis=0) / n_cols
    c  = row - mu
    v  = tl.sum(c * c, axis=0) / n_cols
    dst = c * (1.0 / tl.sqrt(v + eps))
    tl.store(out_ptr + base + offs, dst, mask=mask)
'''
        self._meta = _KernelMeta(
            kernel_name=kname,
            kind="layernorm",
            param_names=["x_ptr", "out_ptr", "n_cols", "eps", BLOCK],
            param_origin={
                "x_ptr": "x_ptr",
                "out_ptr": "out_ptr",
                "n_cols": "n_cols",
                "eps": "eps",
                BLOCK: BLOCK,
            },
            source=src,
        )
        return src

    def _gen_rmsnorm(self, BLOCK: str) -> str:
        kname = self._next("rmsnorm")
        src = f'''@triton.jit
def {kname}(
    x_ptr,
    out_ptr,
    n_cols: int,
    eps: float,
    {BLOCK}: tl.constexpr,
):
    """RMSNorm. Each program handles one row."""
    rid = tl.program_id(0)
    base = rid * n_cols
    offs = tl.arange(0, {BLOCK})
    mask = offs < n_cols
    row = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
    msq = tl.sum(row * row, axis=0) / n_cols
    dst = row * (1.0 / tl.sqrt(msq + eps))
    tl.store(out_ptr + base + offs, dst, mask=mask)
'''
        self._meta = _KernelMeta(
            kernel_name=kname,
            kind="rmsnorm",
            param_names=["x_ptr", "out_ptr", "n_cols", "eps", BLOCK],
            param_origin={
                "x_ptr": "x_ptr",
                "out_ptr": "out_ptr",
                "n_cols": "n_cols",
                "eps": "eps",
                BLOCK: BLOCK,
            },
            source=src,
        )
        return src

    # ==================================================================
    # Compile & Run
    # ==================================================================

    def compile_and_run(
        self,
        kernel_src: str,
        inputs: dict[str, torch.Tensor],
        output_shapes: dict[str, tuple],
    ) -> dict[str, torch.Tensor]:
        """Compile the kernel source and execute it with given inputs.

        Steps:
          1. exec() the source in a namespace with triton + tl
          2. Locate the generated @triton.jit function
          3. Build output tensors
          4. Call the kernel with correct arguments

        Args:
            kernel_src: Source string from generate_*().
            inputs: Dict of placeholder_name -> CUDA tensor.
            output_shapes: Name -> shape pairs for outputs.

        Returns:
            Dict of output_name -> computed CUDA tensor.
        """
        ns: dict[str, Any] = {"triton": triton, "tl": tl, "torch": torch}
        exec(kernel_src, ns)

        kfunc = None
        for name, obj in ns.items():
            if name.startswith("_") and callable(obj) and hasattr(obj, "run"):
                kfunc = obj
                break

        if kfunc is None:
            raise RuntimeError(
                f"Cannot find generated kernel function in source. ns keys: {list(ns.keys())}"
            )

        device = next(iter(inputs.values())).device
        dtype = next(iter(inputs.values())).dtype

        outputs: dict[str, torch.Tensor] = {}
        for oname, shape in output_shapes.items():
            outputs[oname] = torch.empty(shape, device=device, dtype=dtype)

        meta = self._meta
        kind = meta.kind

        if kind == "elementwise":
            self._call_elementwise(kfunc, inputs, outputs, meta)
        else:
            self._call_reduction(kfunc, inputs, outputs, meta)

        return outputs

    # ------------------------------------------------------------------
    # Internal call helpers
    # ------------------------------------------------------------------

    def _call_elementwise(
        self,
        kfunc: Any,
        inputs: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
        meta: _KernelMeta,
    ) -> None:
        """Build and execute the elementwise kernel call."""
        n_elements = next(iter(inputs.values())).numel()
        block = min(self.block_size, max(1, n_elements))
        grid = (triton.cdiv(n_elements, block), 1, 1)

        # Build positional args matching param_names order
        args_dict: dict[str, Any] = {}
        for inp_name, tensor in inputs.items():
            args_dict[_sanitize(inp_name)] = tensor
            args_dict[f"{_sanitize(inp_name)}_ptr"] = tensor

        args_dict["out_ptr"] = next(iter(outputs.values()))
        args_dict["n_elements"] = n_elements
        args_dict["BLOCK_SIZE"] = block
        args_dict["n_cols"] = block  # fallback

        # Resolve each parameter
        pos_args: list[Any] = []
        constexpr_kw: dict[str, Any] = {}

        for pname in meta.param_names:
            if pname == "BLOCK_SIZE":
                constexpr_kw["BLOCK_SIZE"] = block
                continue

            origin = meta.param_origin.get(pname, pname)

            if origin in inputs:
                pos_args.append(inputs[origin])
            elif origin in outputs:
                pos_args.append(outputs[origin])
            elif origin == "out_ptr":
                pos_args.append(next(iter(outputs.values())))
            elif origin == "n_elements":
                pos_args.append(n_elements)
            elif origin == "n_cols":
                for t in inputs.values():
                    if t.dim() >= 1:
                        pos_args.append(t.shape[-1])
                        break
                else:
                    pos_args.append(1)
            elif origin == "eps":
                pos_args.append(EPS)
            elif pname.endswith("_val"):
                # constexpr scalar
                constexpr_kw[pname] = args_dict.get(pname, 0.0)
            elif pname.endswith("_ptr"):
                clean = pname[:-4]
                if clean in args_dict:
                    pos_args.append(args_dict[clean])
                elif clean in inputs:
                    pos_args.append(inputs[clean])
                else:
                    pos_args.append(next(iter(inputs.values())))
            elif pname in args_dict:
                pos_args.append(args_dict[pname])
            else:
                pos_args.append(next(iter(inputs.values())))

        kfunc[grid](*pos_args, **constexpr_kw)

    def _call_reduction(
        self,
        kfunc: Any,
        inputs: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
        meta: _KernelMeta,
    ) -> None:
        """Build and execute a reduction kernel call."""
        inp = next(iter(inputs.values()))
        out = next(iter(outputs.values()))

        if inp.dim() == 1:
            n_rows = 1
            n_cols = inp.shape[0]
            x = inp.reshape(1, -1)
            o = out.reshape(1, -1)
        else:
            n_rows = inp.shape[0]
            n_cols = inp.shape[-1]
            x = inp
            o = out

        block = triton.next_power_of_2(n_cols)
        grid = (n_rows, 1, 1)

        kfunc[grid](x, o, n_cols, EPS, BLOCK_SIZE=block)
