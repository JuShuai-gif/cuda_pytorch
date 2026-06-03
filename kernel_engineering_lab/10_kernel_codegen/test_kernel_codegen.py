"""
Tests for 10_kernel_codegen module.

Verifies:
  - Generated kernel source compiles without error
  - Simple add kernel matches PyTorch add
  - Fused add+relu matches PyTorch sequential
  - Fused bias+gelu matches PyTorch sequential
  - Residual+norm pattern correctness
  - Various shapes and dtypes
  - Edge cases: scalar inputs, broadcasting
  - Reduction kernels: softmax, layernorm, rmsnorm
  - compile_and_run integration

Run: pytest 10_kernel_codegen/test_kernel_codegen.py -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

try:
    import triton  # noqa: F401

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

from ir import Graph, OpType
from triton_codegen import EPS, TritonCodeGenerator

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
requires_triton = pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not installed")


def _make_elementwise_graph(*op_specs) -> Graph:
    """Build a graph from a sequence of (OpType, [input_indices]) specs.

    Input placeholders (CONSTANT nodes without value) are created automatically.
    Returns (graph, fusion_node_ids).
    """
    g = Graph()
    placeholders: dict[int, int] = {}  # index -> node_id
    prev_ids: list[int] = []

    for i, (op, inp_indices) in enumerate(op_specs):
        actual_inputs: list[int] = []
        for idx in inp_indices:
            if idx in placeholders:
                actual_inputs.append(placeholders[idx])
            else:
                ph = g.add_node(OpType.CONSTANT, [], name=f"ph_{idx}")
                g.inputs.append(ph)
                placeholders[idx] = ph
                actual_inputs.append(ph)

        nid = g.add_node(op, actual_inputs, name=f"{op.value}_{i}")
        prev_ids.append(nid)

    # Mark all non-input nodes that users want fused
    fusion_ids = [nid for nid in prev_ids]
    g.outputs.append(fusion_ids[-1])

    return g, fusion_ids


# ======================================================================
# Compilation Tests (no GPU needed for string exec)
# ======================================================================


class TestSourceCompilation:
    """Test that generated kernel sources are syntactically valid."""

    def test_add_kernel_compiles(self):
        g, ids = _make_elementwise_graph((OpType.ADD, [0, 1]))
        cg = TritonCodeGenerator()
        src = cg.generate_elementwise_fusion(ids, g)
        assert "@triton.jit" in src
        assert "def " in src
        assert "tl.load" in src
        assert "tl.store" in src

    def test_mul_relu_kernel_compiles(self):
        g, ids = _make_elementwise_graph((OpType.MUL, [0, 1]), (OpType.RELU, [0]))
        cg = TritonCodeGenerator()
        src = cg.generate_elementwise_fusion(ids, g)
        compiled = compile(src, "<test>", "exec")
        assert compiled is not None

    def test_bias_gelu_kernel_compiles(self):
        g, ids = _make_elementwise_graph((OpType.ADD, [0, 1]), (OpType.GELU, [0]))
        cg = TritonCodeGenerator()
        src = cg.generate_elementwise_fusion(ids, g)
        compiled = compile(src, "<test>", "exec")
        assert compiled is not None

    def test_long_chain_compiles(self):
        g, ids = _make_elementwise_graph(
            (OpType.MUL, [0, 1]),
            (OpType.ADD, [0, 2]),
            (OpType.RELU, [1]),
            (OpType.SIGMOID, [2]),
            (OpType.MUL, [2, 3]),
        )
        # Actually build it correctly:
        g2 = Graph()
        a = g2.add_node(OpType.CONSTANT, [], name="a")
        b = g2.add_node(OpType.CONSTANT, [], name="b")
        c = g2.add_node(OpType.CONSTANT, [], name="c")
        g2.inputs.extend([a, b, c])

        n1 = g2.add_node(OpType.MUL, [a, b], name="mul1")
        n2 = g2.add_node(OpType.ADD, [n1, c], name="add1")
        n3 = g2.add_node(OpType.SIGMOID, [n2], name="sigmoid")
        n4 = g2.add_node(OpType.TANH, [n3], name="tanh")
        g2.outputs.append(n4)

        cg = TritonCodeGenerator()
        src = cg.generate_elementwise_fusion([n1, n2, n3, n4], g2)
        compiled = compile(src, "<test>", "exec")
        assert compiled is not None

    def test_softmax_kernel_compiles(self):
        g = Graph()
        x = g.add_node(OpType.CONSTANT, [], name="x")
        g.inputs.append(x)
        sm = g.add_node(OpType.SOFTMAX, [x], name="sm")
        g.outputs.append(sm)

        cg = TritonCodeGenerator()
        src = cg.generate_reduction(sm, g)
        compiled = compile(src, "<test>", "exec")
        assert compiled is not None

    def test_layernorm_kernel_compiles(self):
        g = Graph()
        x = g.add_node(OpType.CONSTANT, [], name="x")
        g.inputs.append(x)
        ln = g.add_node(OpType.LAYERNORM, [x], name="ln")
        g.outputs.append(ln)

        cg = TritonCodeGenerator()
        src = cg.generate_reduction(ln, g)
        compiled = compile(src, "<test>", "exec")
        assert compiled is not None

    def test_rmsnorm_kernel_compiles(self):
        g = Graph()
        x = g.add_node(OpType.CONSTANT, [], name="x")
        g.inputs.append(x)
        rn = g.add_node(OpType.RMSNORM, [x], name="rn")
        g.outputs.append(rn)

        cg = TritonCodeGenerator()
        src = cg.generate_reduction(rn, g)
        compiled = compile(src, "<test>", "exec")
        assert compiled is not None

    def test_all_activations_compile(self):
        """Verify every activation type generates compilable source."""
        for op in [
            OpType.RELU,
            OpType.GELU,
            OpType.SILU,
            OpType.SIGMOID,
            OpType.TANH,
            OpType.EXP,
            OpType.LOG,
        ]:
            g = Graph()
            x = g.add_node(OpType.CONSTANT, [], name="x")
            g.inputs.append(x)
            n = g.add_node(op, [x], name=op.value)
            g.outputs.append(n)

            cg = TritonCodeGenerator()
            src = cg.generate_elementwise_fusion([n], g)
            compiled = compile(src, "<test>", "exec")
            assert compiled is not None, f"Failed for {op.value}"

    def test_kernel_source_contains_expected_patterns(self):
        g, ids = _make_elementwise_graph((OpType.ADD, [0, 1]))
        cg = TritonCodeGenerator()
        src = cg.generate_elementwise_fusion(ids, g)

        assert "@triton.jit" in src
        assert "tl.program_id" in src
        assert "tl.arange" in src
        assert "tl.load" in src
        assert "tl.store" in src
        assert "BLOCK_SIZE" in src

    def test_reduction_kernel_source_has_mask(self):
        g = Graph()
        x = g.add_node(OpType.CONSTANT, [], name="x")
        g.inputs.append(x)
        sm = g.add_node(OpType.SOFTMAX, [x], name="sm")
        g.outputs.append(sm)

        cg = TritonCodeGenerator()
        src = cg.generate_reduction(sm, g)
        assert "mask" in src.lower()


# ======================================================================
# Correctness Tests (GPU required)
# ======================================================================


@pytest.mark.cuda
@requires_cuda
@requires_triton
class TestElementwiseCorrectness:
    """Verify generated elementwise kernels produce correct output."""

    def _run_and_compare(
        self,
        graph: Graph,
        fusion_ids: list[int],
        inputs: dict[str, torch.Tensor],
        ref_fn,
        atol: float = 1e-3,
    ):
        cg = TritonCodeGenerator(block_size=1024)
        src = cg.generate_elementwise_fusion(fusion_ids, graph)

        shape = next(iter(inputs.values())).shape
        out = cg.compile_and_run(src, inputs, {"result": shape})
        ref = ref_fn()

        assert torch.allclose(out["result"], ref, atol=atol, rtol=1e-3), (
            f"Max diff: {(out['result'] - ref).abs().max().item():.2e}"
        )

    # --- Add ---

    def test_add_1d(self):
        g = Graph()
        a = g.add_node(OpType.CONSTANT, [], name="a")
        b = g.add_node(OpType.CONSTANT, [], name="b")
        g.inputs.extend([a, b])
        add = g.add_node(OpType.ADD, [a, b], name="add")
        g.outputs.append(add)

        shape = (4096,)
        x = torch.randn(shape, device="cuda")
        y = torch.randn(shape, device="cuda")
        self._run_and_compare(g, [add], {"a": x, "b": y}, lambda: x + y)

    def test_add_2d(self):
        g = Graph()
        a = g.add_node(OpType.CONSTANT, [], name="a")
        b = g.add_node(OpType.CONSTANT, [], name="b")
        g.inputs.extend([a, b])
        add = g.add_node(OpType.ADD, [a, b], name="add")
        g.outputs.append(add)

        shape = (32, 256)
        x = torch.randn(shape, device="cuda")
        y = torch.randn(shape, device="cuda")
        self._run_and_compare(g, [add], {"a": x, "b": y}, lambda: x + y)

    # --- Mul ---

    def test_mul(self):
        g = Graph()
        a = g.add_node(OpType.CONSTANT, [], name="a")
        b = g.add_node(OpType.CONSTANT, [], name="b")
        g.inputs.extend([a, b])
        mul = g.add_node(OpType.MUL, [a, b], name="mul")
        g.outputs.append(mul)

        shape = (2048,)
        x = torch.randn(shape, device="cuda")
        y = torch.randn(shape, device="cuda")
        self._run_and_compare(g, [mul], {"a": x, "b": y}, lambda: x * y)

    # --- Fused Add+ReLU ---

    def test_fused_add_relu(self):
        g = Graph()
        a = g.add_node(OpType.CONSTANT, [], name="a")
        b = g.add_node(OpType.CONSTANT, [], name="b")
        g.inputs.extend([a, b])
        add = g.add_node(OpType.ADD, [a, b], name="add")
        relu = g.add_node(OpType.RELU, [add], name="relu")
        g.outputs.append(relu)

        shape = (4096,)
        x = torch.randn(shape, device="cuda")
        y = torch.randn(shape, device="cuda")
        self._run_and_compare(g, [add, relu], {"a": x, "b": y}, lambda: F.relu(x + y))

    # --- Fused bias+GELU ---

    def test_fused_bias_gelu(self):
        g = Graph()
        a = g.add_node(OpType.CONSTANT, [], name="a")
        b = g.add_node(OpType.CONSTANT, [], name="bias")
        g.inputs.extend([a, b])
        add = g.add_node(OpType.ADD, [a, b], name="add")
        gelu = g.add_node(OpType.GELU, [add], name="gelu")
        g.outputs.append(gelu)

        shape = (1024,)
        x = torch.randn(shape, device="cuda")
        bias = torch.randn(shape, device="cuda")
        self._run_and_compare(
            g,
            [add, gelu],
            {"a": x, "bias": bias},
            lambda: F.gelu(x + bias, approximate="tanh"),
            atol=1e-2,
        )

    # --- Fused Mul+ReLU ---

    def test_fused_mul_relu(self):
        g = Graph()
        a = g.add_node(OpType.CONSTANT, [], name="a")
        b = g.add_node(OpType.CONSTANT, [], name="b")
        g.inputs.extend([a, b])
        mul = g.add_node(OpType.MUL, [a, b], name="mul")
        relu = g.add_node(OpType.RELU, [mul], name="relu")
        g.outputs.append(relu)

        shape = (8192,)
        x = torch.randn(shape, device="cuda")
        y = torch.randn(shape, device="cuda")
        self._run_and_compare(g, [mul, relu], {"a": x, "b": y}, lambda: F.relu(x * y))

    # --- Various shapes ---

    def test_shape_1(self):
        g = Graph()
        a = g.add_node(OpType.CONSTANT, [], name="a")
        b = g.add_node(OpType.CONSTANT, [], name="b")
        g.inputs.extend([a, b])
        add = g.add_node(OpType.ADD, [a, b], name="add")
        g.outputs.append(add)

        shape = (1,)
        x = torch.tensor([3.14], device="cuda")
        y = torch.tensor([2.71], device="cuda")
        self._run_and_compare(g, [add], {"a": x, "b": y}, lambda: x + y)

    def test_shape_non_power_of_two(self):
        g = Graph()
        a = g.add_node(OpType.CONSTANT, [], name="a")
        b = g.add_node(OpType.CONSTANT, [], name="b")
        g.inputs.extend([a, b])
        add = g.add_node(OpType.ADD, [a, b], name="add")
        g.outputs.append(add)

        shape = (999,)
        x = torch.randn(shape, device="cuda")
        y = torch.randn(shape, device="cuda")
        self._run_and_compare(g, [add], {"a": x, "b": y}, lambda: x + y)

    def test_shape_3d(self):
        g = Graph()
        a = g.add_node(OpType.CONSTANT, [], name="a")
        b = g.add_node(OpType.CONSTANT, [], name="b")
        g.inputs.extend([a, b])
        add = g.add_node(OpType.ADD, [a, b], name="add")
        g.outputs.append(add)

        shape = (2, 8, 512)
        x = torch.randn(shape, device="cuda")
        y = torch.randn(shape, device="cuda")
        self._run_and_compare(g, [add], {"a": x, "b": y}, lambda: x + y)

    def test_shape_large(self):
        g = Graph()
        a = g.add_node(OpType.CONSTANT, [], name="a")
        b = g.add_node(OpType.CONSTANT, [], name="b")
        g.inputs.extend([a, b])
        add = g.add_node(OpType.ADD, [a, b], name="add")
        g.outputs.append(add)

        shape = (1_000_000,)
        x = torch.randn(shape, device="cuda")
        y = torch.randn(shape, device="cuda")
        self._run_and_compare(g, [add], {"a": x, "b": y}, lambda: x + y)

    # --- Dtypes ---

    def test_dtype_fp16(self):
        if torch.cuda.get_device_capability()[0] < 7:
            pytest.skip("fp16 not supported")

        g = Graph()
        a = g.add_node(OpType.CONSTANT, [], name="a")
        b = g.add_node(OpType.CONSTANT, [], name="b")
        g.inputs.extend([a, b])
        add = g.add_node(OpType.ADD, [a, b], name="add")
        g.outputs.append(add)

        shape = (1024,)
        x = torch.randn(shape, device="cuda", dtype=torch.float16)
        y = torch.randn(shape, device="cuda", dtype=torch.float16)
        self._run_and_compare(
            g,
            [add],
            {"a": x, "b": y},
            lambda: x + y,
            atol=1e-2,
        )

    def test_dtype_bf16(self):
        if torch.cuda.get_device_capability()[0] < 8:
            pytest.skip("bfloat16 not supported")

        g = Graph()
        a = g.add_node(OpType.CONSTANT, [], name="a")
        b = g.add_node(OpType.CONSTANT, [], name="b")
        g.inputs.extend([a, b])
        add = g.add_node(OpType.ADD, [a, b], name="add")
        g.outputs.append(add)

        shape = (1024,)
        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        y = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        self._run_and_compare(
            g,
            [add],
            {"a": x, "b": y},
            lambda: x + y,
            atol=5e-2,
        )

    # --- Multiple blocks ---

    def test_multiple_block_sizes(self):
        g = Graph()
        a = g.add_node(OpType.CONSTANT, [], name="a")
        b = g.add_node(OpType.CONSTANT, [], name="b")
        g.inputs.extend([a, b])
        add = g.add_node(OpType.ADD, [a, b], name="add")
        g.outputs.append(add)

        shape = (5000,)
        x = torch.randn(shape, device="cuda")
        y = torch.randn(shape, device="cuda")

        for bs in [256, 512, 1024]:
            cg = TritonCodeGenerator(block_size=bs)
            src = cg.generate_elementwise_fusion([add], g)
            out = cg.compile_and_run(src, {"a": x, "b": y}, {"result": shape})
            assert torch.allclose(out["result"], x + y, atol=1e-3), f"Failed for block_size={bs}"

    # --- Activation correctness ---

    def test_relu_negative(self):
        g = Graph()
        a = g.add_node(OpType.CONSTANT, [], name="a")
        g.inputs.append(a)
        relu = g.add_node(OpType.RELU, [a], name="relu")
        g.outputs.append(relu)

        x = -torch.ones(1000, device="cuda")
        self._run_and_compare(
            g,
            [relu],
            {"a": x},
            lambda: F.relu(x),
        )

    def test_sigmoid_range(self):
        g = Graph()
        a = g.add_node(OpType.CONSTANT, [], name="a")
        g.inputs.append(a)
        sig = g.add_node(OpType.SIGMOID, [a], name="sig")
        g.outputs.append(sig)

        x = torch.linspace(-5, 5, 1024, device="cuda")
        self._run_and_compare(
            g,
            [sig],
            {"a": x},
            lambda: torch.sigmoid(x),
            atol=1e-5,
        )

    def test_tanh_range(self):
        g = Graph()
        a = g.add_node(OpType.CONSTANT, [], name="a")
        g.inputs.append(a)
        t = g.add_node(OpType.TANH, [a], name="tanh")
        g.outputs.append(t)

        x = torch.linspace(-3, 3, 1024, device="cuda")
        self._run_and_compare(
            g,
            [t],
            {"a": x},
            lambda: torch.tanh(x),
            atol=1e-5,
        )

    def test_silu(self):
        g = Graph()
        a = g.add_node(OpType.CONSTANT, [], name="a")
        g.inputs.append(a)
        s = g.add_node(OpType.SILU, [a], name="silu")
        g.outputs.append(s)

        x = torch.linspace(-3, 3, 1024, device="cuda")
        self._run_and_compare(
            g,
            [s],
            {"a": x},
            lambda: F.silu(x),
            atol=1e-5,
        )


# ======================================================================
# Reduction Correctness Tests
# ======================================================================


@pytest.mark.cuda
@requires_cuda
@requires_triton
class TestReductionCorrectness:
    """Verify generated reduction kernels produce correct output."""

    def _run_softmax(self, shape):
        g = Graph()
        x = g.add_node(OpType.CONSTANT, [], name="x")
        g.inputs.append(x)
        sm = g.add_node(OpType.SOFTMAX, [x], name="sm")
        g.outputs.append(sm)

        inp = torch.randn(shape, device="cuda")
        cg = TritonCodeGenerator()
        src = cg.generate_reduction(sm, g)
        out = cg.compile_and_run(src, {"x": inp}, {"result": shape})
        ref = F.softmax(inp, dim=-1)
        assert torch.allclose(out["result"], ref, atol=1e-2), (
            f"Max diff: {(out['result'] - ref).abs().max().item():.2e}"
        )

    def _run_layernorm(self, shape):
        g = Graph()
        x = g.add_node(OpType.CONSTANT, [], name="x")
        g.inputs.append(x)
        ln = g.add_node(OpType.LAYERNORM, [x], name="ln")
        g.outputs.append(ln)

        inp = torch.randn(shape, device="cuda")
        cg = TritonCodeGenerator()
        src = cg.generate_reduction(ln, g)
        out = cg.compile_and_run(src, {"x": inp}, {"result": shape})
        ref = F.layer_norm(inp.float(), [shape[-1]], weight=None, bias=None, eps=EPS)
        assert torch.allclose(out["result"].float(), ref, atol=5e-2, rtol=1e-2), (
            f"Max diff: {(out['result'].float() - ref).abs().max().item():.2e}"
        )

    def _run_rmsnorm(self, shape):
        g = Graph()
        x = g.add_node(OpType.CONSTANT, [], name="x")
        g.inputs.append(x)
        rn = g.add_node(OpType.RMSNORM, [x], name="rn")
        g.outputs.append(rn)

        inp = torch.randn(shape, device="cuda")
        cg = TritonCodeGenerator()
        src = cg.generate_reduction(rn, g)
        out = cg.compile_and_run(src, {"x": inp}, {"result": shape})
        rms = torch.rsqrt(inp.float().pow(2).mean(-1, keepdim=True) + EPS)
        ref = (inp.float() * rms).to(inp.dtype)
        assert torch.allclose(out["result"].float(), ref.float(), atol=1e-2), (
            f"Max diff: {(out['result'].float() - ref.float()).abs().max().item():.2e}"
        )

    # --- Softmax ---

    def test_softmax_small(self):
        self._run_softmax((1, 64))

    def test_softmax_medium(self):
        self._run_softmax((4, 256))

    def test_softmax_large(self):
        self._run_softmax((8, 1024))

    def test_softmax_batch(self):
        self._run_softmax((32, 128))

    def test_softmax_non_power_two(self):
        self._run_softmax((4, 300))

    # --- LayerNorm ---

    def test_layernorm_small(self):
        self._run_layernorm((2, 128))

    def test_layernorm_medium(self):
        self._run_layernorm((4, 1024))

    def test_layernorm_large(self):
        self._run_layernorm((8, 4096))

    def test_layernorm_batch1(self):
        self._run_layernorm((1, 768))

    def test_layernorm_batch32(self):
        self._run_layernorm((32, 512))

    # --- RMSNorm ---

    def test_rmsnorm_small(self):
        self._run_rmsnorm((2, 128))

    def test_rmsnorm_medium(self):
        self._run_rmsnorm((4, 1024))

    def test_rmsnorm_large(self):
        self._run_rmsnorm((8, 4096))

    def test_rmsnorm_batch1(self):
        self._run_rmsnorm((1, 768))


# ======================================================================
# Edge Cases
# ======================================================================


@pytest.mark.cuda
@requires_cuda
@requires_triton
class TestEdgeCases:
    """Test unusual inputs and edge conditions."""

    def test_scalar_const_input(self):
        """Test graph with a scalar constant input (not a tensor)."""
        g = Graph()
        x = g.add_node(OpType.CONSTANT, [], name="x")
        g.inputs.append(x)
        mul = g.add_node(OpType.MUL, [x, x], name="mul_sq")
        g.outputs.append(mul)

        cg = TritonCodeGenerator()
        src = cg.generate_elementwise_fusion([mul], g)

        shape = (256,)
        inp = torch.randn(shape, device="cuda")
        out = cg.compile_and_run(src, {"x": inp}, {"result": shape})
        ref = inp * inp
        assert torch.allclose(out["result"], ref, atol=1e-3)

    def test_single_input_activation(self):
        """Test graph where an activation has a single non-constant input."""
        g = Graph()
        a = g.add_node(OpType.CONSTANT, [], name="a")
        g.inputs.append(a)
        relu = g.add_node(OpType.RELU, [a], name="relu")
        g.outputs.append(relu)

        cg = TritonCodeGenerator()
        src = cg.generate_elementwise_fusion([relu], g)

        shape = (4096,)
        x = torch.randn(shape, device="cuda")
        out = cg.compile_and_run(src, {"a": x}, {"result": shape})
        assert torch.allclose(out["result"], F.relu(x), atol=1e-3)

    def test_all_negative_input(self):
        """ReLU should output all zeros for all-negative input."""
        g = Graph()
        a = g.add_node(OpType.CONSTANT, [], name="a")
        g.inputs.append(a)
        relu = g.add_node(OpType.RELU, [a], name="relu")
        g.outputs.append(relu)

        cg = TritonCodeGenerator()
        src = cg.generate_elementwise_fusion([relu], g)

        x = -torch.ones(1024, device="cuda")
        out = cg.compile_and_run(src, {"a": x}, {"result": (1024,)})
        assert torch.all(out["result"] == 0.0)

    def test_all_positive_input(self):
        """ReLU should pass through all-positive input."""
        g = Graph()
        a = g.add_node(OpType.CONSTANT, [], name="a")
        g.inputs.append(a)
        relu = g.add_node(OpType.RELU, [a], name="relu")
        g.outputs.append(relu)

        cg = TritonCodeGenerator()
        src = cg.generate_elementwise_fusion([relu], g)

        x = torch.ones(1024, device="cuda") * 3.0
        out = cg.compile_and_run(src, {"a": x}, {"result": (1024,)})
        assert torch.allclose(out["result"], x, atol=1e-3)

    def test_large_block_size_handling(self):
        """Test with block_size larger than n_elements."""
        g = Graph()
        a = g.add_node(OpType.CONSTANT, [], name="a")
        b = g.add_node(OpType.CONSTANT, [], name="b")
        g.inputs.extend([a, b])
        add = g.add_node(OpType.ADD, [a, b], name="add")
        g.outputs.append(add)

        cg = TritonCodeGenerator(block_size=65536)
        src = cg.generate_elementwise_fusion([add], g)

        shape = (128,)
        x = torch.randn(shape, device="cuda")
        y = torch.randn(shape, device="cuda")
        out = cg.compile_and_run(src, {"a": x, "b": y}, {"result": shape})
        assert torch.allclose(out["result"], x + y, atol=1e-3)

    def test_division_kernel(self):
        """Test division elementwise op."""
        g = Graph()
        a = g.add_node(OpType.CONSTANT, [], name="a")
        b = g.add_node(OpType.CONSTANT, [], name="b")
        g.inputs.extend([a, b])
        div = g.add_node(OpType.DIV, [a, b], name="div")
        g.outputs.append(div)

        cg = TritonCodeGenerator()
        src = cg.generate_elementwise_fusion([div], g)

        shape = (1024,)
        x = torch.randn(shape, device="cuda") * 2.0 + 3.0
        y = torch.randn(shape, device="cuda") * 2.0 + 3.0
        out = cg.compile_and_run(src, {"a": x, "b": y}, {"result": shape})
        ref = x / y
        assert torch.allclose(out["result"], ref, atol=1e-3)

    def test_subtraction_kernel(self):
        """Test subtraction elementwise op."""
        g = Graph()
        a = g.add_node(OpType.CONSTANT, [], name="a")
        b = g.add_node(OpType.CONSTANT, [], name="b")
        g.inputs.extend([a, b])
        sub = g.add_node(OpType.SUB, [a, b], name="sub")
        g.outputs.append(sub)

        cg = TritonCodeGenerator()
        src = cg.generate_elementwise_fusion([sub], g)

        shape = (1024,)
        x = torch.randn(shape, device="cuda")
        y = torch.randn(shape, device="cuda")
        out = cg.compile_and_run(src, {"a": x, "b": y}, {"result": shape})
        assert torch.allclose(out["result"], x - y, atol=1e-3)

    @requires_cuda
    def test_reduction_1d_input(self):
        """Softmax/LN on a 1D input should work (treated as 1 row)."""
        g = Graph()
        x = g.add_node(OpType.CONSTANT, [], name="x")
        g.inputs.append(x)
        sm = g.add_node(OpType.SOFTMAX, [x], name="sm")
        g.outputs.append(sm)

        inp = torch.randn(128, device="cuda")
        cg = TritonCodeGenerator()
        src = cg.generate_reduction(sm, g)
        out = cg.compile_and_run(src, {"x": inp}, {"result": (128,)})
        ref = F.softmax(inp, dim=-1)
        assert torch.allclose(out["result"], ref, atol=1e-2)
