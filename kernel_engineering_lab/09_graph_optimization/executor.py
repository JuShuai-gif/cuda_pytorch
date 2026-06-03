"""
Execute a computation graph IR by running each node in topological order.

Industrial context: Before generating machine code, we need a reference interpreter
to verify correctness of graph transformations. This is the equivalent of running
a graph through an eager executor - every XLA/TVM/TensorRT system has one.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ir import DType, Graph, OpType, TensorShape


def _apply_activation(x: torch.Tensor, op: OpType) -> torch.Tensor:
    if op == OpType.RELU:
        return F.relu(x)
    elif op == OpType.GELU:
        return F.gelu(x)
    elif op == OpType.SILU:
        return F.silu(x)
    elif op == OpType.SIGMOID:
        return torch.sigmoid(x)
    elif op == OpType.TANH:
        return torch.tanh(x)
    elif op == OpType.EXP:
        return torch.exp(x)
    elif op == OpType.LOG:
        return torch.log(x)
    return x


def _layernorm(x: torch.Tensor) -> torch.Tensor:
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mu) / torch.sqrt(var + 1e-5)


def _rmsnorm(x: torch.Tensor) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-5)
    return x / rms


def execute_graph(graph: Graph, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Execute graph in topological order.

    Args:
        graph: The computation graph to execute.
        inputs: Dict mapping input node names to torch.Tensor values.

    Returns:
        Dict mapping output node names to computed tensor values.
    """
    tensors: dict[int, torch.Tensor] = {}
    order = graph.topological_sort()

    # Map input names -> node ids (reverse of what's stored)
    input_by_name: dict[str, int] = {}
    for nid in graph.inputs:
        if nid in graph.nodes:
            node = graph.nodes[nid]
            input_by_name[node.name] = nid

    for name, tensor in inputs.items():
        if name in input_by_name:
            tensors[input_by_name[name]] = tensor
        else:
            # Try matching by exact node name matching
            for nid in graph.inputs:
                if nid in graph.nodes and graph.nodes[nid].name == name:
                    tensors[nid] = tensor
                    break

    for nid in order:
        if nid in tensors:
            continue  # Already has a value (input nodes)

        node = graph.nodes[nid]
        inp_tensors = [tensors[inp] for inp in node.inputs if inp in tensors]

        if node.op == OpType.CONSTANT:
            val = node.attrs.get("value")
            if isinstance(val, torch.Tensor):
                tensors[nid] = val
            elif isinstance(val, (int, float)):
                shape = node.attrs.get("_const_shape", None)
                if shape is not None:
                    tensors[nid] = torch.full(shape, float(val))
                else:
                    tensors[nid] = torch.tensor(float(val))

        elif node.op == OpType.ADD:
            tensors[nid] = inp_tensors[0] + inp_tensors[1]
        elif node.op == OpType.SUB:
            tensors[nid] = inp_tensors[0] - inp_tensors[1]
        elif node.op == OpType.MUL:
            tensors[nid] = inp_tensors[0] * inp_tensors[1]
        elif node.op == OpType.DIV:
            tensors[nid] = inp_tensors[0] / (inp_tensors[1] + 1e-8)

        elif node.op in (
            OpType.RELU,
            OpType.GELU,
            OpType.SILU,
            OpType.SIGMOID,
            OpType.TANH,
            OpType.EXP,
            OpType.LOG,
        ):
            tensors[nid] = _apply_activation(inp_tensors[0], node.op)

        elif node.op == OpType.MATMUL:
            tensors[nid] = torch.matmul(inp_tensors[0], inp_tensors[1])

        elif node.op == OpType.SOFTMAX:
            dim = node.attrs.get("dim", -1)
            tensors[nid] = F.softmax(inp_tensors[0], dim=dim)

        elif node.op == OpType.REDUCE_SUM:
            dim = node.attrs.get("dim", -1)
            keepdim = node.attrs.get("keepdim", False)
            tensors[nid] = torch.sum(inp_tensors[0], dim=dim, keepdim=keepdim)

        elif node.op == OpType.LAYERNORM:
            tensors[nid] = _layernorm(inp_tensors[0])

        elif node.op == OpType.RMSNORM:
            tensors[nid] = _rmsnorm(inp_tensors[0])

        elif node.op == OpType.RESHAPE:
            shape_list = node.attrs.get("shape", None)
            if shape_list is not None:
                tensors[nid] = inp_tensors[0].reshape(shape_list)
            else:
                tensors[nid] = inp_tensors[0]

        elif node.op == OpType.TRANSPOSE:
            dim0 = node.attrs.get("dim0", -2)
            dim1 = node.attrs.get("dim1", -1)
            tensors[nid] = inp_tensors[0].transpose(dim0, dim1)

        elif node.op == OpType.FUSED_ADD_RELU:
            if len(inp_tensors) >= 2:
                tensors[nid] = F.relu(inp_tensors[0] + inp_tensors[1])
            else:
                tensors[nid] = F.relu(inp_tensors[0])

        elif node.op == OpType.FUSED_BIAS_GELU:
            if len(inp_tensors) >= 2:
                tensors[nid] = F.gelu(inp_tensors[0] + inp_tensors[1])
            else:
                tensors[nid] = F.gelu(inp_tensors[0])

        elif node.op == OpType.FUSED_RESIDUAL_RMSNORM:
            if len(inp_tensors) >= 2:
                residual = inp_tensors[0] + inp_tensors[1]
                tensors[nid] = _rmsnorm(residual)
            else:
                tensors[nid] = _rmsnorm(inp_tensors[0])

        else:
            raise ValueError(f"Unknown op type: {node.op}")

    # Build output dict
    output: dict[str, torch.Tensor] = {}
    for nid in graph.outputs:
        if nid in tensors and nid in graph.nodes:
            output[graph.nodes[nid].name] = tensors[nid]

    return output
