#!/usr/bin/env python3
"""
Demo script: build and optimize real transformer-like computation graphs.

Shows the full optimization pipeline:
1. Build a transformer block graph (QKV, activation, residual, RMSNorm)
2. Print original graph (DOT format)
3. Run optimization passes
4. Print optimized graph
5. Execute both original and optimized graphs
6. Verify results match
7. Show node count reduction

Run: python 09_graph_optimization/graph_demo.py
"""

from __future__ import annotations

import sys

import torch

from executor import execute_graph
from ir import DType, Graph, OpType, TensorShape
from passes import optimize_graph


def build_transformer_block_graph() -> Graph:
    """Build a graph representing part of a transformer block.

    Operations:
      x = input (hidden_dim)
      h = matmul(x, w_qkv)        # QKV projection (simplified as single matmul)
      h = add(h, bias)            # bias add
      h = gelu(h)                 # activation
      h2 = matmul(h, w_o)         # output projection
      h2 = add(h2, bias2)         # bias add
      h2 = add(x, h2)             # residual connection
      out = rmsnorm(h2)           # RMSNorm
    """
    g = Graph()

    # Input placeholders
    x_id = g.add_node(OpType.CONSTANT, [], attrs={}, name="input_x")
    g.inputs.append(x_id)

    w_qkv_id = g.add_node(OpType.CONSTANT, [], attrs={}, name="weight_qkv")
    g.inputs.append(w_qkv_id)

    bias_id = g.add_node(OpType.CONSTANT, [], attrs={}, name="bias")
    g.inputs.append(bias_id)

    w_o_id = g.add_node(OpType.CONSTANT, [], attrs={}, name="weight_o")
    g.inputs.append(w_o_id)

    bias2_id = g.add_node(OpType.CONSTANT, [], attrs={}, name="bias2")
    g.inputs.append(bias2_id)

    # QKV projection: h = matmul(x, w_qkv)
    qkv_id = g.add_node(
        OpType.MATMUL,
        [x_id, w_qkv_id],
        attrs={},
        name="qkv_projection",
    )

    # Add bias: h = add(h, bias)
    add1_id = g.add_node(
        OpType.ADD,
        [qkv_id, bias_id],
        attrs={},
        name="add_bias1",
    )

    # GELU activation: h = gelu(h)
    gelu_id = g.add_node(
        OpType.GELU,
        [add1_id],
        attrs={},
        name="gelu_activation",
    )

    # Output projection: h2 = matmul(h, w_o)
    out_proj_id = g.add_node(
        OpType.MATMUL,
        [gelu_id, w_o_id],
        attrs={},
        name="output_projection",
    )

    # Add bias2: h2 = add(h2, bias2)
    add2_id = g.add_node(
        OpType.ADD,
        [out_proj_id, bias2_id],
        attrs={},
        name="add_bias2",
    )

    # Residual: h2 = add(x, h2)  -- this creates an ADD+ADD+RMSNORM opportunity
    residual_id = g.add_node(
        OpType.ADD,
        [x_id, add2_id],
        attrs={},
        name="residual_add",
    )

    # RMSNorm: out = rmsnorm(h2)
    rmsnorm_id = g.add_node(
        OpType.RMSNORM,
        [residual_id],
        attrs={},
        name="final_rmsnorm",
    )

    g.outputs.append(rmsnorm_id)

    return g


def demo_optimization_pipeline() -> None:
    """Run the full optimization pipeline and show results."""
    print("=" * 70)
    print("  GRAPH OPTIMIZATION DEMO")
    print("=" * 70)

    graph = build_transformer_block_graph()

    print(f"\n[1] Original graph: {len(graph.nodes)} nodes")
    print("-" * 40)
    print(graph.to_dot())
    print("-" * 40)

    optimized = optimize_graph(graph)

    print(f"\n[2] Optimized graph: {len(optimized.nodes)} nodes")
    print(
        f"    Reduction: {len(graph.nodes) - len(optimized.nodes)} nodes removed "
        f"({(1 - len(optimized.nodes) / len(graph.nodes)) * 100:.1f}%)"
    )
    print("-" * 40)
    print(optimized.to_dot())
    print("-" * 40)

    # Create tensors for execution
    hidden_dim = 128
    intermediate_dim = 384  # 3x for QKV

    x = torch.randn(1, hidden_dim)
    w_qkv = torch.randn(hidden_dim, intermediate_dim)
    bias = torch.randn(intermediate_dim)
    w_o = torch.randn(intermediate_dim, hidden_dim)
    bias2 = torch.randn(hidden_dim)

    inputs = {
        "input_x": x,
        "weight_qkv": w_qkv,
        "bias": bias,
        "weight_o": w_o,
        "bias2": bias2,
    }

    print(f"\n[3] Executing original graph...")
    original_output = execute_graph(graph, inputs)

    print(f"[4] Executing optimized graph...")
    optimized_output = execute_graph(optimized, inputs)

    print(f"\n[5] Verification:")
    for name in original_output:
        if name in optimized_output or True:
            for opt_name, opt_val in optimized_output.items():
                orig_val = original_output[name]
                # Find matching key (names may differ after optimization)
                if orig_val.shape == opt_val.shape:
                    diff = (orig_val - opt_val).abs().max().item()
                    print(f"    {name} vs {opt_name}: max diff = {diff:.2e}")
                break
            break

    # More thorough comparison
    orig_vals = list(original_output.values())
    opt_vals = list(optimized_output.values())
    if orig_vals and opt_vals:
        # Find matching shapes
        for ov in orig_vals:
            for opv in opt_vals:
                if ov.shape == opv.shape:
                    diff = (ov - opv).abs().max().item()
                    print(f"    Output match: max diff = {diff:.2e}")
                    if diff > 1e-3:
                        print(f"    WARNING: Large numerical difference detected!")
                    else:
                        print(f"    PASS: Results match within tolerance")
                    break
            break

    print(f"\n[6] Optimization analysis:")
    # Count ops by type in original
    orig_ops: dict[str, int] = {}
    for node in graph.nodes.values():
        orig_ops[node.op.value] = orig_ops.get(node.op.value, 0) + 1

    # Count ops by type in optimized
    opt_ops: dict[str, int] = {}
    for node in optimized.nodes.values():
        opt_ops[node.op.value] = opt_ops.get(node.op.value, 0) + 1

    print(f"    {'Op Type':<30} {'Original':>10} {'Optimized':>10} {'Delta':>8}")
    print(f"    {'-' * 30} {'-' * 10} {'-' * 10} {'-' * 8}")
    all_ops = sorted(set(list(orig_ops.keys()) + list(opt_ops.keys())))
    for op in all_ops:
        o_count = orig_ops.get(op, 0)
        n_count = opt_ops.get(op, 0)
        delta = n_count - o_count
        print(f"    {op:<30} {o_count:>10} {n_count:>10} {delta:>+8}")

    print(f"\n[7] Graph saved to DOT format:")
    print(f"    Original DOT:\n{graph.to_dot()}")
    print(f"\n    Optimized DOT:\n{optimized.to_dot()}")


if __name__ == "__main__":
    demo_optimization_pipeline()
