#!/usr/bin/env python3
"""
Benchmark graph optimization vs eager execution.

Builds larger graphs with repeated patterns (simulating a deep network).
Measures: execution time, node count reduction, memory savings.
Shows speedup from fusion patterns, effect of CSE on duplicated subgraphs.

Run: python 09_graph_optimization/benchmark_graph_optimization.py
"""

from __future__ import annotations

import sys
import time
from typing import Callable, Sequence

import torch

from executor import execute_graph
from ir import DType, Graph, OpType
from passes import optimize_graph


def _build_deep_network(num_layers: int, hidden_dim: int, with_redundancy: bool = False) -> Graph:
    """Build a deep network graph with repeated transformer-like blocks.

    Each block: x = relu(linear(x) + bias)
    If with_redundancy: duplicate every other block to simulate CSE opportunity.
    """
    g = Graph()

    x_id = g.add_node(OpType.CONSTANT, [], name="input_x")
    g.inputs.append(x_id)

    weight_ids: Sequence[int] = []
    bias_ids: Sequence[int] = []
    for i in range(num_layers):
        w_id = g.add_node(OpType.CONSTANT, [], name=f"w_{i}")
        b_id = g.add_node(OpType.CONSTANT, [], name=f"b_{i}")
        g.inputs.extend([w_id, b_id])
        weight_ids.append(w_id)
        bias_ids.append(b_id)

    current = x_id
    for i in range(num_layers):
        # linear: h = matmul(x, w)
        matmul_id = g.add_node(OpType.MATMUL, [current, weight_ids[i]], name=f"matmul_{i}")

        # bias: h = add(h, bias)
        add_id = g.add_node(OpType.ADD, [matmul_id, bias_ids[i]], name=f"add_bias_{i}")

        # activation: h = relu(h) or gelu on even layers
        if i % 2 == 0:
            act_id = g.add_node(OpType.RELU, [add_id], name=f"relu_{i}")
        else:
            act_id = g.add_node(OpType.GELU, [add_id], name=f"gelu_{i}")

        current = act_id

        # Optionally add duplicate computation (for CSE testing)
        if with_redundancy and i > 0 and i % 3 == 0:
            dup_matmul = g.add_node(OpType.MATMUL, [current, weight_ids[i]], name=f"matmul_{i}_dup")
            dup_add = g.add_node(OpType.ADD, [dup_matmul, bias_ids[i]], name=f"add_bias_{i}_dup")
            if i % 2 == 0:
                dup_act = g.add_node(OpType.RELU, [dup_add], name=f"relu_{i}_dup")
            else:
                dup_act = g.add_node(OpType.GELU, [dup_add], name=f"gelu_{i}_dup")
            # Add to output (makes it "used")
            g.outputs.append(dup_act)

    g.outputs.append(current)

    return g


def _time_execution(fn: Callable, warmup: int = 5, repeats: int = 20) -> float:
    """Time execution in milliseconds."""
    for _ in range(warmup):
        fn()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        total += (t1 - t0) * 1000.0

    return total / repeats


def _build_and_execute(
    num_layers: int,
    hidden_dim: int,
    with_redundancy: bool,
) -> tuple[dict, Graph, Graph]:
    """Build, optimize, execute, and return results."""
    graph = _build_deep_network(num_layers, hidden_dim, with_redundancy)
    optimized = optimize_graph(graph)

    # Create tensors
    inputs: dict[str, torch.Tensor] = {}
    for nid in graph.inputs:
        if nid in graph.nodes:
            node = graph.nodes[nid]
            if "w_" in node.name or node.name.startswith("w_"):
                inputs[node.name] = torch.randn(hidden_dim, hidden_dim)
            elif "b_" in node.name or node.name.startswith("b_"):
                inputs[node.name] = torch.randn(hidden_dim)
            elif "input" in node.name.lower():
                inputs[node.name] = torch.randn(1, hidden_dim)

    orig_out = execute_graph(graph, inputs)
    opt_out = execute_graph(optimized, inputs)

    return orig_out, opt_out, graph, optimized


def bench_layers_sweep() -> None:
    """Sweep number of layers and measure optimization impact."""
    print(f"\n{'=' * 60}")
    print("  LAYER COUNT SWEEP")
    print(f"{'=' * 60}")

    hidden_dim = 128
    layer_counts = [4, 8, 16, 32, 64]

    print(
        f"\n  {'Layers':>8}  {'Orig Nodes':>12}  {'Opt Nodes':>12}  {'Reduction':>10}  {'Orig(ms)':>10}  {'Opt(ms)':>10}"
    )
    print(f"  {'-' * 8}  {'-' * 12}  {'-' * 12}  {'-' * 10}  {'-' * 10}  {'-' * 10}")

    for nl in layer_counts:
        graph = _build_deep_network(nl, hidden_dim, with_redundancy=False)
        optimized = optimize_graph(graph)

        n_orig = len(graph.nodes)
        n_opt = len(optimized.nodes)
        reduction = (1.0 - n_opt / n_orig) * 100.0 if n_orig > 0 else 0.0

        # Time execution on both (CPU tensors to avoid CUDA overhead for graph exec)
        # We use CPU since the executor runs on CPU
        inputs: dict[str, torch.Tensor] = {}
        for nid in graph.inputs:
            if nid in graph.nodes:
                node = graph.nodes[nid]
                if "w_" in node.name or node.name.startswith("w_"):
                    inputs[node.name] = torch.randn(hidden_dim, hidden_dim)
                elif "b_" in node.name or node.name.startswith("b_"):
                    inputs[node.name] = torch.randn(hidden_dim)
                elif "input" in node.name.lower():
                    inputs[node.name] = torch.randn(1, hidden_dim)

        t_orig = _time_execution(lambda: execute_graph(graph, inputs))
        t_opt = _time_execution(lambda: execute_graph(optimized, inputs))

        print(
            f"  {nl:>8}  {n_orig:>12}  {n_opt:>12}  {reduction:>9.1f}%"
            f"  {t_orig:>9.3f} ms  {t_opt:>9.3f} ms"
        )


def bench_cse_impact() -> None:
    """Measure how CSE reduces node count in redundant graphs."""
    print(f"\n{'=' * 60}")
    print("  CSE IMPACT (Graphs with Redundancy)")
    print(f"{'=' * 60}")

    hidden_dim = 128
    layer_counts = [8, 16, 32]

    print(
        f"\n  {'Layers':>8}  {'Orig Nodes':>12}  {'Opt Nodes':>12}  {'Dup Removed':>12}  {'Reduction':>10}"
    )
    print(f"  {'-' * 8}  {'-' * 12}  {'-' * 12}  {'-' * 12}  {'-' * 10}")

    for nl in layer_counts:
        graph = _build_deep_network(nl, hidden_dim, with_redundancy=True)
        optimized = optimize_graph(graph)

        n_orig = len(graph.nodes)
        n_opt = len(optimized.nodes)
        reduction = (1.0 - n_opt / n_orig) * 100.0 if n_orig > 0 else 0.0

        print(f"  {nl:>8}  {n_orig:>12}  {n_opt:>12}  {n_orig - n_opt:>12}  {reduction:>9.1f}%")

    # Also show comparisons for non-redundant
    print(f"\n  Without redundancy (for comparison):")
    print(f"  {'Layers':>8}  {'Orig Nodes':>12}  {'Opt Nodes':>12}  {'Reduction':>10}")
    print(f"  {'-' * 8}  {'-' * 12}  {'-' * 12}  {'-' * 10}")

    for nl in layer_counts:
        graph = _build_deep_network(nl, hidden_dim, with_redundancy=False)
        optimized = optimize_graph(graph)

        n_orig = len(graph.nodes)
        n_opt = len(optimized.nodes)
        reduction = (1.0 - n_opt / n_orig) * 100.0 if n_orig > 0 else 0.0

        print(f"  {nl:>8}  {n_orig:>12}  {n_opt:>12}  {reduction:>9.1f}%")


def bench_fusion_speedup() -> None:
    """Measure speedup from pattern fusion in a graph with many fusible patterns."""
    print(f"\n{'=' * 60}")
    print("  FUSION SPEEDUP")
    print(f"{'=' * 60}")

    hidden_dim = 64

    # Build a graph with many ADD+RELU patterns
    g = Graph()
    x = g.add_node(OpType.CONSTANT, [], name="x")
    g.inputs.append(x)

    current = x
    num_fusible = 50

    for i in range(num_fusible):
        bias = g.add_node(OpType.CONSTANT, [], name=f"bias_{i}")
        g.inputs.append(bias)

        add_node = g.add_node(OpType.ADD, [current, bias], name=f"add_{i}")
        relu_node = g.add_node(OpType.RELU, [add_node], name=f"relu_{i}")
        current = relu_node

    g.outputs.append(current)

    inputs: dict[str, torch.Tensor] = {}
    inputs["x"] = torch.randn(1, hidden_dim)
    for i in range(num_fusible):
        inputs[f"bias_{i}"] = torch.randn(hidden_dim)

    n_orig = len(g.nodes)
    t_orig = _time_execution(lambda: execute_graph(g, inputs), warmup=3, repeats=10)

    optimized = optimize_graph(g)
    n_opt = len(optimized.nodes)
    t_opt = _time_execution(lambda: execute_graph(optimized, inputs), warmup=3, repeats=10)

    print(f"\n  Fusible patterns: {num_fusible}")
    print(f"  Original nodes: {n_orig}")
    print(f"  Optimized nodes: {n_opt}")
    print(f"  Node reduction: {n_orig - n_opt} ({(1 - n_opt / n_orig) * 100:.1f}%)")
    print(f"  Original time: {t_orig:.3f} ms")
    print(f"  Optimized time: {t_opt:.3f} ms")
    speedup = t_orig / t_opt if t_opt > 0 else 0
    print(f"  Speedup: {speedup:.2f}x")

    # Count fused ops
    fused_count = sum(1 for n in optimized.nodes.values() if "fused" in n.op.value)
    print(f"  Fused ops detected: {fused_count}")


def bench_memory_savings() -> None:
    """Estimate memory savings from reduced intermediate tensors."""
    print(f"\n{'=' * 60}")
    print("  MEMORY SAVINGS ESTIMATION")
    print(f"{'=' * 60}")

    hidden_dim = 512
    num_patterns = 20

    # Original: 2 intermediates per pattern (add output + relu input = same tensor)
    # Fused: 0 intermediates per pattern
    # Each intermediate is hidden_dim * 4 bytes (fp32)

    elements_per_tensor = hidden_dim
    bytes_per_tensor = elements_per_tensor * 4  # fp32

    # Original graph has: inputs + (num_patterns * 2 intermediates) + 1 output
    # Fused version has: inputs + 1 output
    orig_intermediates = num_patterns * 2
    fused_intermediates = 0

    orig_mem = orig_intermediates * bytes_per_tensor
    fused_mem = fused_intermediates * bytes_per_tensor
    saved_mem = orig_mem - fused_mem
    saved_mb = saved_mem / (1024 * 1024)

    print(f"\n  Hidden dim: {hidden_dim}")
    print(f"  Number of fusible patterns: {num_patterns}")
    print(f"  dtype: float32 (4 bytes)")
    print(f"\n  Original intermediates: {orig_intermediates}")
    print(f"  Fused intermediates: {fused_intermediates}")
    print(f"  Peak memory saved: {saved_mb:.2f} MB")

    # For larger sizes
    large_dim = 4096
    large_elems = large_dim
    large_bytes = large_elems * 4
    large_orig = num_patterns * 2 * large_bytes
    large_saved_mb = large_orig / (1024 * 1024)
    print(f"\n  For hidden_dim={large_dim} (typical LLM):")
    print(f"  Peak memory saved: {large_saved_mb:.2f} MB")

    # Multiple layers
    num_layers = 32  # typical LLM
    total_saved = large_saved_mb * num_layers
    print(f"  Across {num_layers} layers: {total_saved:.1f} MB ({total_saved / 1024:.1f} GB)")


def bench_different_hidden_sizes() -> None:
    """Benchmark graph optimization at different hidden sizes."""
    print(f"\n{'=' * 60}")
    print("  HIDDEN SIZE SWEEP")
    print(f"{'=' * 60}")

    hidden_sizes = [32, 64, 128, 256, 512, 1024]
    num_layers = 8

    print(
        f"\n  {'Hidden':>8}  {'Orig Nodes':>12}  {'Opt Nodes':>12}  {'Reduction':>10}  {'Orig(ms)':>10}  {'Opt(ms)':>10}  {'Speedup':>8}"
    )
    print(f"  {'-' * 8}  {'-' * 12}  {'-' * 12}  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 8}")

    for hd in hidden_sizes:
        graph = _build_deep_network(num_layers, hd, with_redundancy=False)
        optimized = optimize_graph(graph)

        inputs: dict[str, torch.Tensor] = {}
        for nid in graph.inputs:
            if nid in graph.nodes:
                node = graph.nodes[nid]
                if "w_" in node.name or node.name.startswith("w_"):
                    inputs[node.name] = torch.randn(hd, hd)
                elif "b_" in node.name or node.name.startswith("b_"):
                    inputs[node.name] = torch.randn(hd)
                elif "input" in node.name.lower():
                    inputs[node.name] = torch.randn(1, hd)

        t_orig = _time_execution(lambda: execute_graph(graph, inputs), warmup=3, repeats=10)
        t_opt = _time_execution(lambda: execute_graph(optimized, inputs), warmup=3, repeats=10)

        n_orig = len(graph.nodes)
        n_opt = len(optimized.nodes)
        reduction = (1.0 - n_opt / n_orig) * 100.0 if n_orig > 0 else 0.0
        speedup = t_orig / t_opt if t_opt > 0 else 0.0

        print(
            f"  {hd:>8}  {n_orig:>12}  {n_opt:>12}  {reduction:>9.1f}%"
            f"  {t_orig:>9.3f} ms  {t_opt:>9.3f} ms  {speedup:>7.2f}x"
        )


def main() -> None:
    print("=" * 70)
    print("  GRAPH OPTIMIZATION BENCHMARKS")
    print("=" * 70)
    print(f"  PyTorch: {torch.__version__}")

    bench_layers_sweep()
    bench_cse_impact()
    bench_fusion_speedup()
    bench_memory_savings()
    bench_different_hidden_sizes()

    print(f"\n{'=' * 70}")
    print("  BENCHMARKS COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
