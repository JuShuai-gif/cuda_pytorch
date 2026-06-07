#!/usr/bin/env python3
"""
MIT 6.5940 Lecture 19: Distributed Training Simulation

Topics covered:
  - Simulate Data Parallelism: split data across "nodes", sync gradients
    with allreduce
  - Simulate ZeRO stages: show memory reduction at stage 1/2/3
  - Calculate: GPU memory needed per node for different parallelism
    strategies
  - Demonstrate communication overhead calculation

All computation runs on CPU.  No GPU required.
"""

from __future__ import annotations

import math
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


# ===========================================================================
# 1. Model definition (simulated for distributed training)
# ===========================================================================


class ToyModel(nn.Module):
    """A toy model with configurable size for distributed training simulation."""

    def __init__(self, hidden_dim: int = 256, num_layers: int = 4):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = hidden_dim * (2 ** min(i, 3))
            out_dim = hidden_dim * (2 ** min(i + 1, 4))
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers, nn.Linear(out_dim, 10))
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ===========================================================================
# 2. Data Parallelism Simulation
# ===========================================================================


def simulate_allreduce(
    grad_size_bytes: int, num_nodes: int, bandwidth_gbps: float
) -> float:
    """Simulate AllReduce communication time using the Ring-AllReduce algorithm.

    Ring-AllReduce: each node sends (N-1)/N of data to neighbor, total
    2*(N-1)/N rounds.  Communication volume per node = 2*(N-1)/N * data_size.

    Args:
        grad_size_bytes: total gradient size (bytes) across all parameters
        num_nodes: number of participating nodes
        bandwidth_gbps: inter-node bandwidth in Gbps

    Returns:
        Communication time in seconds.
    """
    data_per_node = grad_size_bytes * 2.0 * (num_nodes - 1) / num_nodes
    bandwidth_byps = bandwidth_gbps * 1e9 / 8  # bytes per second
    return data_per_node / bandwidth_byps


def simulate_data_parallelism(
    model: ToyModel,
    batch_size: int,
    num_nodes: int,
    bandwidth_gbps: float = 100.0,
) -> Dict[str, float]:
    """Simulate Data Parallel training step.

    Returns a dictionary with memory and timing metrics.
    """
    grad_size = sum(p.numel() for p in model.parameters()) * 4  # float32
    param_size = grad_size  # same size for parameters

    # Per-node memory
    optimizer_state_size = param_size  # Adam: one extra copy of params
    mem_params = param_size
    mem_grads = grad_size
    mem_opt = optimizer_state_size * 2  # m + v for Adam
    mem_activations = param_size * 0.5  # rough estimate

    total_mem_per_node = mem_params + mem_grads + mem_opt + mem_activations
    comm_time = simulate_allreduce(grad_size, num_nodes, bandwidth_gbps)

    return {
        "strategy": "Data Parallelism (DP)",
        "num_nodes": float(num_nodes),
        "per_node_mem_gb": total_mem_per_node / 1e9,
        "comm_time_ms": comm_time * 1000,
        "comm_volume_gb": grad_size * 2 * (num_nodes - 1) / num_nodes / 1e9,
    }


# ===========================================================================
# 3. ZeRO Stages Simulation
# ===========================================================================


def simulate_zero_stages(model: ToyModel, num_nodes: int) -> List[Dict[str, float]]:
    """Simulate memory usage under ZeRO stage 1/2/3.

    Reference: Rajbhandari et al., "ZeRO: Memory Optimizations Toward
    Training Trillion Parameter Models", SC 2020.

    Key concepts:
      - Stage 1 (P_os): Partition optimizer states across nodes.
      - Stage 2 (P_os + P_g): Additionally partition gradients.
      - Stage 3 (P_os + P_g + P_p): Additionally partition parameters.
    """
    param_size = sum(p.numel() for p in model.parameters()) * 4  # bytes
    grad_size = param_size
    opt_size = param_size * 2  # Adam: m + v
    act_size = param_size * 0.5  # activation estimate
    N = float(num_nodes)

    results = []

    # Baseline (no ZeRO)
    baseline_mem = param_size + grad_size + opt_size + act_size
    results.append(
        {
            "stage": "Baseline (no ZeRO)",
            "per_node_mem_gb": baseline_mem / 1e9,
            "reduction": "0%",
            "comm_overhead": "baseline",
        }
    )

    # ZeRO-1: partition optimizer states
    z1_mem = param_size + grad_size + opt_size / N + act_size
    z1_reduction = (1.0 - z1_mem / baseline_mem) * 100
    results.append(
        {
            "stage": "ZeRO-1 (P_os partitioned)",
            "per_node_mem_gb": z1_mem / 1e9,
            "reduction": f"{z1_reduction:.1f}%",
            "comm_overhead": "same as DP",
        }
    )

    # ZeRO-2: partition optimizer states + gradients
    z2_mem = param_size + grad_size / N + opt_size / N + act_size
    z2_reduction = (1.0 - z2_mem / baseline_mem) * 100
    results.append(
        {
            "stage": "ZeRO-2 (P_os + P_g partitioned)",
            "per_node_mem_gb": z2_mem / 1e9,
            "reduction": f"{z2_reduction:.1f}%",
            "comm_overhead": "same as DP + reduce-scatter",
        }
    )

    # ZeRO-3: partition everything
    z3_mem = param_size / N + grad_size / N + opt_size / N + act_size
    z3_reduction = (1.0 - z3_mem / baseline_mem) * 100
    results.append(
        {
            "stage": "ZeRO-3 (P_os + P_g + P_p partitioned)",
            "per_node_mem_gb": z3_mem / 1e9,
            "reduction": f"{z3_reduction:.1f}%",
            "comm_overhead": "increased: param all-gather per layer",
        }
    )

    return results


# ===========================================================================
# 4. Communication Overhead Calculator
# ===========================================================================


def communication_overhead_calculator(
    model_size_gb: float,
    num_nodes: int,
    world_size: int,
    bandwidth_gbps: float = 100.0,
) -> Dict[str, float]:
    """Calculate communication overhead for distributed strategies.

    Args:
        model_size_gb: size of model parameters in GB
        num_nodes: number of physical nodes
        world_size: total number of GPUs/processes
        bandwidth_gbps: inter-node bandwidth

    Returns:
        Dictionary of communication metrics.
    """
    model_bytes = model_size_gb * 1e9
    grad_bytes = model_bytes  # same as params in FP32
    opt_bytes = model_bytes * 2  # Adam states
    bw = bandwidth_gbps * 1e9 / 8  # bytes/sec

    # Data Parallel allreduce: 2*(N-1)/N * grad
    dp_volume = grad_bytes * 2.0 * (world_size - 1) / world_size
    dp_time = dp_volume / bw

    # ZeRO-1: same as DP
    z1_volume = dp_volume
    z1_time = dp_time

    # ZeRO-2: reduce-scatter for grads: (N-1)/N * grad
    z2_volume = grad_bytes * (world_size - 1) / world_size
    z2_time = z2_volume / bw

    # ZeRO-3: all-gather params each layer (approximately 1x param size per layer)
    z3_volume = model_bytes  # all-gather all params once
    z3_time = z3_volume / bw

    return {
        "dp_volume_gb": dp_volume / 1e9,
        "dp_time_ms": dp_time * 1000,
        "z1_volume_gb": z1_volume / 1e9,
        "z1_time_ms": z1_time * 1000,
        "z2_volume_gb": z2_volume / 1e9,
        "z2_time_ms": z2_time * 1000,
        "z3_volume_gb": z3_volume / 1e9,
        "z3_time_ms": z3_time * 1000,
    }


# ===========================================================================
# 5. GPU Memory Calculator by Parallelism Strategy
# ===========================================================================


def calculate_gpu_memory(
    model_params: int,
    data_size: int,
    num_devices: int,
    strategy: str,
) -> Dict[str, float]:
    """Calculate per-GPU memory usage for different parallelism strategies.

    Args:
        model_params: number of model parameters
        data_size: batch size * sequence_length * hidden_dim (activations proxy)
        num_devices: number of GPUs/devices
        strategy: "dp", "pp", "tp", or "dp+pp+tp"

    Returns:
        Memory breakdown in GB.
    """
    P_fp32 = model_params * 4  # parameter bytes
    G_fp32 = P_fp32  # gradient bytes
    O_fp32 = P_fp32 * 2  # optimizer (Adam)
    A_fp32 = model_params * 4 * 0.3  # activation estimate
    N = float(num_devices)

    if strategy == "dp":
        # All replicas hold full model
        mem = (P_fp32 + G_fp32 + O_fp32 + A_fp32) / 1e9
        desc = "Full model replicated on each device"
    elif strategy == "pp":
        # Pipeline Parallelism: each device holds 1/N of layers
        mem = (P_fp32 + G_fp32 + O_fp32 + A_fp32) / N / 1e9
        desc = "Model split across devices by layers"
    elif strategy == "tp":
        # Tensor Parallelism: each device holds 1/N of each layer
        mem = (P_fp32 + G_fp32 + O_fp32) / N / 1e9 + A_fp32 / 1e9
        desc = "Each layer's weights split across devices"
    elif strategy == "dp+pp+tp":
        # Hybrid: split across dimensions
        dp_size = max(1, int(N ** (1 / 3)))
        pp_size = max(1, int(N ** (1 / 3)))
        tp_size = max(1, N // (dp_size * pp_size))
        total_factor = dp_size * pp_size * tp_size
        mem = (P_fp32 + G_fp32 + O_fp32) / total_factor / 1e9 + A_fp32 / (
            pp_size * tp_size
        ) / 1e9
        desc = f"Hybrid DP={dp_size} PP={pp_size} TP={tp_size}"
    else:
        mem = (P_fp32 + G_fp32 + O_fp32 + A_fp32) / 1e9
        desc = "Unknown strategy"

    return {"strategy": strategy, "mem_gb": mem, "description": desc}


# ===========================================================================
# 6. Main demonstration
# ===========================================================================


def main() -> None:
    print("=" * 72)
    print("MIT 6.5940 Lecture 19: Distributed Training Simulation")
    print("=" * 72)

    # ---------- Model setup ----------
    print("\n--- 1. Model Setup ---")
    model = ToyModel(hidden_dim=256, num_layers=4)
    param_count = sum(p.numel() for p in model.parameters())
    param_mb = param_count * 4 / 1e6
    print(f"  Model parameters: {param_count:,}")
    print(f"  Model size (FP32): {param_mb:.1f} MB")

    # ---------- Data Parallelism ----------
    print("\n--- 2. Data Parallelism Simulation ---")
    for n in [1, 2, 4, 8]:
        metrics = simulate_data_parallelism(model, batch_size=64, num_nodes=n)
        print(
            f"  Nodes={n}: mem={metrics['per_node_mem_gb']:.3f} GB/node, "
            f"comm={metrics['comm_time_ms']:.2f} ms, "
            f"volume={metrics['comm_volume_gb']:.3f} GB"
        )

    # ---------- ZeRO Stages ----------
    print("\n--- 3. ZeRO Stages Memory Comparison ---")
    zero_results = simulate_zero_stages(model, num_nodes=4)
    print(
        f"  {'Stage':<38} {'Mem/Node(GB)':>14} {'Reduction':>10} {'Comm Overhead':>20}"
    )
    print(f"  {'-' * 82}")
    for r in zero_results:
        print(
            f"  {r['stage']:<38} {r['per_node_mem_gb']:>14.3f} {r['reduction']:>10} "
            f"{r['comm_overhead']:>20}"
        )

    # ---------- Communication Overhead ----------
    print("\n--- 4. Communication Overhead Calculator ---")
    model_gb = param_count * 4 / 1e9
    comm = communication_overhead_calculator(model_gb, num_nodes=4, world_size=8)
    print(f"  Strategy        Volume(GB)  Time(ms)")
    print(f"  {'-' * 45}")
    print(
        f"  Data Parallel   {comm['dp_volume_gb']:>10.3f}  {comm['dp_time_ms']:>7.1f}"
    )
    print(
        f"  ZeRO-1          {comm['z1_volume_gb']:>10.3f}  {comm['z1_time_ms']:>7.1f}"
    )
    print(
        f"  ZeRO-2          {comm['z2_volume_gb']:>10.3f}  {comm['z2_time_ms']:>7.1f}"
    )
    print(
        f"  ZeRO-3          {comm['z3_volume_gb']:>10.3f}  {comm['z3_time_ms']:>7.1f}"
    )

    # ---------- GPU Memory by Strategy ----------
    print("\n--- 5. Per-GPU Memory by Parallelism Strategy ---")
    for strategy in ["dp", "pp", "tp", "dp+pp+tp"]:
        mem = calculate_gpu_memory(
            param_count, data_size=64 * 128 * 256, num_devices=8, strategy=strategy
        )
        print(f"  {strategy:<12}: {mem['mem_gb']:.3f} GB  ({mem['description']})")

    # ---------- Summary ----------
    print("\n--- 6. Summary ---")
    print("  Key takeaways:")
    print("    - Data Parallelism: simplest, but memory scales with model size")
    print("    - ZeRO-1: 4x less optimizer memory (with 4 nodes)")
    print("    - ZeRO-2: additionally 4x less gradient memory")
    print("    - ZeRO-3: additionally 4x less parameter memory  (near-linear scaling)")
    print("    - Communication cost increases with ZeRO stages (more collective ops)")
    print("    - Hybrid parallelism (DP+PP+TP) needed for very large models")

    print("\nDone. All computations on CPU.\n")


if __name__ == "__main__":
    main()
