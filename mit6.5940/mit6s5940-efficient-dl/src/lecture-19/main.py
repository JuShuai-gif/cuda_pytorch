#!/usr/bin/env python3
"""
MIT 6.5940 第19讲：分布式训练模拟

涵盖主题：
  - 模拟数据并行：将数据拆分到多个"节点"，通过allreduce同步梯度
  - 模拟ZeRO各阶段：展示阶段1/2/3的内存减少效果
  - 计算：不同并行策略下每个节点所需的GPU内存
  - 演示通信开销的计算方法

所有计算均在CPU上运行，无需GPU。
"""

from __future__ import annotations

import math
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


# ===========================================================================
# 1. 模型定义（用于分布式训练模拟）
# ===========================================================================


class ToyModel(nn.Module):
    """一个可配置大小的玩具模型，用于分布式训练模拟。"""

    def __init__(self, hidden_dim: int = 256, num_layers: int = 4):
        super().__init__()
        layers = []
        for i in range(num_layers):
            # 每层的输入输出维度逐层扩大，模拟真实模型的结构
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
# 2. 数据并行模拟
# ===========================================================================


def simulate_allreduce(
    grad_size_bytes: int, num_nodes: int, bandwidth_gbps: float
) -> float:
    """使用Ring-AllReduce算法模拟AllReduce通信时间。

    Ring-AllReduce：每个节点将(N-1)/N的数据发送给邻居，
    总共需要2*(N-1)/N轮。每个节点的通信量 = 2*(N-1)/N * 数据量。

    参数：
        grad_size_bytes: 所有参数的总梯度大小（字节）
        num_nodes: 参与的节点数量
        bandwidth_gbps: 节点间带宽，单位Gbps

    返回：
        通信时间（秒）。
    """
    # 每个节点需要发送和接收的数据量（发送+接收共2倍）
    data_per_node = grad_size_bytes * 2.0 * (num_nodes - 1) / num_nodes
    bandwidth_byps = bandwidth_gbps * 1e9 / 8  # 将Gbps转换为bytes/秒
    return data_per_node / bandwidth_byps


def simulate_data_parallelism(
    model: ToyModel,
    batch_size: int,
    num_nodes: int,
    bandwidth_gbps: float = 100.0,
) -> Dict[str, float]:
    """模拟数据并行训练的一个步骤。

    返回包含内存和时序指标的字典。
    """
    # 梯度大小：参数量 × 4字节（float32）
    grad_size = sum(p.numel() for p in model.parameters()) * 4  # float32
    param_size = grad_size  # 参数大小与梯度大小相同

    # ---------- 每个节点的内存估算 ----------
    optimizer_state_size = param_size  # Adam：需要额外一份参数的副本
    mem_params = param_size
    mem_grads = grad_size
    mem_opt = optimizer_state_size * 2  # Adam的m和v两个状态
    mem_activations = param_size * 0.5  # 粗略估计激活值内存

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
# 3. ZeRO各阶段模拟
# ===========================================================================


def simulate_zero_stages(model: ToyModel, num_nodes: int) -> List[Dict[str, float]]:
    """模拟ZeRO阶段1/2/3下的内存使用。

    参考：Rajbhandari等人，"ZeRO: Memory Optimizations Toward
    Training Trillion Parameter Models"，SC 2020。

    核心概念：
      - 阶段1（P_os）：将优化器状态分区到各个节点
      - 阶段2（P_os + P_g）：在此基础上再分区梯度
      - 阶段3（P_os + P_g + P_p）：在此基础上再分区参数
    """
    param_size = sum(p.numel() for p in model.parameters()) * 4  # 字节
    grad_size = param_size
    opt_size = param_size * 2  # Adam：m + v
    act_size = param_size * 0.5  # 激活值估算
    N = float(num_nodes)

    results = []

    # 基线（无ZeRO）：所有数据在每个节点都有一份完整副本
    baseline_mem = param_size + grad_size + opt_size + act_size
    results.append(
        {
            "stage": "Baseline (no ZeRO)",
            "per_node_mem_gb": baseline_mem / 1e9,
            "reduction": "0%",
            "comm_overhead": "baseline",
        }
    )

    # ZeRO-1：将优化器状态分区到N个节点
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

    # ZeRO-2：将优化器状态和梯度都分区
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

    # ZeRO-3：将优化器状态、梯度和参数全部分区
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
# 4. 通信开销计算器
# ===========================================================================


def communication_overhead_calculator(
    model_size_gb: float,
    num_nodes: int,
    world_size: int,
    bandwidth_gbps: float = 100.0,
) -> Dict[str, float]:
    """计算各种分布式策略的通信开销。

    参数：
        model_size_gb: 模型参数大小（GB）
        num_nodes: 物理节点数
        world_size: 总的GPU/进程数
        bandwidth_gbps: 节点间带宽

    返回：
        通信指标的字典。
    """
    model_bytes = model_size_gb * 1e9
    grad_bytes = model_bytes  # FP32下梯度大小与参数相同
    opt_bytes = model_bytes * 2  # Adam状态大小
    bw = bandwidth_gbps * 1e9 / 8  # 转换为bytes/秒

    # 数据并行allreduce通信量：2*(N-1)/N * 梯度大小
    dp_volume = grad_bytes * 2.0 * (world_size - 1) / world_size
    dp_time = dp_volume / bw

    # ZeRO-1：通信量与数据并行相同
    z1_volume = dp_volume
    z1_time = dp_time

    # ZeRO-2：梯度使用reduce-scatter，通信量为(N-1)/N * 梯度大小
    z2_volume = grad_bytes * (world_size - 1) / world_size
    z2_time = z2_volume / bw

    # ZeRO-3：每层需要all-gather参数，大约1倍参数大小的额外通信
    z3_volume = model_bytes  # 将所有参数all-gather一次
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
# 5. 按并行策略计算GPU内存
# ===========================================================================


def calculate_gpu_memory(
    model_params: int,
    data_size: int,
    num_devices: int,
    strategy: str,
) -> Dict[str, float]:
    """计算不同并行策略下每个GPU的内存使用。

    参数：
        model_params: 模型参数数量
        data_size: batch_size * sequence_length * hidden_dim（激活值代理变量）
        num_devices: GPU/设备数
        strategy: "dp"（数据并行）、"pp"（流水线并行）、"tp"（张量并行）或"dp+pp+tp"（混合并行）

    返回：
        内存拆解信息（GB）。
    """
    P_fp32 = model_params * 4  # 参数字节数
    G_fp32 = P_fp32  # 梯度字节数
    O_fp32 = P_fp32 * 2  # 优化器状态（Adam）
    A_fp32 = model_params * 4 * 0.3  # 激活值估算
    N = float(num_devices)

    if strategy == "dp":
        # 数据并行：所有副本持有完整模型
        mem = (P_fp32 + G_fp32 + O_fp32 + A_fp32) / 1e9
        desc = "Full model replicated on each device"
    elif strategy == "pp":
        # 流水线并行：每个设备持有1/N层
        mem = (P_fp32 + G_fp32 + O_fp32 + A_fp32) / N / 1e9
        desc = "Model split across devices by layers"
    elif strategy == "tp":
        # 张量并行：每个设备持有每层的1/N
        mem = (P_fp32 + G_fp32 + O_fp32) / N / 1e9 + A_fp32 / 1e9
        desc = "Each layer's weights split across devices"
    elif strategy == "dp+pp+tp":
        # 混合并行：跨多个维度拆分
        # 将设备均匀分配到三个并行维度上
        dp_size = max(1, int(N ** (1 / 3)))
        pp_size = max(1, int(N ** (1 / 3)))
        tp_size = max(1, N // (dp_size * pp_size))
        total_factor = dp_size * pp_size * tp_size
        # 参数/梯度/优化器按全部并行因子缩减，激活值按PP*TP缩减
        mem = (P_fp32 + G_fp32 + O_fp32) / total_factor / 1e9 + A_fp32 / (
            pp_size * tp_size
        ) / 1e9
        desc = f"Hybrid DP={dp_size} PP={pp_size} TP={tp_size}"
    else:
        mem = (P_fp32 + G_fp32 + O_fp32 + A_fp32) / 1e9
        desc = "Unknown strategy"

    return {"strategy": strategy, "mem_gb": mem, "description": desc}


# ===========================================================================
# 6. 主演示
# ===========================================================================


def main() -> None:
    print("=" * 72)
    print("MIT 6.5940 第19讲：分布式训练模拟")
    print("=" * 72)

    # ---------- 模型设置 ----------
    print("\n--- 1. 模型设置 ---")
    model = ToyModel(hidden_dim=256, num_layers=4)
    param_count = sum(p.numel() for p in model.parameters())
    param_mb = param_count * 4 / 1e6
    print(f"  模型参数数量: {param_count:,}")
    print(f"  模型大小 (FP32): {param_mb:.1f} MB")

    # ---------- 数据并行 ----------
    print("\n--- 2. 数据并行模拟 ---")
    for n in [1, 2, 4, 8]:
        metrics = simulate_data_parallelism(model, batch_size=64, num_nodes=n)
        print(
            f"  节点数={n}: 内存={metrics['per_node_mem_gb']:.3f} GB/节点, "
            f"通信时间={metrics['comm_time_ms']:.2f} ms, "
            f"通信量={metrics['comm_volume_gb']:.3f} GB"
        )

    # ---------- ZeRO各阶段 ----------
    print("\n--- 3. ZeRO各阶段内存对比 ---")
    zero_results = simulate_zero_stages(model, num_nodes=4)
    print(f"  {'阶段':<38} {'内存/节点(GB)':>14} {'减少比例':>10} {'通信开销':>20}")
    print(f"  {'-' * 82}")
    for r in zero_results:
        print(
            f"  {r['stage']:<38} {r['per_node_mem_gb']:>14.3f} {r['reduction']:>10} "
            f"{r['comm_overhead']:>20}"
        )

    # ---------- 通信开销 ----------
    print("\n--- 4. 通信开销计算器 ---")
    model_gb = param_count * 4 / 1e9
    comm = communication_overhead_calculator(model_gb, num_nodes=4, world_size=8)
    print(f"  策略              通信量(GB)  时间(ms)")
    print(f"  {'-' * 45}")
    print(
        f"  数据并行 (DP)     {comm['dp_volume_gb']:>10.3f}  {comm['dp_time_ms']:>7.1f}"
    )
    print(
        f"  ZeRO-1            {comm['z1_volume_gb']:>10.3f}  {comm['z1_time_ms']:>7.1f}"
    )
    print(
        f"  ZeRO-2            {comm['z2_volume_gb']:>10.3f}  {comm['z2_time_ms']:>7.1f}"
    )
    print(
        f"  ZeRO-3            {comm['z3_volume_gb']:>10.3f}  {comm['z3_time_ms']:>7.1f}"
    )

    # ---------- 按策略计算GPU内存 ----------
    print("\n--- 5. 不同并行策略下每GPU内存使用 ---")
    for strategy in ["dp", "pp", "tp", "dp+pp+tp"]:
        mem = calculate_gpu_memory(
            param_count, data_size=64 * 128 * 256, num_devices=8, strategy=strategy
        )
        print(f"  {strategy:<12}: {mem['mem_gb']:.3f} GB  ({mem['description']})")

    # ---------- 总结 ----------
    print("\n--- 6. 总结 ---")
    print("  核心要点：")
    print("    - 数据并行：最简单，但内存随模型大小线性增长")
    print("    - ZeRO-1：优化器内存减少4倍（以4节点为例）")
    print("    - ZeRO-2：在此基础上梯度内存再减少4倍")
    print("    - ZeRO-3：在此基础上参数内存再减少4倍（接近线性扩展）")
    print("    - 通信成本随ZeRO阶段增加而增加（更多的集合通信操作）")
    print("    - 超大规模模型需要混合并行（DP+PP+TP）")

    print("\n完成。所有计算均在CPU上运行。\n")


if __name__ == "__main__":
    main()
