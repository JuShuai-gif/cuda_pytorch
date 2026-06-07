"""
FSDP（完全分片数据并行）与 ZeRO 优化原理。

本文件通过带注释的代码和说明图展示了 FSDP/ZeRO 的核心概念。
它不直接使用 torch.distributed.fsdp，而是从概念层面解释 parameter sharding、
gradient sharding 和 optimizer state sharding 的工作原理。

ZeRO 各阶段概览：
- ZeRO-1：optimizer state 分区（每个 rank 存储 1/world_size 的 optimizer state）
- ZeRO-2：+ gradient 分区（每个 rank 存储 1/world_size 的 gradient）
- ZeRO-3：+ parameter 分区（每个 rank 存储 1/world_size 的 parameter）
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


# =============================================================================
# ZeRO Stage 1：Optimizer State 分区
# =============================================================================
#
# 在标准 DDP 中，每张 GPU 存储完整的 optimizer state（momentum、variance）。
# 使用 N 张 GPU 时，总 optimizer state 内存 = N * full_state_size。
#
# ZeRO-1 将 optimizer state 分区到各 GPU 上：
#   - GPU 0 存储 params[0:N/p] 的 state，GPU 1 存储 params[N/p:2N/p] 的 state，依此类推。
#   - 更新时，每张 GPU 计算其分区的更新，然后执行 all-gather。
#
# 内存节省：optimizer state 内存减少为原来的 1/N


def zero_stage_1_demo() -> None:
    """
    演示 ZeRO-1 的 optimizer state 分区概念。

    在 ZeRO-1 中，每个 rank 仅负责一部分 parameter 的 optimizer state（Adam 中的 m 和 v）。
    （经过 all-reduce 的）gradient 计算完成后，每个 rank：
    1. 使用其本地 optimizer state 更新其分区的 parameter。
    2. 通过 all-gather 将更新后的 parameter 同步到所有 rank。
    """
    print("=" * 60)
    print("ZeRO Stage 1: Optimizer State Partitioning")
    print("=" * 60)

    # 模拟 4 张 GPU
    world_size: int = 4
    total_params: int = 400
    params_per_rank: int = total_params // world_size

    # 完整的 optimizer state（Adam: m + v = 2 * params）
    full_optimizer_state_size: int = total_params * 2

    print(f"  Total parameters: {total_params}")
    print(f"  World size (GPUs): {world_size}")
    print(f"  Full optimizer state size: {full_optimizer_state_size} floats")
    print(f"  Per-rank optimizer state (DDP): {full_optimizer_state_size} floats")
    print(f"  Per-rank optimizer state (ZeRO-1): {params_per_rank * 2} floats")
    print(
        f"  Memory saved: {full_optimizer_state_size - params_per_rank * 2} floats per GPU"
    )
    print()

    # 演示 all-gather 模式
    # 每个 rank 持有一份 parameter 分区
    all_params: torch.Tensor = torch.arange(total_params, dtype=torch.float32)

    for rank in range(world_size):
        start: int = rank * params_per_rank
        end: int = start + params_per_rank
        partition: torch.Tensor = all_params[start:end]

        print(
            f"  Rank {rank} owns parameters [{start}:{end}] = {partition[:4].tolist()}..."
        )

        # 实际实现中，更新完该分区后，
        # 该 rank 会执行 all_gather 获取完整的更新后 parameter
        # dist.all_gather(all_params_list, partition)

    print()


# =============================================================================
# ZeRO Stage 2：Gradient 分区
# =============================================================================
#
# 除了 optimizer state 分区外，ZeRO-2 还对 gradient 进行分区。
# 不同于对所有 gradient 执行 all-reduce（需要存储完整 gradient），
# 每个 rank 只需存储其 parameter 分区对应的 gradient。
#
# Reduce-Scatter 操作取代了 All-Reduce：
#   - 每个 rank 对其分区的 gradient 执行 reduce（求和）
#   - 其他 rank 发送完即可释放各自的 gradient 分区
#
# 内存节省：Gradient + Optimizer state 减少为原来的 1/N


def zero_stage_2_demo() -> None:
    """
    演示 ZeRO-2 的 gradient + optimizer state 分区。

    核心要点：通过 gradient 分区，我们使用 reduce-scatter 替代 all-reduce。
    每个 rank 只需要其 parameter 分区对应的汇总 gradient。
    归约完成后，其他 rank 释放其 gradient 内存。
    """
    print("=" * 60)
    print("ZeRO Stage 2: Gradient + Optimizer State Partitioning")
    print("=" * 60)

    # 模拟 gradient 归约
    world_size: int = 4
    num_params: int = 16
    params_per_rank: int = num_params // world_size

    # 每个 rank 计算其本地 gradient
    local_grads: torch.Tensor = torch.randn(world_size, num_params)

    print("  Local gradients (each rank's computation):")
    for rank in range(world_size):
        print(f"    Rank {rank}: {local_grads[rank].round(decimals=2).tolist()}")

    # reduce-scatter：每个 rank 获取其分区的汇总值
    print()
    print("  After reduce-scatter (each rank gets summed grad for its partition):")
    global_sum: torch.Tensor = local_grads.sum(dim=0)
    for rank in range(world_size):
        start: int = rank * params_per_rank
        end: int = start + params_per_rank
        partition_sum: torch.Tensor = global_sum[start:end]
        print(
            f"    Rank {rank}: params[{start}:{end}] -> {partition_sum.round(decimals=2).tolist()}"
        )

    # 之后，每个 rank 释放其 gradient 内存（无需保留其他分区）
    print()
    print("  Memory usage comparison:")
    print(f"    DDP (all-reduce):     {num_params} floats per rank for gradients")
    print(
        f"    ZeRO-2 (reduce-scatter): {params_per_rank} floats per rank for gradients"
    )
    print(f"    Gradient memory saved: {num_params - params_per_rank} floats per rank")
    print()


# =============================================================================
# ZeRO Stage 3：Parameter 分区
# =============================================================================
#
# 最激进的形式：模型 parameter 本身也被分区。
# 没有单张 GPU 持有完整模型。在 forward/backward 过程中按需 all-gather
# parameter，用完即释放。
#
# Forward pass：
#   1. 对每一层：all-gather 所需 parameter
#   2. 计算该层输出
#   3. 释放 parameter（释放内存）
#
# Backward pass：
#   1. 再次 all-gather parameter
#   2. 计算 gradient
#   3. 通过 reduce-scatter 将 gradient 分发至其所属 rank
#   4. 释放 parameter
#
# 这使得训练超出单张 GPU 内存的模型成为可能。


def zero_stage_3_demo() -> None:
    """
    演示 ZeRO-3 的 parameter 分区和按需汇聚。

    核心思想：parameter 仅在计算需要时才从所属 rank 获取，然后释放。
    这与 FSDP 使用的原理相同。
    """
    print("=" * 60)
    print("ZeRO Stage 3: Parameter Partitioning")
    print("=" * 60)

    # 模拟一个具有 4 层的模型，每层有自己的 parameter
    num_layers: int = 4
    world_size: int = 4  # 每张 GPU 拥有 1 层
    params_per_layer: int = 256  # 每层的 parameter 数量

    print(f"  Model: {num_layers} layers, {params_per_layer} params/layer")
    print(f"  Total parameters: {num_layers * params_per_layer}")

    # 每个 rank 永久存储自己那层的 parameter
    print()
    print("  Parameter ownership:")
    for rank in range(world_size):
        print(
            f"    Rank {rank}: owns layer {rank} parameters ({params_per_layer} floats)"
        )

    print()
    print("  Forward pass flow:")
    for layer_idx in range(num_layers):
        owner_rank: int = layer_idx % world_size
        print(f"    Layer {layer_idx}:")
        print(
            f"      1. Rank {owner_rank} broadcasts layer {layer_idx} params to all ranks "
            f"({params_per_layer} floats transferred)"
        )
        print(f"      2. All ranks compute layer {layer_idx} output")
        print(
            f"      3. All ranks discard layer {layer_idx} params (free {params_per_layer} floats)"
        )

    print()
    print("  Memory comparison:")
    print(f"    DDP (full model):    {num_layers * params_per_layer} floats per rank")
    print(f"    ZeRO-3 (sharded):    {params_per_layer} floats per rank (params only)")
    print(
        f"    Peak memory (forward): {params_per_layer * 2} floats "
        f"(own + gathered layer)"
    )
    print(f"    Memory reduction: {num_layers}x")


# =============================================================================
# FSDP Wrapper 概念
# =============================================================================
#
# FSDP（完全分片数据并行）是 PyTorch 对 ZeRO-3 的实现。
# 它包裹每个子模块，管理 parameter sharding、汇聚和释放。
#
# FSDP 的关键组件：
# 1. Parameter sharding（FlatParameter）：展平并在各 rank 间分片。
# 2. Forward 时 all-gather：计算前汇聚 parameter。
# 3. Forward 后释放：释放汇聚的 parameter 以节省内存。
# 4. Backward 后重新分片：计算完 gradient 后重新分片 parameter。
# 5. Reduce-scatter gradient：每个 rank 仅获取其 shard 的 gradient。


class SimpleFSDPWrapper(nn.Module):
    """
    FSDP 包裹的简化演示。

    包裹一个模块以展示以下概念：
    - 初始化时 shard parameter
    - Forward 前汇聚
    - Backward 后释放

    注意：这是一个概念性实现，并非真正的 FSDP 实现。
    它不会实际在进程间通信。

    Args:
        module：要包裹的子模块。
        rank：当前进程的 rank。
        world_size：进程总数。
    """

    def __init__(self, module: nn.Module, rank: int, world_size: int) -> None:
        super().__init__()
        self.module: nn.Module = module
        self.rank: int = rank
        self.world_size: int = world_size

        # 真实 FSDP 中：展平并 shard parameter
        self._flat_param: Optional[torch.Tensor] = None
        self._sharded_params: Optional[torch.Tensor] = None
        self._gathered_params: Optional[torch.Tensor] = None

    def _flatten_and_shard(self) -> None:
        """
        将所有 parameter 展平为一维张量，仅保留属于本 rank 的 shard。
        """
        all_params: list[torch.Tensor] = []
        shapes: list[torch.Size] = []

        for param in self.module.parameters():
            shapes.append(param.shape)
            all_params.append(param.data.view(-1))

        self._flat_param = torch.cat(all_params)
        total_elements: int = self._flat_param.numel()
        shard_size: int = (total_elements + self.world_size - 1) // self.world_size

        start: int = self.rank * shard_size
        end: int = min(start + shard_size, total_elements)

        # 每个 rank 只保留其 shard
        # 真实 FSDP 中：这会释放 GPU 内存
        self._sharded_params = self._flat_param[start:end].clone()

        # 释放完整展平 param（概念上，内存被释放）
        self._flat_param = None

    def _gather_for_forward(self) -> None:
        """
        在 forward pass 前从所有 rank all-gather parameter。
        真实 FSDP 中：dist.all_gather 到一个预分配缓冲区。
        """
        # 真实 FSDP 中这会是这样：
        # dist.all_gather(all_shards, self._sharded_params)
        # 演示用：简单复制一份
        total_elements: int = sum(p.numel() for p in self.module.parameters())
        self._gathered_params = torch.zeros(total_elements)

    def _free_after_forward(self) -> None:
        """Forward pass 后释放汇聚的 parameter 以节省内存。"""
        # 真实 FSDP 中：delete 并 empty_cache
        # del self._gathered_params
        # torch.cuda.empty_cache()
        self._gathered_params = None

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass 配合汇聚/释放循环。"""
        # 1. All-gather parameter
        self._gather_for_forward()

        # 2. 运行模块
        output: Any = self.module(*args, **kwargs)

        # 3. 释放汇聚的 parameter
        # 真实 FSDP 中：注册 backward hook 在 backward 后重新分片
        self._free_after_forward()

        return output


def fsdp_demo() -> None:
    """演示 FSDP 包裹概念。"""
    print("=" * 60)
    print("FSDP (Fully Sharded Data Parallel) Concepts")
    print("=" * 60)

    world_size: int = 4
    rank: int = 2  # 模拟 rank 2

    # 创建一个简单模块
    linear: nn.Linear = nn.Linear(256, 128, bias=False)
    num_params: int = sum(p.numel() for p in linear.parameters())
    shard_size: int = (num_params + world_size - 1) // world_size

    print(f"  Module: Linear(256, 128)")
    print(f"  Total parameters: {num_params}")
    print(f"  World size: {world_size}")
    print(f"  Shard size per rank: {shard_size}")
    print()

    # 用 FSDP 概念包裹
    wrapper = SimpleFSDPWrapper(linear, rank, world_size)
    wrapper._flatten_and_shard()

    print(f"  Rank {rank} state:")
    print(f"    Owned (sharded) params: {wrapper._sharded_params.numel()} floats")
    print(f"    Total params: {num_params}")
    print(f"    Memory before forward: {wrapper._sharded_params.numel()} floats")
    print(
        f"    Memory during forward: {wrapper._sharded_params.numel() + num_params} floats "
        f"(shard + gathered)"
    )
    print(f"    Memory after forward: {wrapper._sharded_params.numel()} floats")
    print()

    print("  FSDP Key Operations:")
    print("    1. Flatten + Shard at init:     each rank keeps 1/N of params")
    print("    2. All-gather at forward:       gather full params (O(N) communication)")
    print("    3. Compute forward:             use gathered params")
    print("    4. Free after forward:          release gathered params")
    print("    5. All-gather at backward:      gather again for backward")
    print("    6. Compute backward:            compute gradients")
    print("    7. Reduce-scatter gradients:    each rank gets grad for its shard")
    print("    8. Update:                      apply optimizer to local shard")


# =============================================================================
# 内存对比总结
# =============================================================================


def memory_comparison() -> None:
    """输出 DDP 与各 ZeRO 阶段的内存对比。"""
    print("=" * 60)
    print("Memory Comparison: DDP vs ZeRO Stages")
    print("=" * 60)
    print()

    # 模型 param 数量（fp16 时每个 2 字节，fp32 时每个 4 字节）
    M: int = 10_000_000  # ~10M param
    N: int = 4  # GPU 数量
    K: int = 2  # optimizer state 数量（Adam 的 m 和 v）

    print(f"  Model size: {M:,} parameters")
    print(f"  GPUs: {N}")
    print(f"  Optimizer states: {K} (Adam moment estimates)")
    print()

    # DDP：每张 GPU 存储完整 param + 完整 gradient + 完整 optimizer state
    ddp_mem: int = M + M + K * M

    # ZeRO-1：完整 param + 完整 gradient + 分区 optimizer state
    z1_mem: int = M + M + (K * M) // N

    # ZeRO-2：完整 param + 分区 gradient + 分区 optimizer state
    z2_mem: int = M + M // N + (K * M) // N

    # ZeRO-3：分区 param + 分区 gradient + 分区 optimizer state
    z3_mem: int = M // N + M // N + (K * M) // N

    print(f"  | Strategy  | Params | Grads | OptState | Total (floats) | vs DDP |")
    print(f"  |-----------|--------|-------|----------|----------------|--------|")
    print(
        f"  | DDP       | {M:>6,} | {M:>5,} | {K * M:>8,} | {ddp_mem:>14,} |  1.00x |"
    )
    print(
        f"  | ZeRO-1    | {M:>6,} | {M:>5,} | {K * M // N:>8,} | {z1_mem:>14,} |  {ddp_mem / z1_mem:.2f}x |"
    )
    print(
        f"  | ZeRO-2    | {M:>6,} | {M // N:>5,} | {K * M // N:>8,} | {z2_mem:>14,} |  {ddp_mem / z2_mem:.2f}x |"
    )
    print(
        f"  | ZeRO-3    | {M // N:>6,} | {M // N:>5,} | {K * M // N:>8,} | {z3_mem:>14,} |  {ddp_mem / z3_mem:.2f}x |"
    )
    print()

    print("  核心洞察：ZeRO-3 可以训练比单张 GPU 能容纳大 N 倍的模型，")
    print("  前提是通信带宽足以支持 all-gather 操作。")


if __name__ == "__main__":
    zero_stage_1_demo()
    zero_stage_2_demo()
    zero_stage_3_demo()
    fsdp_demo()
    memory_comparison()
