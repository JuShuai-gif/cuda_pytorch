"""
分布式数据并行（Distributed Data Parallel，DDP）训练封装器。

使用 PyTorch 的 torch.distributed 基础原语实现 DDP 训练：
- 进程组初始化
- All-reduce 梯度同步
- 梯度累积支持
- Barrier 同步

本模块仅使用 torch.distributed 的基础 API（all_reduce、broadcast、barrier），
不依赖 torch.nn.parallel.DistributedDataParallel。
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

import torch
import torch.distributed as dist


def ddp_setup() -> tuple[int, int, int]:
    """
    初始化分布式环境。

    从环境变量中读取 RANK、WORLD_SIZE、LOCAL_RANK。
    初始化 NCCL 或 Gloo 进程组。

    Returns:
        (rank, world_size, local_rank) 三元组。
    """
    rank: int = int(os.environ.get("RANK", 0))
    world_size: int = int(os.environ.get("WORLD_SIZE", 1))
    local_rank: int = int(os.environ.get("LOCAL_RANK", 0))

    backend: str = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def ddp_cleanup() -> None:
    """清理分布式环境。"""
    dist.destroy_process_group()


class DDPTrainer:
    """
    使用原始 torch.distributed 操作的分布式数据并行训练器。

    本类不通过 DDP 包装模型，而是手动通过 all_reduce 来编排梯度同步。
    这展示了 DDP 内部工作原理的核心机制。

    梯度 all-reduce 流程：
    1. 每个进程在其本地数据分片上独立计算梯度。
    2. all_reduce 将所有进程的梯度求平均。
    3. 每个进程使用相同的平均梯度更新其本地模型副本。
    4. 经过 N 步后，所有模型副本完全一致。

    Args:
        model: 待训练的 PyTorch 模型。
        rank: 进程编号（0 = 主进程）。
        world_size: 进程总数。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        rank: int,
        world_size: int,
    ) -> None:
        self.model: torch.nn.Module = model
        self.rank: int = rank
        self.world_size: int = world_size

    def all_reduce_gradients(self) -> None:
        """
        使用 all_reduce 将所有进程的梯度求平均。

        这是 DDP 的核心：每个进程计算其本地梯度后，
        我们将所有进程的梯度求和并除以 world_size 得到平均值。
        这确保每个进程拥有相同的梯度。

        具体操作：
            对于每个参数 p：
                dist.all_reduce(p.grad, op=SUM)
                p.grad /= world_size
        """
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= self.world_size

    def broadcast_parameters(self, src: int = 0) -> None:
        """
        将模型参数从源进程广播到所有其他进程。

        通常在初始化时调用，以确保所有进程从相同的权重开始。

        Args:
            src: 源进程编号（默认值：0）。
        """
        for param in self.model.parameters():
            dist.broadcast(param.data, src=src)

    def barrier(self) -> None:
        """在屏障点同步所有进程。"""
        dist.barrier()

    def train_step(
        self,
        batch: dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        gradient_accumulation_steps: int = 1,
        step: int = 0,
    ) -> float:
        """
        执行一次带梯度同步的训练步骤。

        Args:
            batch: 包含 "input_ids" 和 "labels" 的字典。
            optimizer: 优化器实例。
            gradient_accumulation_steps: 要累积的微批次数量。
            step: 当前步数计数器。

        Returns:
            用于日志记录的损失值。
        """
        input_ids: torch.Tensor = batch["input_ids"].cuda()
        labels: torch.Tensor = batch["labels"].cuda()

        # 前向传播
        logits, _ = self.model(input_ids)
        loss: torch.Tensor = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )
        loss = loss / gradient_accumulation_steps
        loss.backward()

        # 梯度累积：仅在累积完成后进行同步和优化步骤
        if (step + 1) % gradient_accumulation_steps == 0:
            # 同步所有进程之间的梯度
            self.all_reduce_gradients()

            # 执行优化器步骤
            optimizer.step()
            optimizer.zero_grad()

        return loss.item() * gradient_accumulation_steps


def ddp_main(model: torch.nn.Module) -> None:
    """
    示例 DDP 训练入口点。

    本函数演示完整的 DDP 训练流程：
    1. 初始化进程组
    2. 广播初始参数
    3. 将数据拆分到各 rank
    4. 带梯度同步的训练
    5. Barrier 同步以实现干净退出

    Args:
        model: 以分布式方式训练的模型。
    """
    rank, world_size, local_rank = ddp_setup()

    # 将模型移至 GPU
    model = model.cuda(local_rank)

    # 将模型包装到 DDP 训练器中
    ddp_trainer = DDPTrainer(model, rank, world_size)

    # 广播初始参数，使所有进程从相同状态开始
    ddp_trainer.broadcast_parameters(src=0)

    # --- 示例：创建在各 rank 之间分布的数据 ---
    # 实际应用中，应使用 DistributedSampler 来划分数据。
    # 每个 rank 获取一个子集：rank i 处理样本 [i*N/world_size : (i+1)*N/world_size]

    total_samples: int = 1000
    samples_per_rank: int = total_samples // world_size
    start_idx: int = rank * samples_per_rank
    end_idx: int = start_idx + samples_per_rank

    print(f"[Rank {rank}] Processing samples {start_idx} to {end_idx}")
    # --- 数据划分示例结束 ---

    # barrier 示例：所有进程在此处等待，直到所有进程都到达此点
    ddp_trainer.barrier()
    if rank == 0:
        print("All processes reached the barrier - proceeding with training")

    # ... 训练循环写在这里 ...

    ddp_cleanup()
    if rank == 0:
        print("DDP training complete")


# 快速演示（单进程，无实际分布式设置）
if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from transformer.config import MiniLLMConfig
    from transformer.layers import MiniLLM

    print("DDP Trainer 模块加载成功。")
    print()
    print("核心概念：")
    print("  - all_reduce_gradients(): 将所有进程的梯度求平均")
    print("  - broadcast_parameters(): 从 rank 0 同步初始权重")
    print("  - barrier(): 所有进程的同步点")
    print()
    print("DDP 工作流程：")
    print("  1. 每个 rank 在其数据分片上计算本地梯度")
    print("  2. dist.all_reduce() 汇总所有 rank 的梯度")
    print("  3. 梯度除以 world_size 求平均")
    print("  4. 优化器在所有 rank 上使用相同的平均梯度进行更新")
    print("  5. 所有模型副本保持完全一致")

    # 构建一个小模型来演示参数计数
    config = MiniLLMConfig(
        vocab_size=1000,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=512,
    )
    model = MiniLLM(config)
    print(f"\nExample model: {model.get_num_params():,} parameters")
