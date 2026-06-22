"""
从零实现 DDP（Distributed Data Parallel）。
使用 torch.distributed 原语实现 all-reduce 梯度同步。
不使用 torch.nn.parallel.DistributedDataParallel。

用法：
    torchrun --nproc_per_node=2 ddp_train.py

如果只有 1 块 GPU 可用，脚本会检测到并在单进程模拟模式下运行，
该模式下手动拆分 batch 并同步梯度。
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# 简单模型
# ---------------------------------------------------------------------------


class SimpleMLP(nn.Module):
    """用于演示目的的简单 MLP。"""

    def __init__(
        self, input_dim: int = 128, hidden_dim: int = 256, num_classes: int = 10
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ---------------------------------------------------------------------------
# 手动 all-reduce 梯度同步
# ---------------------------------------------------------------------------


def _all_reduce_gradients(model: nn.Module, world_size: int) -> None:
    """
    使用 all-reduce 求和在所有进程间同步梯度，然后除以 world_size 求平均。
    这是 DDP 的核心：每个 rank 计算其本地梯度后，
    我们将它们全部求和再除以 world_size 得到全局梯度。
    """
    for param in model.parameters():
        if param.grad is not None:
            # All-reduce 将所有 rank 的梯度求和
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    param.grad, op=torch.distributed.ReduceOp.SUM
                )
            # 除以 world_size 求平均梯度
            param.grad /= world_size


def _all_reduce_gradients_simulated(
    model: nn.Module,
    all_models: list[nn.Module],
    world_size: int,
) -> None:
    """
    通过手动求和所有模型副本的梯度来模拟 all-reduce。
    在无法使用真实分布式环境（单 GPU）时使用。
    """
    for param_idx, param in enumerate(model.parameters()):
        if param.grad is not None:
            # 将所有副本的梯度求和
            summed_grad = param.grad.clone()
            for replica in all_models[1:]:
                replica_param = list(replica.parameters())[param_idx]
                if replica_param.grad is not None:
                    summed_grad += replica_param.grad
            # 存储平均后的梯度
            param.grad = summed_grad / world_size


# ---------------------------------------------------------------------------
# 主训练
# ---------------------------------------------------------------------------


def _create_dummy_data(
    num_samples: int = 1000,
    input_dim: int = 128,
    num_classes: int = 10,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """创建合成训练数据。"""
    x = torch.randn(num_samples, input_dim, device=device)
    y = torch.randint(0, num_classes, (num_samples,), device=device)
    return x, y


def train_single(
    model: nn.Module,
    dataloader: DataLoader,
    num_epochs: int,
    device: torch.device,
    world_size: int = 1,
    all_models: list[nn.Module] | None = None,
) -> list[float]:
    """
    带有手动梯度同步的训练循环。
    当 world_size == 1 时，这是普通的单 GPU 训练。
    当 world_size > 1 且没有 torch.distributed 时，使用模拟的 all-reduce。
    """
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()

            # 在所有 rank 之间同步梯度
            if torch.distributed.is_initialized():
                _all_reduce_gradients(model, world_size)
            elif all_models is not None:
                _all_reduce_gradients_simulated(model, all_models, world_size)

            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"  Epoch {epoch + 1}/{num_epochs}，Loss：{avg_loss:.4f}")

    return losses


# ---------------------------------------------------------------------------
# 入口点
# ---------------------------------------------------------------------------


def run_single_process() -> None:
    """在单进程模式下运行，使用模拟的 DDP。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")
    print("在单进程模式下运行，使用模拟的 DDP（world_size=2）")

    world_size = 2
    x, y = _create_dummy_data(num_samples=2000)
    dataset = TensorDataset(x, y)

    # 将数据拆分到各"rank"
    split_size = len(dataset) // world_size
    datasets = [
        TensorDataset(
            x[i * split_size : (i + 1) * split_size],
            y[i * split_size : (i + 1) * split_size],
        )
        for i in range(world_size)
    ]

    # 创建模型副本
    models = [SimpleMLP().to(device) for _ in range(world_size)]
    # 初始时保持参数同步
    for p_replica, p_main in zip(models[1].parameters(), models[0].parameters()):
        p_replica.data.copy_(p_main.data)

    dataloaders = [DataLoader(ds, batch_size=32, shuffle=True) for ds in datasets]

    # 使用模拟的梯度同步进行训练
    print("\n使用模拟的 DDP 梯度同步进行训练：")
    losses_main = []
    for epoch in range(3):
        epoch_loss = 0.0
        num_batches = 0
        for (bx1, by1), (bx2, by2) in zip(dataloaders[0], dataloaders[1]):
            bx1, by1 = bx1.to(device), by1.to(device)
            bx2, by2 = bx2.to(device), by2.to(device)

            # 副本 1 的本地反向传播
            optim1 = optim.SGD(models[0].parameters(), lr=0.01)
            optim2 = optim.SGD(models[1].parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()

            optim1.zero_grad()
            optim2.zero_grad()

            loss1 = criterion(models[0](bx1), by1)
            loss2 = criterion(models[1](bx2), by2)

            loss1.backward()
            loss2.backward()

            # 模拟 all-reduce：平均梯度
            _all_reduce_gradients_simulated(models[0], models, world_size)

            # 将同步后的梯度复制到副本 2
            for p1, p2 in zip(models[0].parameters(), models[1].parameters()):
                if p1.grad is not None and p2.grad is not None:
                    p2.grad.copy_(p1.grad)

            optim1.step()
            optim2.step()

            # 保持参数同步
            for p2, p1 in zip(models[1].parameters(), models[0].parameters()):
                p2.data.copy_(p1.data)

            epoch_loss += (loss1.item() + loss2.item()) / 2
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        losses_main.append(avg_loss)
        print(f"  Epoch {epoch + 1}/3，Loss：{avg_loss:.4f}")

    # 与单 GPU 基线对比
    print("\n基线（单 GPU，完整数据）：")
    model_baseline = SimpleMLP().to(device)
    full_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    losses_baseline = train_single(model_baseline, full_loader, 3, device)

    print(f"\n最终 loss - DDP 模拟：{losses_main[-1]:.4f}")
    print(f"最终 loss - 单 GPU：    {losses_baseline[-1]:.4f}")
    print("（由于 batch 排序不同，loss 可能有所差异）")


def run_distributed(rank: int, world_size: int) -> None:
    """使用真实的 torch.distributed 运行。"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    x, y = _create_dummy_data(num_samples=2000)
    dataset = TensorDataset(x, y)

    # 每个 rank 获得不同的数据分片
    split_size = len(dataset) // world_size
    start = rank * split_size
    end = (rank + 1) * split_size
    local_dataset = TensorDataset(x[start:end], y[start:end])
    dataloader = DataLoader(local_dataset, batch_size=32, shuffle=True)

    model = SimpleMLP().to(device)

    # 从 rank 0 广播初始参数
    for param in model.parameters():
        torch.distributed.broadcast(param.data, src=0)

    print(f"[Rank {rank}] 开始训练……")
    train_single(model, dataloader, 3, device, world_size)

    torch.distributed.destroy_process_group()


def main() -> None:
    print("=" * 60)
    print("从零实现 DDP 训练")
    print("=" * 60)

    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    rank_env = int(os.environ.get("RANK", "0"))
    local_rank_env = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size_env > 1:
        # 在 torchrun 下运行
        print(f"在 torchrun 下运行：rank={rank_env}，world_size={world_size_env}")
        run_distributed(rank_env, world_size_env)
    else:
        # 单进程模式
        print("未在 torchrun 下运行。使用单进程模拟 DDP。")
        run_single_process()


if __name__ == "__main__":
    main()
