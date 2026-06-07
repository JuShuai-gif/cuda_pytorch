"""
DDP (Distributed Data Parallel) from scratch.
Implements all-reduce gradient synchronization using torch.distributed primitives.
Does NOT use torch.nn.parallel.DistributedDataParallel.

Usage:
    torchrun --nproc_per_node=2 ddp_train.py

If only 1 GPU is available, the script detects this and runs in single-process
simulation mode where it manually splits batches and syncs gradients.
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
# Simple model
# ---------------------------------------------------------------------------


class SimpleMLP(nn.Module):
    """A simple MLP for demonstration purposes."""

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
# Manual all-reduce gradient sync
# ---------------------------------------------------------------------------


def _all_reduce_gradients(model: nn.Module, world_size: int) -> None:
    """
    Synchronize gradients across all processes using all-reduce sum,
    then average by world_size.
    This is the core of DDP: after each rank computes its local gradient,
    we sum them all up and divide to get the global gradient.
    """
    for param in model.parameters():
        if param.grad is not None:
            # All-reduce sums gradients from all ranks
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    param.grad, op=torch.distributed.ReduceOp.SUM
                )
            # Average by world size to get the mean gradient
            param.grad /= world_size


def _all_reduce_gradients_simulated(
    model: nn.Module,
    all_models: list[nn.Module],
    world_size: int,
) -> None:
    """
    Simulate all-reduce by manually summing gradients from all model replicas.
    Used when real distributed is not available (single GPU).
    """
    for param_idx, param in enumerate(model.parameters()):
        if param.grad is not None:
            # Sum gradients from all replicas
            summed_grad = param.grad.clone()
            for replica in all_models[1:]:
                replica_param = list(replica.parameters())[param_idx]
                if replica_param.grad is not None:
                    summed_grad += replica_param.grad
            # Store averaged gradient
            param.grad = summed_grad / world_size


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------


def _create_dummy_data(
    num_samples: int = 1000,
    input_dim: int = 128,
    num_classes: int = 10,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic training data."""
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
    Training loop with manual gradient synchronization.
    When world_size == 1, this is plain single-GPU training.
    When world_size > 1 without torch.distributed, uses simulated all-reduce.
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

            # Synchronize gradients across all ranks
            if torch.distributed.is_initialized():
                _all_reduce_gradients(model, world_size)
            elif all_models is not None:
                _all_reduce_gradients_simulated(model, all_models, world_size)

            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"  Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return losses


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def run_single_process() -> None:
    """Run in single-process mode with simulated DDP."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Running single-process mode with simulated DDP (world_size=2)")

    world_size = 2
    x, y = _create_dummy_data(num_samples=2000)
    dataset = TensorDataset(x, y)

    # Split data across "ranks"
    split_size = len(dataset) // world_size
    datasets = [
        TensorDataset(
            x[i * split_size : (i + 1) * split_size],
            y[i * split_size : (i + 1) * split_size],
        )
        for i in range(world_size)
    ]

    # Create model replicas
    models = [SimpleMLP().to(device) for _ in range(world_size)]
    # Keep parameters synchronized initially
    for p_replica, p_main in zip(models[1].parameters(), models[0].parameters()):
        p_replica.data.copy_(p_main.data)

    dataloaders = [DataLoader(ds, batch_size=32, shuffle=True) for ds in datasets]

    # Train with simulated gradient sync
    print("\nTraining with simulated DDP gradient sync:")
    losses_main = []
    for epoch in range(3):
        epoch_loss = 0.0
        num_batches = 0
        for (bx1, by1), (bx2, by2) in zip(dataloaders[0], dataloaders[1]):
            bx1, by1 = bx1.to(device), by1.to(device)
            bx2, by2 = bx2.to(device), by2.to(device)

            # Local backward on replica 1
            optim1 = optim.SGD(models[0].parameters(), lr=0.01)
            optim2 = optim.SGD(models[1].parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()

            optim1.zero_grad()
            optim2.zero_grad()

            loss1 = criterion(models[0](bx1), by1)
            loss2 = criterion(models[1](bx2), by2)

            loss1.backward()
            loss2.backward()

            # Simulate all-reduce: average gradients
            _all_reduce_gradients_simulated(models[0], models, world_size)

            # Copy synced gradients to replica 2
            for p1, p2 in zip(models[0].parameters(), models[1].parameters()):
                if p1.grad is not None and p2.grad is not None:
                    p2.grad.copy_(p1.grad)

            optim1.step()
            optim2.step()

            # Keep parameters synced
            for p2, p1 in zip(models[1].parameters(), models[0].parameters()):
                p2.data.copy_(p1.data)

            epoch_loss += (loss1.item() + loss2.item()) / 2
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        losses_main.append(avg_loss)
        print(f"  Epoch {epoch + 1}/3, Loss: {avg_loss:.4f}")

    # Compare with single-GPU baseline
    print("\nBaseline (single GPU, full data):")
    model_baseline = SimpleMLP().to(device)
    full_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    losses_baseline = train_single(model_baseline, full_loader, 3, device)

    print(f"\nFinal loss - DDP simulated: {losses_main[-1]:.4f}")
    print(f"Final loss - Single GPU:     {losses_baseline[-1]:.4f}")
    print("(Losses may differ due to different batch ordering)")


def run_distributed(rank: int, world_size: int) -> None:
    """Run using actual torch.distributed."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    x, y = _create_dummy_data(num_samples=2000)
    dataset = TensorDataset(x, y)

    # Each rank gets a different split
    split_size = len(dataset) // world_size
    start = rank * split_size
    end = (rank + 1) * split_size
    local_dataset = TensorDataset(x[start:end], y[start:end])
    dataloader = DataLoader(local_dataset, batch_size=32, shuffle=True)

    model = SimpleMLP().to(device)

    # Broadcast initial parameters from rank 0
    for param in model.parameters():
        torch.distributed.broadcast(param.data, src=0)

    print(f"[Rank {rank}] Starting training...")
    train_single(model, dataloader, 3, device, world_size)

    torch.distributed.destroy_process_group()


def main() -> None:
    print("=" * 60)
    print("DDP Training From Scratch")
    print("=" * 60)

    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    rank_env = int(os.environ.get("RANK", "0"))
    local_rank_env = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size_env > 1:
        # Running under torchrun
        print(f"Running under torchrun: rank={rank_env}, world_size={world_size_env}")
        run_distributed(rank_env, world_size_env)
    else:
        # Single process mode
        print("Not running under torchrun. Using single-process simulated DDP.")
        run_single_process()


if __name__ == "__main__":
    main()
