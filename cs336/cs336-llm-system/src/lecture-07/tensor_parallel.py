"""
Megatron-style tensor parallelism for MLP layers.
Implements column-parallel and row-parallel linear layers by manually
splitting weight matrices. No distributed communication required to
understand the concept.

Reference: Megatron-LM: Training Multi-Billion Parameter Language Models
           Using Model Parallelism (Shoeybi et al., 2019)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ColumnParallelLinear(nn.Module):
    """
    Column-parallel linear layer.

    The weight matrix W is split along its column dimension across devices.
    Input is replicated (same on all devices).
    Output is partitioned along the last dimension.

    Forward:
        y_i = x @ W_i  (no communication needed)

    In a transformer, this is typically used for the first linear in the FFN,
    or for QKV projections in attention. The output needs an all-reduce for
    activations if followed by a non-column-parallel layer (like GeLU).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_partitions: int = 2,
        partition_idx: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_partitions = num_partitions
        self.partition_idx = partition_idx

        # Each partition has out_features // num_partitions columns
        assert out_features % num_partitions == 0, (
            f"out_features ({out_features}) must be divisible by num_partitions ({num_partitions})"
        )
        self.partition_out_features = out_features // num_partitions

        # Local weight: (in_features, out_features // num_partitions)
        self.weight = nn.Parameter(
            torch.randn(in_features, self.partition_out_features) * 0.02
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.partition_out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, in_features) - replicated input
        Returns: (batch, seq_len, partition_out_features) - partitioned output
        """
        y = x @ self.weight
        if self.bias is not None:
            y = y + self.bias
        return y

    def gather_output(self, local_output: torch.Tensor) -> torch.Tensor:
        """
        Simulate gathering partitioned outputs from all devices.
        In real distributed, this would be an all-gather.
        """
        # In a real implementation, each device performs all-gather here.
        # Since we are simulating, we return what would be the full output.
        full_weight = self.get_full_weight()
        full_bias = self.get_full_bias()
        return local_output @ torch.eye(self.partition_out_features)  # placeholder

    def get_full_weight(self) -> torch.Tensor:
        """Return the conceptual full weight matrix (for explanation only)."""
        # In practice, this never exists on a single GPU
        return torch.randn(self.in_features, self.out_features)

    def get_full_bias(self) -> torch.Tensor | None:
        """Return the conceptual full bias vector."""
        if self.bias is not None:
            return torch.randn(self.out_features)
        return None


class RowParallelLinear(nn.Module):
    """
    Row-parallel linear layer.

    The weight matrix W is split along its row dimension across devices.
    Input is partitioned along the last dimension.
    Output is replicated (same on all devices) after all-reduce.

    Forward:
        y_i = x_i @ W_i (partial sum)
        y = all-reduce(y_i) (sum across partitions)

    In a transformer FFN, this is typically used after the activation:
        [ColumnParallel FC] → GeLU → [RowParallel FC]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_partitions: int = 2,
        partition_idx: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_partitions = num_partitions
        self.partition_idx = partition_idx

        assert in_features % num_partitions == 0, (
            f"in_features ({in_features}) must be divisible by num_partitions ({num_partitions})"
        )
        self.partition_in_features = in_features // num_partitions

        # Local weight: (in_features // num_partitions, out_features)
        self.weight = nn.Parameter(
            torch.randn(self.partition_in_features, out_features) * 0.02
        )
        if bias:
            # Bias is duplicated (each partition has the full bias). Can also be split.
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, partition_in_features) - partitioned input
        Returns: (batch, seq_len, out_features) - partial sum (needs all-reduce)
        """
        y = x @ self.weight
        if self.bias is not None:
            y = y + self.bias
        return y

    def simulate_all_reduce(self, partial_outputs: list[torch.Tensor]) -> torch.Tensor:
        """Simulate all-reduce across partitions."""
        return torch.stack(partial_outputs).sum(dim=0)


class TensorParallelMLP(nn.Module):
    """
    Complete tensor-parallel MLP using Megatron-style column + row parallelism.

    Architecture:
        Input (replicated) → ColumnParallelLinear → ReLU → RowParallelLinear → Output (replicated)

    Communication analysis:
        - ColumnParallel: f (no comm in forward, no comm in backward)
        - ReLU: f (element-wise, no comm)
        - RowParallel: f (all-reduce in forward, identity in backward)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_partitions: int = 2,
        partition_idx: int = 0,
    ):
        super().__init__()
        # Column-parallel: in_features→intermediate, split output columns
        self.fc1 = ColumnParallelLinear(
            in_features=hidden_size,
            out_features=intermediate_size,
            num_partitions=num_partitions,
            partition_idx=partition_idx,
        )
        # Row-parallel: intermediate→hidden, split input rows
        self.fc2 = RowParallelLinear(
            in_features=intermediate_size,
            out_features=hidden_size,
            num_partitions=num_partitions,
            partition_idx=partition_idx,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: replicated input (same on all partitions)
        Returns partial output that needs all-reduce to be replicated
        """
        # Column-parallel: no communication needed
        h = F.relu(self.fc1(x))
        # Row-parallel: each partition computes partial sum
        # In real distributed, all-reduce is needed here to sum partial outputs
        y = self.fc2(h)
        return y  # Needs all-reduce from all partitions


def demo_tensor_parallel() -> None:
    """Demonstrate tensor parallelism by simulating multiple devices."""
    print("=" * 60)
    print("Tensor Parallelism Demo (Megatron-style)")
    print("=" * 60)

    hidden_size = 64
    intermediate_size = 128
    num_partitions = 4
    batch_size = 2
    seq_len = 8

    # Create tensor-parallel MLP partitions
    print("\nCreating tensor-parallel MLP with 4 partitions...")
    partitions = [
        TensorParallelMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_partitions=num_partitions,
            partition_idx=i,
        )
        for i in range(num_partitions)
    ]

    # Simulate forward pass
    x = torch.randn(batch_size, seq_len, hidden_size)

    print(f"\nInput shape: {x.shape} (replicated on all devices)")
    print(f"\nColumn-parallel FC1: {hidden_size} → {intermediate_size}")
    print(
        f"  Weight per partition: ({hidden_size}, {intermediate_size // num_partitions})"
    )
    print(
        f"  Output per partition: ({batch_size}, {seq_len}, {intermediate_size // num_partitions})"
    )
    print(f"  Communication: None (input already replicated)")

    # Forward through column-parallel
    h_partitions = [p.fc1(x) for p in partitions]
    for i, h in enumerate(h_partitions):
        print(f"  Partition {i} output shape: {h.shape}")

    print(f"\nReLU activation: element-wise, no communication")

    h_activated = [F.relu(h) for h in h_partitions]

    print(f"\nRow-parallel FC2: {intermediate_size} → {hidden_size}")
    print(
        f"  Weight per partition: ({intermediate_size // num_partitions}, {hidden_size})"
    )
    print(
        f"  Output per partition: ({batch_size}, {seq_len}, {hidden_size}) [partial sum]"
    )
    print(f"  Communication: all-reduce needed to sum partial outputs")

    y_partitions = [p.fc2(h_act) for p, h_act in zip(partitions, h_activated)]
    y_reduced = torch.stack(y_partitions).sum(dim=0)

    print(f"\nAfter all-reduce (sum): {y_reduced.shape}")
    print(f"Output shape: {y_reduced.shape} (replicated on all devices)")

    # Communication analysis
    print(f"\nCommunication Analysis:")
    print(f"  FC1 (col-parallel): f = 0 bytes (no communication)")
    print(f"  ReLU:               f = 0 bytes")
    print(f"  FC2 (row-parallel): f = batch*seq*hidden*bytes_per_element (all-reduce)")
    print(f"  Total forward comm: 1 all-reduce per transformer block")
    print(f"  Total backward comm: 1 all-reduce (for col-parallel grad)")

    # Memory analysis per device
    print(f"\nMemory per Device:")
    fc1_params = hidden_size * (intermediate_size // num_partitions)
    fc2_params = (intermediate_size // num_partitions) * hidden_size
    total_params = fc1_params + fc2_params
    print(f"  FC1 parameters: {fc1_params:,}")
    print(f"  FC2 parameters: {fc2_params:,}")
    print(
        f"  Total per device: {total_params:,} (vs {hidden_size * intermediate_size * 2:,} without TP)"
    )
    print(
        f"  Memory reduction: {1 - total_params / (hidden_size * intermediate_size * 2):.0%}"
    )


def main() -> None:
    demo_tensor_parallel()


if __name__ == "__main__":
    main()
