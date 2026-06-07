"""
Demonstrate collective communication operations.
Uses PyTorch comments to simulate multi-process behavior without requiring real multi-GPU.
Each operation is illustrated with a concrete numerical example.
"""

from __future__ import annotations

import torch


def demo_broadcast() -> None:
    """Broadcast: one process sends data to all others. All receive the same copy."""
    print("=" * 60)
    print("BROADCAST: One-to-all communication")
    print("=" * 60)
    # Assume rank 0 has data [1, 2, 3, 4], others have zeros
    data = torch.tensor([1.0, 2.0, 3.0, 4.0])
    print(f"  Rank 0 sends:     {data}")
    print(f"  All ranks receive: {data}")
    print(f"  (In practice: torch.distributed.broadcast(tensor, src=0))")
    print()


def demo_scatter() -> None:
    """Scatter: one process distributes chunks of data to all processes."""
    print("=" * 60)
    print("SCATTER: One-to-all, each gets a different chunk")
    print("=" * 60)
    # Rank 0 has [0, 1, 2, 3, 4, 5, 6, 7] with 4 processes
    full_data = torch.arange(8, dtype=torch.float32)
    world_size = 4
    print(f"  Rank 0 input: {full_data}")
    chunk_size = len(full_data) // world_size
    for rank in range(world_size):
        chunk = full_data[rank * chunk_size : (rank + 1) * chunk_size]
        print(f"  Rank {rank} receives: {chunk}")
    print()


def demo_gather() -> None:
    """Gather: all processes send data to one process (reverse of scatter)."""
    print("=" * 60)
    print("GATHER: All-to-one, concatenate chunks")
    print("=" * 60)
    world_size = 4
    chunks = [torch.tensor([r, r + 1], dtype=torch.float32) for r in range(world_size)]
    for rank, chunk in enumerate(chunks):
        print(f"  Rank {rank} sends: {chunk}")
    gathered = torch.cat(chunks)
    print(f"  Rank 0 receives (gathered): {gathered}")
    print()


def demo_reduce() -> None:
    """Reduce: all processes contribute data, result aggregated at one process."""
    print("=" * 60)
    print("REDUCE: All-to-one with an operation (sum, min, max, etc.)")
    print("=" * 60)
    world_size = 4
    data = [torch.tensor([r * 2.0, r * 2.0 + 1.0]) for r in range(world_size)]
    op = "sum"
    print(f"  Operation: {op}")
    for rank, d in enumerate(data):
        print(f"  Rank {rank} contributes: {d}")
    result = data[0].clone()
    for d in data[1:]:
        result += d
    print(f"  Result at root: {result}")
    print()


def demo_all_gather() -> None:
    """All-gather: all processes gather data from all others. Everyone gets the full concatenated result."""
    print("=" * 60)
    print("ALL-GATHER: All processes get the full concatenated result")
    print("=" * 60)
    world_size = 4
    chunks = [torch.tensor([r * 3.0, r * 3.0 + 1.0, r * 3.0 + 2.0]) for r in range(world_size)]
    for rank, chunk in enumerate(chunks):
        print(f"  Rank {rank} sends: {chunk}")
    gathered = torch.cat(chunks)
    for rank in range(world_size):
        print(f"  Rank {rank} receives: {gathered}")
    print()


def demo_reduce_scatter() -> None:
    """Reduce-scatter: reduce then scatter. Each process gets a chunk of the reduced result."""
    print("=" * 60)
    print("REDUCE-SCATTER: Reduce + Scatter combined")
    print("=" * 60)
    world_size = 4
    # Each rank has a full-sized tensor
    data = [torch.tensor([r * 4 + i for i in range(8)], dtype=torch.float32) for r in range(world_size)]
    for rank, d in enumerate(data):
        print(f"  Rank {rank} input: {d}")
    # Sum all
    summed = data[0].clone()
    for d in data[1:]:
        summed += d
    print(f"  After reduce (sum): {summed}")
    # Scatter the result
    chunk_size = len(summed) // world_size
    for rank in range(world_size):
        chunk = summed[rank * chunk_size : (rank + 1) * chunk_size]
        print(f"  Rank {rank} receives: {chunk}")
    print()


def demo_all_reduce() -> None:
    """All-reduce: reduce + broadcast. All processes get the same reduced result."""
    print("=" * 60)
    print("ALL-REDUCE: Reduce + Broadcast, most common in DDP")
    print("=" * 60)
    world_size = 4
    grads = [torch.tensor([r * 0.5, r * 0.5 + 0.25]) for r in range(world_size)]
    print("  Operation: sum (typical for gradient sync in DDP)")
    for rank, g in enumerate(grads):
        print(f"  Rank {rank} gradient: {g}")
    # Sum and average (typical in DDP for gradient sync)
    summed = grads[0].clone()
    for g in grads[1:]:
        summed += g
    avg = summed / world_size
    for rank in range(world_size):
        print(f"  Rank {rank} receives (avg): {avg}")
    print()


def demo_all_to_all() -> None:
    """All-to-all: each process scatters data to all others (transpose operation)."""
    print("=" * 60)
    print("ALL-TO-ALL: Each process scatters to every other (transpose)")
    print("=" * 60)
    world_size = 3
    # Each rank has a matrix. All-to-all scatters columns to all ranks.
    data = [torch.tensor([[r * 10 + c for c in range(3)] for _ in range(2)], dtype=torch.float32) for r in range(world_size)]
    for rank, d in enumerate(data):
        print(f"  Rank {rank} input:\n{d}")
    # Each rank sends column j to rank j
    print("  After all-to-all:")
    for rank_out in range(world_size):
        result = torch.tensor([[r * 10 + rank_out for r in range(3)] for _ in range(2)], dtype=torch.float32)
        print(f"  Rank {rank_out} receives:\n{result}")
    print()


def main() -> None:
    demo_broadcast()
    demo_scatter()
    demo_gather()
    demo_reduce()
    demo_all_gather()
    demo_reduce_scatter()
    demo_all_reduce()
    demo_all_to_all()
    print("Collective operations demo complete.")


if __name__ == "__main__":
    main()
