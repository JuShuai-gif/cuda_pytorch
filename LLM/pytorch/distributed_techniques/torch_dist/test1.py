"""torch.distributed fundamentals demo.

Covers concepts from distributed_techniques/torch_dist/README.md:
  1. meta device:          abstract device for shape analysis (no memory allocation)
  2. process groups:       grouping GPUs for collective communication
  3. device mesh:          structured multi-dimensional process groups
  4. DTensor:              sharded/replicated/partial tensors

This script is designed to run on a single GPU (or CPU) for learning.
For actual multi-GPU runs, use torchrun:

    torchrun --nproc_per_node=2 test1.py multigpu

The single-process demos work without distributed setup.

Run:
    python test1.py                  # full demo (single-process)
    python test1.py meta             # meta device
    python test1.py pg               # process groups (needs torchrun)
    python test1.py mesh             # device mesh (needs torchrun)
"""

import os
import sys

import torch
import torch.nn as nn


# ============ 1. meta device: abstract device ============
def exp_meta():
    print("=" * 60)
    print("1. meta device: shape analysis without memory allocation")
    print("=" * 60)

    # Create model on meta device - no actual parameters allocated
    model = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    ).to("meta")

    print(f"  Model defined on meta device:")
    print(f"    {model}")

    # Check parameters: they exist but have no data
    for name, p in model.named_parameters():
        print(
            f"    {name}: shape={list(p.shape)}, device={p.device}, "
            f"has_data={p._typed_storage().size() > 0 if p.device.type == 'meta' else 'N/A'}"
        )

    # Forward pass: no computation, just shape propagation
    x = torch.randn(4, 1024).to("meta")
    with torch.no_grad():
        y = model(x)
    print(f"\n  Forward pass on meta device:")
    print(f"    input:  {list(x.shape)}")
    print(f"    output: {list(y.shape)}")

    # Count parameters (useful for model sizing)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Estimated FP16 memory: {total_params * 2 / 1e9:.2f} GB")
    print(f"  Estimated FP32 memory: {total_params * 4 / 1e9:.2f} GB")

    print("\n  -> meta device enables model architecture analysis")
    print("     without any GPU/CPU memory allocation")
    print("     (used for large model initialization pipelines)")
    print()


# ============ 2. Process groups ============
def exp_process_group():
    """Requires torchrun to actually run. Shows the API pattern."""
    print("=" * 60)
    print("2. Process groups: grouping GPUs for communication")
    print("=" * 60)

    # Check if running in distributed mode
    if "RANK" not in os.environ:
        print("  [SKIP] Not running in distributed mode.")
        print("  Run with: torchrun --nproc_per_node=2 test1.py pg")
        print()
        print("  Key concepts (shown in code):")
        print("    dist.init_process_group(backend='nccl')  -- initialize")
        print("    dist.new_group([0, 1])                   -- subgroup")
        print("    dist.new_group([2, 3])                   -- another subgroup")
        print(
            "    dist.all_reduce(tensor, group=g)         -- communicate within group"
        )
        return

    import torch.distributed as dist

    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"  Rank {rank}/{world_size}")

    if world_size >= 2:
        # Create subgroups
        group_01 = dist.new_group([0, 1])
        if rank in (0, 1):
            tensor = torch.tensor([rank + 1.0], device=f"cuda:{rank}")
            dist.all_reduce(tensor, group=group_01)
            print(
                f"    rank={rank} in group [0,1]: all_reduce result = {tensor.item()}"
            )

        if world_size >= 4:
            group_23 = dist.new_group([2, 3])
            if rank in (2, 3):
                tensor = torch.tensor([rank + 1.0], device=f"cuda:{rank}")
                dist.all_reduce(tensor, group=group_23)
                print(
                    f"    rank={rank} in group [2,3]: all_reduce result = {tensor.item()}"
                )

    # Global all_reduce
    tensor = torch.tensor(
        [rank + 1.0], device=f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    )
    dist.all_reduce(tensor)
    print(f"    rank={rank} global all_reduce: {tensor.item()}")

    dist.destroy_process_group()
    print()


# ============ 3. Device mesh ============
def exp_device_mesh():
    """Demonstrates device mesh API for structured parallelism."""
    print("=" * 60)
    print("3. Device mesh: structured multi-dimensional process groups")
    print("=" * 60)

    if "RANK" not in os.environ:
        print("  [SKIP] Not running in distributed mode.")
        print("  Run with: torchrun --nproc_per_node=4 test1.py mesh")
        print()
        print("  Key concepts (shown in code):")
        print(
            "    mesh = init_device_mesh('cuda', (2, 2), mesh_dim_names=('dp', 'tp'))"
        )
        print("    dp_group = mesh['dp']   # data-parallel group")
        print("    tp_group = mesh['tp']   # tensor-parallel group")
        print("    mesh['dp', 'tp']       # full 2D mesh")
        print()
        print("  Common configurations:")
        print("    1D: (4,)        -> data parallel only")
        print("    2D: (2, 2)      -> DP x TP")
        print("    3D: (2, 2, 2)   -> DP x TP x PP (pipeline)")
        return

    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank()

    mesh = init_device_mesh(
        "cuda" if torch.cuda.is_available() else "cpu",
        (2, 2),
        mesh_dim_names=("dp", "tp"),
    )

    print(f"  Rank {rank}:")
    print(f"    mesh shape: {mesh.shape}")
    print(f"    mesh dim names: {mesh.mesh_dim_names}")
    print(f"    DP group ranks: {mesh['dp'].get_group_rank()}")
    print(f"    TP group ranks: {mesh['tp'].get_group_rank()}")

    dist.destroy_process_group()
    print()


# ============ 4. DTensor concepts (single-process illustration) ============
def exp_dtensor():
    """Illustrate DTensor concepts without actual distributed run."""
    print("=" * 60)
    print("4. DTensor: sharded/replicated/partial tensors")
    print("=" * 60)

    # Create a tensor to simulate distributed storage
    full_tensor = torch.arange(16, dtype=torch.float32).view(4, 4)
    print(f"  Full tensor [4,4]:\n{full_tensor}")
    print()

    # Simulate Shard(0): split along row dimension
    print("  Shard(0) - split rows across 2 devices:")
    print(f"    device 0:\n{full_tensor[:2]}")
    print(f"    device 1:\n{full_tensor[2:]}")
    print(f"    each device only stores 2x4, not 4x4")

    # Simulate Shard(1): split along column dimension
    print(f"\n  Shard(1) - split columns across 2 devices:")
    print(f"    device 0:\n{full_tensor[:, :2]}")
    print(f"    device 1:\n{full_tensor[:, 2:]}")
    print(f"    each device only stores 4x2")

    # Simulate Replicate: every device has full copy
    print(f"\n  Replicate - every device has full copy:")
    print(f"    device 0: same 4x4 tensor")
    print(f"    device 1: same 4x4 tensor")
    print(f"    memory = N * tensor_size")

    # Simulate Partial: sum of tensors across devices
    dev0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    dev1 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    partial_result = dev0 + dev1
    print(f"\n  Partial (sum across devices):")
    print(f"    device 0 tensor:\n{dev0}")
    print(f"    device 1 tensor:\n{dev1}")
    print(f"    after all_reduce sum:\n{partial_result}")

    print("\n  Usage with actual API (requires distributed):")
    print("    from torch.distributed.tensor import DTensor, Shard, Replicate, Partial")
    print("    dt = DTensor.from_local(local_tensor, mesh, [Shard(0)])")
    print("    dt = DTensor.from_local(local_tensor, mesh, [Replicate()])")
    print("    dt.redistribute(placements=[Shard(1)])  # change sharding layout")
    print()


EXPERIMENTS = {
    "meta": exp_meta,
    "pg": exp_process_group,
    "mesh": exp_device_mesh,
    "dtensor": exp_dtensor,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for exp in exps:
        if exp not in EXPERIMENTS:
            print(f"unknown exp '{exp}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[exp]()

    print("[torch.distributed demo] DONE")


if __name__ == "__main__":
    main()
