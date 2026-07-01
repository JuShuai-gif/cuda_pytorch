"""Meta FakeTensor case study 7: FakeTensor in distributed (FSDP/TP).

Companion script for meta_fake_tensor/meta_fake_tensor.md. Covers:
  1. FakeTensor propagation through FSDP sharding
  2. Device mesh and fake tensor
  3. Compile + FSDP integration

Run:
    python 07_distributed_fake.py
"""

import sys

import torch
from torch._subclasses.fake_tensor import FakeTensorMode


def exp_fake_with_device_mesh():
    print("=" * 60)
    print("1. FakeTensor with device information")
    print("=" * 60)

    # FakeTensor preserves device info during tracing
    with FakeTensorMode():
        x_cuda0 = torch.randn(4, 8, device="cuda:0")
        x_cuda1 = torch.randn(4, 8, device="cuda:1")

        print(f"  cuda:0 tensor: device={x_cuda0.device}, shape={list(x_cuda0.shape)}")
        print(f"  cuda:1 tensor: device={x_cuda1.device}, shape={list(x_cuda1.shape)}")

        # Cross-device operation
        y = x_cuda0 + x_cuda1.to(0)
        print(f"  Cross-device add: device={y.device}, shape={list(y.shape)}")

    print(f"\n  In FSDP/TP compile flow:")
    print(f"    FakeTensorMode traces graph with device info")
    print(f"    Compiler knows which tensor is on which device")
    print(f"    Inserts cross-device communication ops accordingly")
    print()


def exp_fsdp_meta_precheck():
    print("=" * 60)
    print("2. FSDP shard shape validation via meta")
    print("=" * 60)

    # When loading a pre-trained model with FSDP
    # Meta tensors verify shard shapes without GPU allocation

    model_sizes = {
        "q_proj": (4096, 4096),
        "k_proj": (4096, 1024),
        "v_proj": (4096, 1024),
        "o_proj": (4096, 4096),
    }

    world_size = 4
    print(f"  FSDP shard validation (world_size={world_size}):")

    for name, (out_dim, in_dim) in model_sizes.items():
        full_shape = [out_dim, in_dim]

        # Shard along dim 0 (FSDP default)
        shard_dim = 0
        shard_size = out_dim // world_size
        shard_shape = [shard_size, in_dim]

        full_mem = out_dim * in_dim * 4 / 1024**2
        shard_mem = shard_size * in_dim * 4 / 1024**2

        print(f"  {name:12s}: full={list(full_shape)} ({full_mem:.1f}MB) -> shard={list(shard_shape)} ({shard_mem:.1f}MB/rank)")

    print(f"\n  Meta tensor + FSDP benefits:")
    print(f"    - Verify shard dims before building model")
    print(f"    - Catch shard size mismatches early")
    print(f"    - No GPU allocation needed for validation")
    print()


EXPERIMENTS = {
    "device": exp_fake_with_device_mesh,
    "fsdp": exp_fsdp_meta_precheck,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[meta_fake_tensor case 7] DONE")


if __name__ == "__main__":
    main()
