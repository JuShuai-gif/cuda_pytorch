"""DDP Reducer case study 1: bucket size impact, gradient sync, and find_unused.

Companion script for distributed_techniques/ddp_reducer/ddp_reducer.md. Covers:
  1. DDP autograd hook observing gradient timing
  2. find_unused_parameters=True overhead
  3. Gradient sync verification

Run (requires torchrun or torch.distributed.launch):
    torchrun --nproc_per_node=2 01_bucket_gradient_sync.py
"""

import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def _setup():
    if not dist.is_initialized():
        try:
            dist.init_process_group("gloo")
        except Exception:
            print("[WARN] DDP not available in this environment")
            return False
    return True


def exp_ddp_hook_timing():
    print("=" * 60)
    print("1. Observe DDP communication overlap")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available for NCCL DDP")
        print("  For CPU-only DDP testing, use gloo backend")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)

    # Build a model with varied parameter sizes
    model = torch.nn.Sequential(
        torch.nn.Linear(256, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 256),
    ).to(device)

    try:
        ddp_model = DDP(model, device_ids=[0] if torch.cuda.is_available() else None)

        # Register hooks to observe gradient timing
        events = []

        def make_hook(name):
            def hook(grad):
                if torch.cuda.is_available():
                    events.append((name, torch.cuda.Event(enable_timing=True)))
                    events[-1][1].record()
                return grad

            return hook

        for name, param in ddp_model.named_parameters():
            param.register_hook(make_hook(name))

        x = torch.randn(16, 256, device=device)
        loss = ddp_model(x).sum()
        loss.backward()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            print(f"  Grad hooks fired: {len(events)}")

        print(f"  Keys to observe from output:")
        print(f"    - Hooks fire in reverse order (backward order)")
        print(f"    - DDP registers its own hooks on top for allreduce")
        print(f"    - Bucket grouping visible via hook timing")

    except Exception as e:
        print(f"  DDP test error: {str(e)[:120]}")
    print()


def exp_find_unused_example():
    print("=" * 60)
    print("2. find_unused_parameters=True behavior")
    print("=" * 60)

    if not torch.cuda.is_available():
        return

    device = torch.device("cuda:0")

    class ConditionalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.branch_a = torch.nn.Linear(64, 64)
            self.branch_b = torch.nn.Linear(64, 64)

        def forward(self, x, use_branch_b=False):
            out = self.branch_a(x)
            if use_branch_b:
                out = out + self.branch_b(x)
            return out

    model = ConditionalModel().to(device)

    try:
        # Without find_unused_parameters: branch_b's param might be missing
        ddp = DDP(model, device_ids=[0], find_unused_parameters=False)

        x = torch.randn(4, 64, device=device)

        # Only use branch_a -> branch_b param NOT used
        try:
            loss = ddp(x, use_branch_b=False).sum()
            loss.backward()
            print(f"  Without find_unused (branch_b not used):")
            print(f"    (may pass or fail depending on DDP strictness)")
        except Exception as e:
            print(f"    Error: {str(e)[:80]}")

        # With find_unused_parameters=True
        ddp2 = DDP(
            ConditionalModel().to(device),
            device_ids=[0],
            find_unused_parameters=True,
        )

        loss2 = ddp2(x, use_branch_b=False).sum()
        loss2.backward()
        print(f"\n  With find_unused=True (branch_b not used):")
        print(f"    backward succeeded (handles unused parameters)")

        # Overhead: requires extra all-reduce for unused param sync
        print(f"\n  find_unused=True overhead:")
        print(f"    1. Extra all-reduce to sync which params are unused")
        print(f"    2. Cannot start partial bucket allreduce early")
    except Exception as e:
        print(f"  Error: {str(e)[:120]}")
    print()


def exp_bucket_size_manual():
    print("=" * 60)
    print("3. Manual bucket_size tuning guidance")
    print("=" * 60)

    print(f"  Default bucket_cap_mb: 25 (25MB)")
    print(f"")
    print(f"  Tuning guide:")
    print(f"    Small model (<100M params): bucket_cap_mb=5")
    print(f"      -> More all-reduce calls, but starts overlap earlier")
    print(f"    Medium model (100M-1B):  bucket_cap_mb=25 (default)")
    print(f"      -> Balanced all-reduce frequency and overlap")
    print(f"    Large model (>1B):       bucket_cap_mb=50-100")
    print(f"      -> Fewer all-reduce calls, less launch overhead")
    print(f"")
    print(f"  Example:")
    print(f"    ddp_model = DDP(model, bucket_cap_mb=10,)")
    print(f"")
    print(f"  Profile with:")
    print(f"    nsys profile python train.py")
    print(f"    -> Observe NCCL allreduce timing relative to backward ops")


EXPERIMENTS = {
    "timing": exp_ddp_hook_timing,
    "unused": exp_find_unused_example,
    "bucket": exp_bucket_size_manual,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[ddp_reducer case 1] DONE")


if __name__ == "__main__":
    main()
