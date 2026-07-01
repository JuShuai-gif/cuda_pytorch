"""DDP Reducer case study 7: gradient sync hang detection.

Companion script for distributed_techniques/ddp_reducer/ddp_reducer.md. Covers:
  1. Common DDP hang causes
  2. Debugging stuck all-reduce
  3. Timeout and watchdog

Run:
    python 07_hang_detection.py
"""

import sys

import torch


def exp_hang_patterns():
    print("=" * 60)
    print("1. Common DDP hang patterns")
    print("=" * 60)

    patterns = [
        ("Unequal backward calls", "rank0: loss1.backward() + loss2.backward(), rank1: loss1.backward() only"),
        ("Different batch graph", "rank0 uses param_a; rank1 doesn't (no find_unused)"),
        ("Unequal model structure", "rank0 has extra layer not synced to rank1"),
        ("NCCL deadlock", "Two ranks call all-reduce in different order"),
        ("Python GIL + async", "Python callback during all-reduce holds GIL"),
    ]

    for title, desc in patterns:
        print(f"  {title:30s}: {desc}")

    print(f"\n  Detection tools:")
    print(f"    1. Set NCCL_DEBUG=INFO for NCCL level logs")
    print(f"    2. torch.distributed.set_debug_level(DETAIL)")
    print(f"    3. GLOO_SOCKET_IFNAME for TCP backend")
    print(f"    4. TORCH_DISTRIBUTED_DEBUG=DETAIL (PyTorch >= 1.10)")
    print()


def exp_watchdog():
    print("=" * 60)
    print("2. DDP watchdog and timeout")
    print("=" * 60)

    import os
    timeout = os.environ.get("TORCH_DIST_INIT_TIMEOUT", "default=600s")
    print(f"  Default timeout: {timeout}")
    print(f"  Custom timeout:  TORCH_DIST_INIT_TIMEOUT=300")
    print(f"")

    print(f"  DDP watchdog settings:")
    print(f"    - Default timeout: 600s (10 min)")
    print(f"    - Watchdog checks for stuck operations")
    print(f"    - Kills process if no progress within timeout")
    print(f"")

    print(f"  To enable detail debug:")
    print(f"    TORCH_DISTRIBUTED_DEBUG=DETAIL python train.py")
    print(f"    -> Checks parameter usage consistency")
    print(f"    -> Detects mismatched forward/backward between ranks")
    print()


def exp_deterministic_debug():
    print("=" * 60)
    print("3. Deterministic debugging for hangs")
    print("=" * 60)

    print(f"  Reproduce hang deterministically:")
    print(f"    1. Fix all seeds")
    print(f"    2. Disable data shuffling")
    print(f"    3. Use single GPU per process")
    print(f"    4. Add barrier() calls to narrow down hang location")
    print(f"")

    print(f"  Debug script template:")
    print(f"    dist.barrier()                             # sync before")
    print(f"    print(f'rank {rank}: starting forward')")
    print(f"    output = model(data)")
    print(f"    dist.barrier()                             # sync after forward")
    print(f"    print(f'rank {rank}: starting backward')")
    print(f"    loss.backward()")
    print(f"    dist.barrier()                             # sync after backward")
    print(f"    print(f'rank {rank}: backward done')")
    print()


EXPERIMENTS = {
    "patterns": exp_hang_patterns,
    "watchdog": exp_watchdog,
    "debug": exp_deterministic_debug,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[ddp_reducer case 7] DONE")


if __name__ == "__main__":
    main()
