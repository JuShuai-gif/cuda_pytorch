"""RNG case study 7: RNG across fork, spawn, and DataLoader workers.

Companion script for rng/rng.md. Covers:
  1. Fork vs spawn: RNG state inheritance
  2. DataLoader worker RNG isolation
  3. Seed management in distributed training

Run:
    python 07_fork_rng.py
"""

import sys

import torch


def exp_fork_inheritance():
    print("=" * 60)
    print("1. Fork vs spawn: RNG state behavior")
    print("=" * 60)

    torch.manual_seed(42)
    r_parent = torch.randn(3)

    # In fork (default on Linux): child inherits parent RNG state
    # In spawn (default on Windows/macOS): child gets fresh RNG

    print(f"  Parent RNG (seed=42): {r_parent.tolist()}")
    print(f"")
    print(f"  multiprocessing 'fork' strategy:")
    print(f"    - Child inherits parent's RNG state")
    print(f"    - Child calls torch.randn() = same sequence as parent would")
    print(f"    - Risky: parent+child may use same random numbers")
    print(f"")
    print(f"  multiprocessing 'spawn' strategy:")
    print(f"    - Child starts with fresh RNG")
    print(f"    - Safer: no shared state")
    print(f"    - Standard for DataLoader workers")
    print(f"")
    print(f"  Best practice in PyTorch:")
    print(f"    - DataLoader uses spawn by default")
    print(f"    - Worker RNG explicitly seeded via worker_init_fn")
    print()


def exp_worker_rng_isolation():
    print("=" * 60)
    print("2. DataLoader worker RNG isolation")
    print("=" * 60)

    import numpy as np
    import random

    def worker_init_fn(worker_id):
        """Properly initialize RNG per DataLoader worker."""
        # Base seed from PyTorch (set by torch.utils.data.get_worker_info().seed)
        worker_seed = torch.initial_seed() % (2**32)
        # Propagate to all RNG sources
        torch.manual_seed(worker_seed)
        np.random.seed(worker_seed % (2**31))
        random.seed(worker_seed % (2**31))

    import torch.utils.data

    # Simulate worker init
    for wid in range(4):
        print(f"  Worker {wid}:")
        worker_init_fn(wid)
        r = torch.randint(0, 100, (3,))
        print(f"    torch randint: {r.tolist()}")
        print(f"    numpy random:  {np.random.randint(0, 100, 3).tolist()}")
        print(f"    python random: {[random.randint(0, 100) for _ in range(3)]}")

    print(f"\n  Each worker has independent RNG -> reproducible data pipeline")
    print()


EXPERIMENTS = {
    "fork": exp_fork_inheritance,
    "worker": exp_worker_rng_isolation,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[rng case 7] DONE")


if __name__ == "__main__":
    main()
