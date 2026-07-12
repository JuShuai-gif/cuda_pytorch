"""RNG case study 4: DataLoader worker seed and Philox offset manipulation.

Companion script for rng/rng.md. Covers:
  1. DataLoader worker_init_fn
  2. Philox offset inspection
  3. RNG state serialization

Run:
    python 04_dataloader_rng.py
"""

import sys

import torch


def exp_worker_init():
    print("=" * 60)
    print("1. DataLoader worker RNG: worker_init_fn")
    print("=" * 60)

    def worker_init_fn(worker_id):
        """Each DataLoader worker gets a unique RNG seed."""
        worker_seed = torch.initial_seed() % (2**32)
        torch.manual_seed(worker_seed)
        print(f"    Worker {worker_id}: seed = {worker_seed}")

    print(f"  worker_init_fn sets RNG per-worker:")
    worker_init_fn(0)
    worker_init_fn(1)
    worker_init_fn(2)

    print(f"\n  In DataLoader:")
    print(f"    loader = DataLoader(dataset, num_workers=4,")
    print(f"                        worker_init_fn=worker_init_fn)")
    print()


def exp_philox_offset():
    print("=" * 60)
    print("2. Philox offset behavior in CUDA RNG")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return

    # Reset CUDA RNG
    torch.cuda.manual_seed(42)
    gen = torch.cuda.default_generators[0]

    # CUDA Philox: seed + offset counter
    print(f"  Default CUDA generator:")
    print(f"    seed:   {gen.initial_seed()}")

    # Generate random numbers and observe offset increment
    torch.cuda.synchronize()
    r1 = torch.randn(1024, 1024, device="cuda")  # consumes 1M random numbers
    torch.cuda.synchronize()

    r2 = torch.randn(1024, 1024, device="cuda")  # continues from offset
    torch.cuda.synchronize()

    # Verify they are different sequences
    print(f"  r1[0,0]: {r1[0, 0].item():.6f}")
    print(f"  r2[0,0]: {r2[0, 0].item():.6f}")
    print(f"  Same? {torch.allclose(r1, r2)}")
    print(f"  -> Philox offset auto-increments after each op")
    print()


def exp_rng_state_serialization():
    print("=" * 60)
    print("3. RNG state save/restore")
    print("=" * 60)

    # CPU RNG state
    torch.manual_seed(42)
    r_before = torch.randn(3)
    state = torch.get_rng_state()

    torch.randn(10)  # consume some entropy
    torch.set_rng_state(state)  # restore
    r_after = torch.randn(3)

    print(f"  CPU RNG state save/restore:")
    print(f"    Before: {r_before.tolist()}")
    print(f"    After restore: {r_after.tolist()}")
    print(f"    Match: {torch.allclose(r_before, r_after)}")

    # CUDA RNG state
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        r_cuda_before = torch.randn(3, device="cuda")
        cuda_state = torch.cuda.get_rng_state()

        torch.randn(10, device="cuda")  # consume
        torch.cuda.set_rng_state(cuda_state)
        r_cuda_after = torch.randn(3, device="cuda")

        print(f"\n  CUDA RNG state save/restore:")
        print(f"    State size: {cuda_state.shape} (much smaller than CPU)")
        print(f"    Match: {torch.allclose(r_cuda_before, r_cuda_after)}")
    print()


EXPERIMENTS = {
    "worker": exp_worker_init,
    "philox": exp_philox_offset,
    "serialize": exp_rng_state_serialization,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[rng case 4] DONE")


if __name__ == "__main__":
    main()
