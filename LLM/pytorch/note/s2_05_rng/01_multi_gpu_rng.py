"""RNG case study 1: CPU vs CUDA RNG and multi-GPU reproducibility.

Companion script for rng/rng.md. Covers:
  1. CPU RNG vs CUDA RNG independence
  2. Multi-GPU seed setup
  3. Reproducibility verification

Run:
    python 01_multi_gpu_rng.py
"""

import sys

import torch


def exp_cpu_vs_cuda_rng():
    print("=" * 60)
    print("1. CPU RNG vs CUDA RNG: independent states")
    print("=" * 60)

    torch.manual_seed(42)

    # CPU random
    cpu_rand = torch.randn(3)
    print(f"  CPU randn (seed=42): {cpu_rand.tolist()}")

    # CUDA random (different state!)
    if torch.cuda.is_available():
        cuda_rand = torch.randn(3, device="cuda")
        print(f"  CUDA randn (seed=42): {cuda_rand.tolist()}")

        # They are different because CPU and CUDA have separate RNG
        print(f"  CPU == CUDA? {torch.allclose(cpu_rand, cuda_rand.cpu())}")
        print(f"  -> Different because CPU uses mt19937, CUDA uses Philox")
        print(f"  -> torch.manual_seed sets BOTH, but they produce different sequences")

    # Check RNG states
    cpu_state = torch.get_rng_state()
    print(f"\n  CPU RNG state bytes: {cpu_state.shape}")
    if torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state()
        print(f"  CUDA RNG state bytes: {cuda_state.shape}")
        # CUDA state is much smaller (seed+offset)
    print()


def exp_multi_gpu_rng():
    print("=" * 60)
    print("2. Multi-GPU RNG setup")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return

    n_gpus = torch.cuda.device_count()
    print(f"  Number of GPUs: {n_gpus}")

    if n_gpus < 2:
        print("  Need 2+ GPUs for multi-GPU demo")
        return

    # Per-GPU seeds
    base_seed = 42
    for gpu_id in range(min(n_gpus, 4)):
        with torch.cuda.device(gpu_id):
            torch.cuda.manual_seed(base_seed + gpu_id)
            r = torch.randn(3)
            print(f"  GPU {gpu_id} (seed={base_seed + gpu_id}): {r.tolist()}")

    # Verify independence
    print(f"\n  Each GPU has independent RNG state")
    print(f"  Key API:")
    print(f"    torch.cuda.manual_seed(seed)        -> set current device")
    print(f"    torch.cuda.manual_seed_all(seed)     -> set ALL devices")

    # Reset all to same seed
    torch.cuda.manual_seed_all(99)
    for gpu_id in range(min(n_gpus, 4)):
        with torch.cuda.device(gpu_id):
            r = torch.randn(3)
            print(f"  GPU {gpu_id} after manual_seed_all(99): {r.tolist()}")
    print()


def exp_reproducible_training():
    print("=" * 60)
    print("3. Reproducible training checklist")
    print("=" * 60)

    print(f"  Essential settings for reproducibility:")
    print(f"    1. torch.manual_seed(seed)")
    print(f"    2. torch.cuda.manual_seed(seed)")
    print(f"    3. torch.cuda.manual_seed_all(seed)  # all GPUs")
    print(f"    4. torch.backends.cudnn.deterministic = True")
    print(f"    5. torch.backends.cudnn.benchmark = False")
    print(f"    6. DataLoader with worker_init_fn for worker RNG")
    print(f"    7. export CUBLAS_WORKSPACE_CONFIG=:4096:8 (for CUDA >= 10.2)")

    # Demo: DataLoader worker_init_fn
    print(f"\n  DataLoader seed example:")
    print(f"    def worker_init_fn(worker_id):")
    print(f"        seed = torch.initial_seed() % (2**32)")
    print(f"        np.random.seed(seed)")
    print(f"        random.seed(seed)")
    print(f"")
    print(f"    loader = DataLoader(dataset, worker_init_fn=worker_init_fn)")

    # Verifying reproducibility
    print(f"\n  Verification test:")
    print(f"    def run_training():")
    print(f"        set_all_seeds(42)")
    print(f"        model = MyModel()")
    print(f"        train(model)")
    print(f"        return model.state_dict()")
    print(f"")
    print(f"    sd1 = run_training()")
    print(f"    sd2 = run_training()")
    print(f"    assert all(torch.equal(sd1[k], sd2[k]) for k in sd1)")
    print()


EXPERIMENTS = {
    "cpu_cuda": exp_cpu_vs_cuda_rng,
    "multi_gpu": exp_multi_gpu_rng,
    "repro": exp_reproducible_training,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[rng case 1] DONE")


if __name__ == "__main__":
    main()
