"""RNG case study 6: Generator API for fine-grained control.

Companion script for rng/rng.md. Covers:
  1. torch.Generator for per-op RNG
  2. Multiple generators, multiple streams
  3. Generator in DataLoader

Run:
    python 06_generator_api.py
"""

import sys

import torch


def exp_generator_basics():
    print("=" * 60)
    print("1. torch.Generator: per-operation RNG control")
    print("=" * 60)

    gen_a = torch.Generator()
    gen_a.manual_seed(42)

    gen_b = torch.Generator()
    gen_b.manual_seed(42)

    # Same seed = same sequence
    r_a1 = torch.randn(3, generator=gen_a)
    r_b1 = torch.randn(3, generator=gen_b)
    print(f"  gen_a(42) randn: {r_a1.tolist()}")
    print(f"  gen_b(42) randn: {r_b1.tolist()}")
    print(f"  Match: {torch.allclose(r_a1, r_b1)}")

    # Continue using gen_a
    r_a2 = torch.randn(3, generator=gen_a)
    r_b2 = torch.randn(3, generator=gen_b)  # independent sequence
    print(f"\n  gen_a second call: {r_a2.tolist()}")
    print(f"  gen_b second call: {r_b2.tolist()}")
    print()


def exp_multi_generator():
    print("=" * 60)
    print("2. Independent generators for different modules")
    print("=" * 60)

    # Data augmentation vs dropout: separate RNG
    aug_gen = torch.Generator()
    aug_gen.manual_seed(123)

    drop_gen = torch.Generator()
    drop_gen.manual_seed(456)

    # Data augmentation random
    x_aug = torch.randn(3, generator=aug_gen)
    # Dropout random
    x_drop = torch.randn(3, generator=drop_gen)

    print(f"  Augmentation RNG: {x_aug.tolist()}")
    print(f"  Dropout RNG:      {x_drop.tolist()}")
    print(f"  -> Generators are independent, seeding one doesn't affect others")

    # Global generator is separate
    global_r = torch.randn(3)
    print(f"  Global RNG:       {global_r.tolist()} (different)")
    print()


def exp_cuda_generator():
    print("=" * 60)
    print("3. CUDA Generator: device-local RNG")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return

    cuda_gen = torch.Generator(device="cuda")
    cuda_gen.manual_seed(42)

    r1 = torch.randn(3, device="cuda", generator=cuda_gen)
    r2 = torch.randn(3, device="cuda", generator=cuda_gen)

    print(f"  CUDA gen randn 1: {r1.tolist()}")
    print(f"  CUDA gen randn 2: {r2.tolist()}")

    # State introspection
    state = cuda_gen.get_state()
    print(f"  State bytes: {state.shape}")

    # Save and restore
    r_before = torch.randn(3, device="cuda", generator=cuda_gen)
    cuda_gen.set_state(state)
    r_restored = torch.randn(3, device="cuda", generator=cuda_gen)
    print(f"  Restored randn: {r_restored.tolist()}")
    print(f"  Match: {torch.allclose(r_before, r_restored)}")
    print()


EXPERIMENTS = {
    "basics": exp_generator_basics,
    "multi": exp_multi_generator,
    "cuda": exp_cuda_generator,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[rng case 6] DONE")


if __name__ == "__main__":
    main()
