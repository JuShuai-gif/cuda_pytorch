"""RNG case study 5: deterministic algorithms and cuDNN interaction.

Companion script for rng/rng.md. Covers:
  1. cuDNN deterministic mode
  2. CUDA convolution reproducibility
  3. Full reproducibility checklist

Run:
    python 05_deterministic_cuda.py
"""

import sys

import torch


def exp_cudnn_deterministic():
    print("=" * 60)
    print("1. cuDNN deterministic mode: reproducibility test")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return

    conv = torch.nn.Conv2d(3, 16, 3, padding=1).cuda()
    x = torch.randn(2, 3, 32, 32, device="cuda")

    # Non-deterministic (default)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    outputs = []
    for _ in range(5):
        torch.manual_seed(0)
        outputs.append(conv(x).sum().item())

    all_same = len(set(f"{o:.6f}" for o in outputs)) == 1
    print(f"  Non-deterministic mode (benchmark=True):")
    print(f"    5 runs: {'ALL SAME' if all_same else 'DIFFERENT'}")

    # Deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    outputs_det = []
    for _ in range(5):
        torch.manual_seed(0)
        outputs_det.append(conv(x).sum().item())

    all_same_det = len(set(f"{o:.6f}" for o in outputs_det)) == 1
    print(f"\n  Deterministic mode:")
    print(f"    5 runs: {'ALL SAME' if all_same_det else 'DIFFERENT'}")

    # Reset
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    print()


def exp_full_checklist():
    print("=" * 60)
    print("2. Full reproducibility checklist")
    print("=" * 60)

    # Demonstrate each step
    seed = 42

    print(f"  Checklist for reproducible training:")
    checks = [
        ("python RNG", "random.seed(seed)", True),
        ("numpy RNG", "np.random.seed(seed)", True),
        ("torch CPU RNG", "torch.manual_seed(seed)", True),
        ("torch CUDA RNG (current)", "torch.cuda.manual_seed(seed)", True),
        ("torch CUDA RNG (all GPUs)", "torch.cuda.manual_seed_all(seed)", True),
        ("cuDNN deterministic", "torch.backends.cudnn.deterministic = True", True),
        ("cuDNN benchmark off", "torch.backends.cudnn.benchmark = False", True),
        ("CUDA env", 'os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"', True),
    ]

    for name, cmd, status in checks:
        marker = "[x]" if status else "[ ]"
        print(f"  {marker} {name:30s}: {cmd}")
    print()


EXPERIMENTS = {
    "cudnn": exp_cudnn_deterministic,
    "checklist": exp_full_checklist,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[rng case 5] DONE")


if __name__ == "__main__":
    main()
