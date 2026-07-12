"""Design Patterns case study 1: RAII in PyTorch.

Companion script for 40_design_patterns/design_patterns.md.

Run:
    python 01_raii_guard.py
"""

import sys
import torch


def exp_device_guard():
    print("=" * 60)
    print("1. CUDAGuard: RAII device switching")
    print("=" * 60)
    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return
    with torch.cuda.device(0):
        print(f"  Enter GPU 0: current = {torch.cuda.current_device()}")
        with torch.cuda.device(1):
            print(f"    Enter GPU 1: current = {torch.cuda.current_device()}")
        print(f"  Exit GPU 1 (auto-restore): current = {torch.cuda.current_device()}")
    print(f"  Exit GPU 0: current = {torch.cuda.current_device()}")


def exp_grad_guard():
    print("=" * 60)
    print("2. AutoDispatchBelowAutograd: torch.no_grad()")
    print("=" * 60)
    x = torch.randn(3, requires_grad=True)
    with torch.no_grad():
        y = x + 1
        print(f"  no_grad: requires_grad={y.requires_grad}, grad_fn={y.grad_fn}")
    z = x + 1
    print(f"  normal:  requires_grad={z.requires_grad}, grad_fn={z.grad_fn}")


def exp_autocast_guard():
    print("=" * 60)
    print("3. AutocastGuard: torch.autocast RAII")
    print("=" * 60)
    if not torch.cuda.is_available():
        return
    x = torch.randn(3, 3, device="cuda", dtype=torch.float32)
    with torch.autocast("cuda"):
        y = x @ x
        print(f"  autocast: output dtype={y.dtype} (should be fp16)")
    z = x @ x
    print(f"  normal:   output dtype={z.dtype} (fp32)")


EXPERIMENTS = {"device": exp_device_guard, "grad": exp_grad_guard, "autocast": exp_autocast_guard}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}'")
            continue
        EXPERIMENTS[name]()
    print("[design_patterns case 1] DONE")


if __name__ == "__main__":
    main()
