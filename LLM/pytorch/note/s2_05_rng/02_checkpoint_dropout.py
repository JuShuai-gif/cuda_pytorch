"""RNG case study 2: activation checkpoint + dropout RNG interaction.

Companion script for rng/rng.md. Covers:
  1. Checkpoint RNG save/restore
  2. Compare reentrant vs non-reentrant checkpoint
  3. Verify RNG correctness

Run:
    python 02_checkpoint_dropout.py
"""

import sys

import torch
from torch.utils.checkpoint import checkpoint


def exp_rng_consistency():
    print("=" * 60)
    print("1. Verify checkpoint RNG consistency")
    print("=" * 60)

    dropout = torch.nn.Dropout(p=0.5)
    x = torch.ones(8, requires_grad=True)

    def f(t):
        return dropout(t).sum()

    # Run 1: no checkpoint
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    y1 = f(x)

    # Run 2: with checkpoint (non-reentrant)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    y2 = checkpoint(f, x, use_reentrant=False)

    print(f"  Without checkpoint: {y1.item():.4f}")
    print(f"  With checkpoint (non-reentrant): {y2.item():.4f}")
    print(f"  Match: {torch.allclose(y1, y2)}")
    print(f"  -> Non-reentrant checkpoint preserves RNG state")
    print()


def exp_reentrant_comparison():
    print("=" * 60)
    print("2. Reentrant vs Non-reentrant RNG behavior")
    print("=" * 60)

    dropout = torch.nn.Dropout(p=0.5)
    x = torch.ones(8, requires_grad=True)

    def f(t):
        # Generate two random values (consumes RNG state twice)
        d1 = dropout(t)
        d2 = dropout(t + 1)
        return (d1 + d2).sum()

    # Non-reentrant
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    try:
        y1 = checkpoint(f, x, use_reentrant=False)
        print(f"  Non-reentrant result: {y1.item():.4f}")
    except Exception as e:
        print(f"  Non-reentrant error: {str(e)[:100]}")

    # Reentrant
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    try:
        y2 = checkpoint(f, x, use_reentrant=True)
        print(f"  Reentrant result: {y2.item():.4f}")
    except Exception as e:
        print(f"  Reentrant error: {str(e)[:100]}")

    print(f"\n  Difference:")
    print(f"    Non-reentrant: saves/restores RNG state automatically")
    print(f"    Reentrant:     user is responsible for RNG management")
    print()


def exp_gradient_verification():
    print("=" * 60)
    print("3. Verify gradient with checkpoint + dropout")
    print("=" * 60)

    dropout = torch.nn.Dropout(p=0.3)
    x = torch.randn(16, requires_grad=True)

    def model_fn(t):
        h1 = torch.nn.functional.linear(t, torch.eye(16))
        h2 = dropout(h1)
        return h2.sum()

    # Eager gradient
    torch.manual_seed(0)
    x1 = x.clone().detach().requires_grad_(True)
    loss1 = model_fn(x1)
    loss1.backward()

    # Checkpoint gradient
    torch.manual_seed(0)
    x2 = x.clone().detach().requires_grad_(True)
    loss2 = checkpoint(model_fn, x2, use_reentrant=False)
    loss2.backward()

    grad_diff = (x1.grad - x2.grad).abs().max().item()
    print(f"  Max gradient diff: {grad_diff:.2e}")
    print(f"  Gradients match: {torch.allclose(x1.grad, x2.grad)}")
    print()


EXPERIMENTS = {
    "consistency": exp_rng_consistency,
    "reentrant": exp_reentrant_comparison,
    "gradient": exp_gradient_verification,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[rng case 2] DONE")


if __name__ == "__main__":
    main()
