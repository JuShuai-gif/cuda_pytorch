"""Functionalization case study 2: debug view + alias errors under torch.compile.

Companion script for functionalization/functionalization.md. Covers:
  1. Common mutation patterns that break torch.compile
  2. Slice assignment causing compile errors
  3. Rewrite in-place to functional for compile compatibility

Run:
    python 02_view_alias_error.py
"""

import sys

import torch


def exp_slice_assign():
    print("=" * 60)
    print("1. Slice assignment: a frequent compile breaker")
    print("=" * 60)

    def eager_f(x):
        """Slice assignment works in eager."""
        x[:, 0] = 0  # Write to slice
        return x.sum()

    x = torch.ones(3, 4)
    print(f"  Eager: {eager_f(x.clone())}")

    # Try compile
    try:
        compiled = torch.compile(eager_f)
        result = compiled(x.clone())
        print(f"  Compile OK: {result}")
    except Exception as e:
        print(f"  Compile ERROR: {str(e)[:120]}")
        print(f"  -> Slice assignment = view + inplace mutation")
        print(f"  -> Functionalization must decompose this")

    print(f"\n  How functionalization rewrites x[:, 0] = 0:")
    print(f"    1. x[:, 0] = 0  becomes:")
    print(f"    2. mask = zeros_like(x)")
    print(f"    3. mask[:, 0] = 1")
    print(f"    4. x = x * (1-mask) + zeros_like(x) * mask")
    print(f"    (functional scatter operation)")
    print()


def exp_inplace_chain():
    print("=" * 60)
    print("2. Chained inplace operations")
    print("=" * 60)

    def eager_chain(x):
        x = x + 1
        x.add_(2)      # in-place add
        x.mul_(3)      # in-place mul
        return x.sum()

    x = torch.randn(3)
    print(f"  Eager: {eager_chain(x.clone()):.4f}")

    try:
        compiled = torch.compile(eager_chain)
        result = compiled(x.clone())
        print(f"  Compile OK: {result:.4f}")
    except Exception as e:
        print(f"  Compile ERROR: {str(e)[:120]}")

    print(f"\n  Functionalization rewrites add_/mul_ to functional:")
    print(f"    x.add_(2) -> x = x.add(2)")
    print(f"    x.mul_(3) -> x = x.mul(3)")
    print()


def exp_rewrite_for_compile():
    print("=" * 60)
    print("3. Rewrite in-place ops for compile safety")
    print("=" * 60)

    class BadPattern(torch.nn.Module):
        """Uses in-place that may confuse compile."""
        def forward(self, x):
            x = x * 2
            x.add_(1)   # in-place
            x[:, 0] = 0  # slice assignment
            return x.sum()

    class GoodPattern(torch.nn.Module):
        """Functional rewrite - safe for compile."""
        def forward(self, x):
            x = x * 2
            x = x + 1   # functional add (no in-place)
            # Use mask approach for slice write
            mask = torch.zeros_like(x)
            mask[:, 0] = 1
            x = x * (1 - mask)  # zero out col 0
            return x.sum()

    x = torch.randn(3, 4)

    # Eager: both work
    bad = BadPattern()
    good = GoodPattern()
    print(f"  Eager (bad pattern):  {bad(x.clone()):.4f}")
    print(f"  Eager (good pattern): {good(x.clone()):.4f}")

    # Compile: good pattern should work
    try:
        compiled_good = torch.compile(GoodPattern())
        print(f"  Compile (good pattern): {compiled_good(x.clone()):.4f}")
    except Exception as e:
        print(f"  Compile (good) ERROR: {str(e)[:120]}")

    try:
        compiled_bad = torch.compile(BadPattern())
        print(f"  Compile (bad pattern): {compiled_bad(x.clone()):.4f}")
    except Exception as e:
        print(f"  Compile (bad) ERROR: {str(e)[:120]}")
    print()


EXPERIMENTS = {
    "slice": exp_slice_assign,
    "chain": exp_inplace_chain,
    "rewrite": exp_rewrite_for_compile,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[functionalization case 2] DONE")


if __name__ == "__main__":
    main()
