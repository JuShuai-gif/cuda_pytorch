"""Functionalization case study 5: torch.compile interaction.

Companion script for functionalization/functionalization.md. Covers:
  1. compile + mutation: what gets functionalized
  2. Graph break from mutation
  3. Re-compile triggers from mutation patterns

Run:
    python 05_compile_interaction.py
"""

import sys

import torch


def exp_compile_inplace_add():
    print("=" * 60)
    print("1. torch.compile with in-place add: what happens")
    print("=" * 60)

    # In-place add inside compile
    def f(x):
        y = x + 1
        y.add_(2)
        return y.sum()

    x = torch.randn(4)
    expected = f(x)

    compiled = torch.compile(f)
    result = compiled(x)
    print(f"  Eager:    {expected:.4f}")
    print(f"  Compiled: {result:.4f}")
    print(f"  Match: {torch.allclose(expected, result)}")
    print(f"  -> Functionalization rewrote add_() to add() in compiled graph")
    print()


def exp_compile_residual_mutation():
    print("=" * 60)
    print("2. Residual connection with mutation pattern")
    print("=" * 60)

    # Residual pattern: y = f(x) + x
    # This is safe (no mutation)
    # But volatile if we use x += f(x)

    def safe_residual(x, w):
        return torch.relu(x @ w) + x

    def unsafe_residual(x, w):
        x += torch.relu(x @ w)  # modifies x!
        return x

    x = torch.randn(4, 8)
    w = torch.randn(8, 8)

    compiled_safe = torch.compile(safe_residual)
    r_safe = compiled_safe(x, w)
    print(f"  Safe residual OK: {r_safe.sum():.4f}")

    try:
        compiled_unsafe = torch.compile(unsafe_residual)
        r_unsafe = compiled_unsafe(x.clone(), w)
        print(f"  Unsafe (in-place) OK: {r_unsafe.sum():.4f}")
    except Exception as e:
        print(f"  Unsafe error: {str(e)[:80]}")
    print()


def exp_compile_graph_break_from_mutation():
    print("=" * 60)
    print("3. Graphs break from mutation across boundaries")
    print("=" * 60)

    # Dynamo can capture mutation inside the compiled region
    # But mutation across compile/eager boundary is a graph break

    @torch.compile
    def compiled_part(x):
        return x * 2

    x = torch.randn(4)
    y = compiled_part(x)  # compiled
    x.add_(1)             # eager mutation after compile
    z = x + y

    print(f"  Post-compile mutation + combine: {z.sum():.4f}")
    print(f"  -> Mutation after compile boundary = eager section")
    print(f"  -> Each eager section = potential graph break")
    print()


EXPERIMENTS = {
    "add": exp_compile_inplace_add,
    "residual": exp_compile_residual_mutation,
    "break": exp_compile_graph_break_from_mutation,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[functionalization case 5] DONE")


if __name__ == "__main__":
    main()
