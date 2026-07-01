"""AOTAutograd case study 6: trace through graph break and reentry.

Companion script for aot_autograd/aot_autograd.md. Covers:
  1. Dynamo graph break -> AOTAutograd reentry
  2. Multiple subgraphs compilation
  3. CacheEntry and recompilation

Run:
    python 06_graph_break_reentry.py
"""

import sys

import torch


def exp_graph_break():
    print("=" * 60)
    print("1. Graph break produces multiple AOTAutograd calls")
    print("=" * 60)

    # Function with potential graph breaks
    def f_with_break(x):
        y = x.relu()
        y = y + 1
        # torch.where with non-tensor condition -> graph break
        if x.sum() > 0:
            y = y * 2
        else:
            y = y / 2
        return y.sum()

    x = torch.randn(4, 8, requires_grad=True)

    # Eager works normally
    y_eager = f_with_break(x)
    y_eager.backward()
    print(f"  Eager: {y_eager.item():.4f}")

    # Compile: Dynamo handles graph break, AOTAutograd per-subgraph
    compiled = torch.compile(f_with_break)

    try:
        y_compiled = compiled(x.clone().detach().requires_grad_(True))
        y_compiled.backward()
        print(f"  Compile: {y_compiled.item():.4f}")
    except Exception as e:
        print(f"  Compile: {str(e)[:80]}")
    print()


def exp_recompilation():
    print("=" * 60)
    print("2. Guard failure -> recompile via AOTAutograd")
    print("=" * 60)

    compile_count = [0]

    def counting_backend(gm, inputs):
        compile_count[0] += 1
        return gm

    def f(x):
        return x.relu().sum()

    compiled = torch.compile(f, backend=counting_backend)

    # Trigger recompilations via shape changes
    for shape in [(4, 8), (8, 8), (4, 8), (4, 16)]:
        x = torch.randn(*shape)
        try:
            compiled(x)
        except Exception:
            pass
        print(f"  shape={list(shape)}: compilations={compile_count[0]}")

    print(f"\n  Total compilations: {compile_count[0]}")
    print(f"  Each recompile = AOTAutograd traces new joint graph")
    print()


def exp_multi_subgraph():
    print("=" * 60)
    print("3. Multiple subgraphs from one function")
    print("=" * 60)

    # Complex function with multiple break points
    import math

    @torch.compile
    def complex_fn(x):
        y = x.relu()
        y = y * 2

        # Break 1: Python int
        n = int(y.sum().item())
        y = y + n

        # Break 2: numpy/data-dependent
        y = y.sin()

        # Break 3: list comprehension
        vals = [y[i].sum() for i in range(len(y))]
        return sum(vals)

    x = torch.randn(4, 4, requires_grad=True)
    try:
        y = complex_fn(x)
        y.backward()
        print(f"  Complex function OK: {y.item():.4f}")
        print(f"  Graph breaks split function into subgraphs")
        print(f"  Each subgraph -> separate AOTAutograd trace")
    except Exception as e:
        print(f"  Error: {str(e)[:100]}")
    print()


EXPERIMENTS = {
    "break": exp_graph_break,
    "recompile": exp_recompilation,
    "multi": exp_multi_subgraph,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[aot_autograd case 6] DONE")


if __name__ == "__main__":
    main()
