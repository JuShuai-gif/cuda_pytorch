"""AOTAutograd case study 1: compare eager autograd with AOTAutograd.

Companion script for aot_autograd/aot_autograd.md. Covers:
  1. Eager autograd: dynamic node creation
  2. AOTAutograd: joint graph tracing
  3. Compare memory and execution differences

Run:
    python 01_compare_eager_aot.py
"""

import sys

import torch


def f(x, w):
    return (x @ w).sin().sum()


def exp_eager_autograd():
    print("=" * 60)
    print("1. Eager autograd: dynamic graph construction")
    print("=" * 60)

    x = torch.randn(4, 8, requires_grad=True)
    w = torch.randn(8, 3, requires_grad=True)

    # Eager: each operation creates an Autograd Node at runtime
    loss = f(x, w)
    print(f"  Forward: loss={loss.item():.6f}")
    print(f"  loss.grad_fn: {loss.grad_fn}")

    # Backward traverses the Node graph
    loss.backward()
    print(f"  x.grad shape: {list(x.grad.shape)}")
    print(f"  w.grad shape: {list(w.grad.shape)}")

    print(f"\n  Eager creates Autograd Nodes on-the-fly")
    print(f"  Each op is recorded as a separate node")
    print()


def exp_aot_autograd():
    print("=" * 60)
    print("2. AOTAutograd: trace entire forward+backward")
    print("=" * 60)

    from torch._functorch.aot_autograd import aot_function

    def fw_printer(gm, inputs):
        print(f"  ---- Forward Graph ----")
        for node in gm.graph.nodes:
            print(f"    {node.op:10s} {node.target}")
        return gm

    def bw_printer(gm, inputs):
        print(f"\n  ---- Backward Graph ----")
        for node in gm.graph.nodes:
            print(f"    {node.op:10s} {node.target}")
        return gm

    x = torch.randn(4, 8, requires_grad=True)
    w = torch.randn(8, 3, requires_grad=True)

    aot_f = aot_function(f, fw_printer, bw_printer)
    loss = aot_f(x, w)
    loss.backward()

    print(f"\n  AOTAutograd captures the entire computation as FX graphs")
    print(f"  Forward and backward are separate compiled graphs")
    print()


def exp_no_grad_interaction():
    print("=" * 60)
    print("3. torch.no_grad inside AOTAutograd")
    print("=" * 60)

    # AOTAutograd handles no_grad by not tracing that section
    def g(x, w):
        z = x @ w
        with torch.no_grad():
            y = z.relu()  # not included in autograd graph
        return (z * y).sum()

    from torch._functorch.aot_autograd import aot_function

    def printer(gm, inputs):
        print(f"  Graph nodes:")
        for node in gm.graph.nodes:
            print(f"    {node.op:10s} {node.target}")
        return gm

    x = torch.randn(4, 8, requires_grad=True)
    w = torch.randn(8, 3, requires_grad=True)

    aot_g = aot_function(g, printer, printer)
    loss = aot_g(x, w)
    loss.backward()
    print()


EXP_list = [
    ("eager", exp_eager_autograd),
    ("aot", exp_aot_autograd),
    ("nograd", exp_no_grad_interaction),
]


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else [name for name, _ in EXP_list]
    exp_map = dict(EXP_list)
    for name in exps:
        if name not in exp_map:
            print(f"unknown exp '{name}', choose from: {list(exp_map)}")
            continue
        exp_map[name]()

    print("[aot_autograd case 1] DONE")


if __name__ == "__main__":
    main()
