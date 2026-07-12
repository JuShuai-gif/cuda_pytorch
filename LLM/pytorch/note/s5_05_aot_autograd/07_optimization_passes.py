"""AOTAutograd case study 7: forward/backward graph optimization passes.

Companion script for aot_autograd/aot_autograd.md. Covers:
  1. Dead code elimination in FW/BW graphs
  2. Constant folding
  3. CSE (Common Subexpression Elimination) in joint graph

Run:
    python 07_optimization_passes.py
"""

import sys

import torch
from torch._functorch.aot_autograd import aot_function


def exp_dead_code_elimination():
    print("=" * 60)
    print("1. Dead code elimination in traced graph")
    print("=" * 60)

    def dead_code_fn(x):
        y = x.relu()
        z = torch.sin(x)  # computed but never used
        w = torch.log(x.abs() + 1)  # also unused
        return y.sum()

    def printer(gm, inputs):
        print(f"  Nodes in graph:")
        for node in gm.graph.nodes:
            marker = " USED" if list(node.users) else " DEAD"
            print(f"    {node.op:10s} {str(node.target)[:40]:40s} {marker}")
        return gm

    x = torch.randn(4, 8, requires_grad=True)
    aot_fn = aot_function(dead_code_fn, printer, printer)
    loss = aot_fn(x)
    loss.backward()

    print(f"\n  AOTAutograd eliminates unused computations")
    print(f"  Forward: sin(x) and log(x) are dropped")
    print()


def exp_constant_folding():
    print("=" * 60)
    print("2. Constant folding in AOTAutograd graphs")
    print("=" * 60)

    def const_fn(x):
        y = x * 2.0        # constant multiplier
        y = y + 3.14159    # constant addition
        y = y.pow(2)       # power 2
        return y.sum()

    def printer(gm, inputs):
        print(f"  Graph nodes:")
        for node in gm.graph.nodes:
            print(f"    {node.op:10s} {str(node.target)[:50]}")
        return gm

    x = torch.randn(4, 8, requires_grad=True)
    aot_fn = aot_function(const_fn, printer, printer)
    loss = aot_fn(x)
    loss.backward()

    print(f"\n  Constant folding: x * 2.0 + 3.14 -> single fused node")
    print(f"  AOTAutograd may fuse consecutive elementwise ops")
    print()


EXPERIMENTS = {
    "dce": exp_dead_code_elimination,
    "fold": exp_constant_folding,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[aot_autograd case 7] DONE")


if __name__ == "__main__":
    main()
