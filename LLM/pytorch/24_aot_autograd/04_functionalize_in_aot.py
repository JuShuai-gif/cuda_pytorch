"""AOTAutograd case study 4: functionalization in AOTAutograd pipeline.

Companion script for aot_autograd/aot_autograd.md. Covers:
  1. AOTAutograd with mutation in model
  2. Functionalization rewrite visible in graph
  3. BN running stats handling

Run:
    python 04_functionalize_in_aot.py
"""

import sys

import torch
from torch._functorch.aot_autograd import aot_function


def exp_mutation_in_aot():
    print("=" * 60)
    print("1. AOTAutograd handles mutation in captured graph")
    print("=" * 60)

    def fn_with_mutation(x):
        y = x + 1
        y.mul_(2)  # in-place mutation
        return y.sum()

    def print_graph(gm, inputs):
        print(f"  Graph nodes:")
        for node in gm.graph.nodes:
            print(f"    {node.op:10s} {str(node.target)[:60]}")
        return gm

    x = torch.randn(4, requires_grad=True)
    aot_fn = aot_function(fn_with_mutation, print_graph, print_graph)
    loss = aot_fn(x)
    loss.backward()
    print(f"\n  AOTAutograd functionalizes mul_() -> mul() in the graph")
    print()


def exp_bn_training_mode():
    print("=" * 60)
    print("2. BatchNorm training mode: internal mutation")
    print("=" * 60)

    bn = torch.nn.BatchNorm1d(8)
    bn.train()

    def bn_forward(x):
        return bn(x)

    def print_fw(gm, inputs):
        print(f"  BN forward graph ({len(list(gm.graph.nodes))} nodes)")
        return gm

    x = torch.randn(4, 8, requires_grad=True)
    aot_bn = aot_function(bn_forward, print_fw, lambda gm, i: gm)
    loss = aot_bn(x)
    loss.sum().backward()

    print(f"\n  BN has internal mutation (running_mean, running_var)")
    print(f"  AOTAutograd functionalizes these writes -> pure graph")
    print()


def exp_requires_grad_control():
    print("=" * 60)
    print("3. requires_grad control in AOTAutograd")
    print("=" * 60)

    def fn(x, flag):
        z = x @ torch.eye(8)
        if flag:
            z.requires_grad_(True)
        return z.sum()

    # AOTAutograd needs to know requires_grad at trace time
    # data-dependent requires_grad_ changes cause issues
    x = torch.randn(4, 8, requires_grad=True)

    def printer(gm, inputs):
        all_nodes = list(gm.graph.nodes)
        print(f"  Graph has {len(all_nodes)} nodes")
        for n in all_nodes[:5]:
            print(f"    {n.op:10s} {str(n.target)[:50]}")
        if len(all_nodes) > 5:
            print(f"    ... {len(all_nodes) - 5} more nodes")
        return gm

    try:
        aot_fn = aot_function(lambda x: fn(x, True), printer, printer)
        loss = aot_fn(x)
        loss.backward()
        print(f"  AOTAutograd OK")
    except Exception as e:
        print(f"  Error: {str(e)[:100]}")
    print()


EXPERIMENTS = {
    "mutation": exp_mutation_in_aot,
    "bn": exp_bn_training_mode,
    "grad": exp_requires_grad_control,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[aot_autograd case 4] DONE")


if __name__ == "__main__":
    main()
