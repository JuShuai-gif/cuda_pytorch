"""Graph Passes case study 1: FX graph transformation and optimization.

Companion script for graph_passes/ directory. Covers:
  1. FX graph pass: node replacement
  2. Pattern matching
  3. Graph optimization example

Run:
    python 03_graph_transform.py
"""

import sys

import torch


def exp_fx_graph_transform():
    print("=" * 60)
    print("1. FX graph transformation: replace nodes")
    print("=" * 60)

    import torch.fx as fx

    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(8, 8)

        def forward(self, x):
            return self.linear(x).relu().mean()

    model = SimpleModel()
    gm = fx.symbolic_trace(model)

    print(f"  Original graph:")
    gm.graph.print_tabular()

    # Graph pass: replace relu with gelu
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.relu:
            with gm.graph.inserting_after(node):
                new_node = gm.graph.call_function(torch.nn.functional.gelu, (node.args[0],))
                node.replace_all_uses_with(new_node)

    gm.graph.lint()
    gm.recompile()

    print(f"\n  After replacing relu -> gelu:")
    print()

    x = torch.randn(4, 8)
    y = gm(x)
    print(f"  Output: {y.item():.4f}")
    print()


EXPERIMENTS = {
    "transform": exp_fx_graph_transform,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[graph_passes case 1] DONE")


if __name__ == "__main__":
    main()
