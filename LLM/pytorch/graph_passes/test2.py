"""Graph passes advanced: shape propagation, split_module, custom PassBase.

Companion script for graph_passes/graph_passes.md.
  1. shape_prop:             propagate shapes through FX graph
  2. split_module:           split graph across devices
  3. custom PassBase:        write your own graph pass
  4. PassManager:            orchestrate multiple passes

Run:
    python test2.py               # full demo
    python test2.py shape_prop    # shape propagation
    python test2.py split         # module splitting
    python test2.py custom_pass   # custom PassBase
    python test2.py pass_manager  # PassManager orchestration
"""

import sys
import torch
import torch.nn as nn
import torch.fx as fx


# ============ 1. Shape propagation ============
def exp_shape_prop():
    print("=" * 60)
    print("1. Shape propagation through FX graph")
    print("=" * 60)

    class DynamicModel(nn.Module):
        def forward(self, x):
            a = x * 2
            b = a.view(x.size(0), -1)
            c = b.sum(dim=-1)
            return c

    model = DynamicModel()
    gm = fx.symbolic_trace(model)

    # Propagate shapes using fake tensors
    from torch.fx.passes.shape_prop import ShapeProp

    ShapeProp(gm).propagate(torch.randn(4, 3, 16))

    print("  Shape propagation result:")
    for node in gm.graph.nodes:
        if "tensor_meta" in node.meta:
            meta = node.meta["tensor_meta"]
            print(f"    {node.name:10s} shape={list(meta.shape)} dtype={meta.dtype}")

    print("  -> ShapeProp runs the graph with fake tensors to infer shapes")
    print()


# ============ 2. Split module ============
def exp_split():
    print("=" * 60)
    print("2. Split module: partition graph")
    print("=" * 60)

    class BigModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3)
            self.fc1 = nn.Linear(16 * 6 * 6, 32)
            self.fc2 = nn.Linear(32, 10)

        def forward(self, x):
            x = self.conv(x).relu()
            x = x.flatten(1)
            x = self.fc1(x).relu()
            x = self.fc2(x)
            return x

    model = BigModel()
    gm = fx.symbolic_trace(model)

    # Split at fc1: everything before fc1 -> device 0, fc1+ -> device 1
    split_node = None
    for n in gm.graph.nodes:
        if n.op == "call_module" and "fc1" in n.target:
            split_node = n
            break

    if split_node:
        from torch.fx.passes.split_module import split_module

        # split_module needs a 'qualname_map' dict
        try:
            submodules = split_module(gm, None, lambda node: 0)
            print(f"  Split produced: {type(submodules).__name__}")
        except Exception as e:
            print(f"  split_module: {type(e).__name__}: {str(e)[:80]}")
    print()


# ============ 3. Custom PassBase ============
def exp_custom_pass():
    print("=" * 60)
    print("3. Custom PassBase: dropout removal pass")
    print("=" * 60)

    from torch.fx.passes.infra.pass_base import PassBase, PassResult

    class RemoveDropoutPass(PassBase):
        """Replace all dropout calls with identity in eval mode."""

        def call(self, gm):
            modified = False
            for node in list(gm.graph.nodes):
                if node.op == "call_function" and node.target in (
                    torch.nn.functional.dropout,
                    torch.dropout,
                ):
                    # Replace dropout with identity
                    node.replace_all_uses_with(node.args[0])
                    gm.graph.erase_node(node)
                    modified = True
            gm.recompile()
            return PassResult(gm, modified)

    class ModelWithDropout(nn.Module):
        def forward(self, x):
            return torch.nn.functional.dropout(x, p=0.5, training=False)

    model = ModelWithDropout().eval()
    gm = fx.symbolic_trace(model)

    before = len(gm.graph.nodes)
    print(f"  Before pass: {before} nodes")
    for n in gm.graph.nodes:
        print(f"    {n.op:>15s}: {str(n.target)[:40]}")

    # Apply pass
    result = RemoveDropoutPass()(gm)
    gm = result.graph_module

    print(f"  After pass:  {len(gm.graph.nodes)} nodes (modified={result.modified})")
    for n in gm.graph.nodes:
        print(f"    {n.op:>15s}: {str(n.target)[:40]}")

    # Verify: output unchanged (dropout in eval is identity)
    x = torch.randn(10)
    with torch.no_grad():
        y_before = model(x)
        y_after = gm(x)
    print(f"  Output match: {torch.allclose(y_before, y_after)}")
    print()


# ============ 4. PassManager ============
def exp_pass_manager():
    print("=" * 60)
    print("4. PassManager: orchestrate multiple passes")
    print("=" * 60)

    from torch.fx.passes.infra.pass_manager import PassManager
    from torch.fx.passes.infra.pass_base import PassBase, PassResult
    from torch.fx.passes.dialect.common.cse_pass import CSEPass

    class ReplaceAddWithMulPass(PassBase):
        def call(self, gm):
            modified = False
            for node in gm.graph.nodes:
                if node.op == "call_function" and node.target == torch.add:
                    new_node = gm.graph.call_function(
                        torch.mul, args=node.args, kwargs=node.kwargs
                    )
                    node.replace_all_uses_with(new_node)
                    gm.graph.erase_node(node)
                    modified = True
            gm.recompile()
            return PassResult(gm, modified)

    class Model(nn.Module):
        def forward(self, x):
            a = x + 1
            b = x + 1  # duplicate, CSE will catch
            c = a + b
            return c

    model = Model()
    gm = fx.symbolic_trace(model)

    # Run multiple passes: CSE first, then DCE
    pm = PassManager(
        passes=[CSEPass()],
        steps=1,
        run_checks_after_each_pass=True,
    )
    try:
        result = pm(gm)
        print(f"  PassManager result: modified={result.modified}")
        print(f"  Nodes: {len(result.graph_module.graph.nodes)}")
    except Exception as e:
        print(f"  PassManager: {type(e).__name__} — (try import cse pass?)")

    print()


EXPERIMENTS = {
    "shape_prop": exp_shape_prop,
    "split": exp_split,
    "custom_pass": exp_custom_pass,
    "pass_manager": exp_pass_manager,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[graph_passes test2] DONE")


if __name__ == "__main__":
    main()
