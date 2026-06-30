"""TorchExport advanced: constraints, unflatten, programmatic validation.

Companion script for deploy/deploy.md.
  1. cross-dim constraints:  Constraint linking two dynamic dims
  2. unflatten:             restore original module hierarchy
  3. programmatic check:    validate graph nodes + inputs
  4. multiple inputs:       export with multiple dynamic inputs

Run:
    python test2.py                # full demo
    python test2.py constraint     # cross-dim constraints
    python test2.py unflatten      # unflatten ExportedProgram
    python test2.py validate       # programmatic validation
    python test2.py multi_input    # multiple dynamic inputs
"""

import sys
import torch
import torch.nn as nn
from torch.export import export, Dim, Constraint
from torch.export.dynamic_shapes import ShapesCollection


# ============ 1. Cross-dim constraints ============
def exp_constraint():
    print("=" * 60)
    print("1. Cross-dim constraints: linking two dynamic dimensions")
    print("=" * 60)

    class TwoInputModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 4)

        def forward(self, a, b):
            return self.fc(a + b)

    model = TwoInputModel().eval()
    batch = Dim("batch", max=64)

    # Without constraint: a and b can have different batch dims
    a = torch.randn(4, 8)
    b = torch.randn(4, 8)
    ep_no_constraint = export(
        model,
        (a, b),
        dynamic_shapes=(
            {"a": {0: Dim("batch_a", max=64)}, "b": {0: Dim("batch_b", max=64)}}
        ),
    )

    # With constraint: a.batch == b.batch
    ep = export(
        model,
        (a, b),
        dynamic_shapes=ShapesCollection(
            a=(batch, 8),
            b=(batch, 8),
        ),
    )

    # Run within constraint: same batch
    y = ep.module()(torch.randn(8, 8), torch.randn(8, 8))
    print(f"  batch=8+8 within constraint: OK, shape={list(y.shape)}")

    # Try violating constraint: different batch sizes
    try:
        ep.module()(torch.randn(4, 8), torch.randn(8, 8))
    except RuntimeError as e:
        lines = str(e).split("\n")
        print(f"  batch mismatch (4,8):")
        for line in lines[:2]:
            print(f"    {line[:100]}")

    print("  -> Constraints enforce a.shape[0] == b.shape[0]")
    print()


# ============ 2. Unflatten ============
def exp_unflatten():
    print("=" * 60)
    print("2. Unflatten: restore original module hierarchy")
    print("=" * 60)

    class NestedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3)
            self.block = nn.Sequential(nn.Linear(8 * 6 * 6, 32), nn.ReLU())
            self.head = nn.Linear(32, 10)

        def forward(self, x):
            x = self.conv(x).flatten(1)
            x = self.block(x)
            return self.head(x)

    model = NestedModel().eval()
    x = torch.randn(2, 3, 8, 8)
    ep = export(model, (x,))

    # ExportedProgram has flattened graph — unflatten restores hierarchy
    from torch.export import unflatten

    unflattened = unflatten(ep)

    print(f"  ExportedProgram type: {type(ep).__name__}")
    print(f"  Unflattened type:     {type(unflattened).__name__}")
    print(f"  Original submodules:  {list(model.named_children())}")
    print(f"  Unflattened children: {list(unflattened.named_children())}")
    print("  -> unflatten() restores module call hierarchy from flat graph")
    print()


# ============ 3. Programmatic validation ============
def exp_validate():
    print("=" * 60)
    print("3. Programmatic graph validation")
    print("=" * 60)

    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2)).eval()
    x = torch.randn(2, 4)
    ep = export(model, (x,))

    # Check all input/output specs
    sig = ep.graph_signature
    user_inputs = [s for s in sig.input_specs if s.kind.name == "USER_INPUT"]
    params = [s for s in sig.input_specs if s.kind.name == "PARAMETER"]
    buffers = [s for s in sig.input_specs if s.kind.name == "BUFFER"]
    outputs = sig.output_specs

    print(f"  User inputs:  {len(user_inputs)}")
    print(f"  Parameters:   {len(params)}")
    for p in params:
        print(f"    {p.arg.name}: {p.target}")
    print(f"  Buffers:      {len(buffers)}")
    print(f"  Outputs:      {len(outputs)}")

    # Check all ops are pure ATen (decomposed)
    all_aten = all(
        n.op != "call_module" or "nn." not in str(n.target)
        for n in ep.graph.nodes
        if n.op == "call_module"
    )
    print(f"\n  All ops decomposed to ATen: {all_aten}")

    # Count op types
    from collections import Counter

    op_counts = Counter()
    for n in ep.graph.nodes:
        if n.op == "call_function":
            target_str = (
                str(n.target).split(".")[-1]
                if hasattr(n.target, "__module__")
                else str(n.target)
            )
            op_counts[target_str.split("::")[-1]] += 1

    print(f"  Top ops:")
    for op, cnt in op_counts.most_common(8):
        print(f"    {op:>25s}: {cnt}")
    print()


# ============ 4. Multiple dynamic inputs ============
def exp_multi_input():
    print("=" * 60)
    print("4. Multiple dynamic inputs + constraints")
    print("=" * 60)

    class MultiInputModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 4)

        def forward(self, x, y, scale=1.0):
            return self.fc(x * y * scale)

    model = MultiInputModel().eval()

    batch_x = Dim("batch_x", min=1, max=32)
    batch_y = Dim("batch_y", min=1, max=32)

    x = torch.randn(4, 8)
    y = torch.randn(4, 8)

    ep = export(
        model,
        (x, y, 2.0),
        dynamic_shapes=ShapesCollection(
            x=(batch_x, 8),
            y=(batch_y, 8),
            constraints=[Constraint(x_batch=batch_x, y_batch=batch_y)],
        ),
    )

    # Run with different sizes within constraint
    for size in [2, 6, 8]:
        out = ep.module()(torch.randn(size, 8), torch.randn(size, 8), 1.5)
        print(f"  batch={size}: output shape={list(out.shape)}")

    # Check graph signature shows 3 user inputs (x, y, scale)
    user_inputs = [
        s for s in ep.graph_signature.input_specs if s.kind.name == "USER_INPUT"
    ]
    print(f"\n  User inputs in graph: {len(user_inputs)}")
    for inp in user_inputs:
        print(f"    {inp.arg.name}")

    print()


EXPERIMENTS = {
    "constraint": exp_constraint,
    "unflatten": exp_unflatten,
    "validate": exp_validate,
    "multi_input": exp_multi_input,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[deploy test2] DONE")


if __name__ == "__main__":
    main()
