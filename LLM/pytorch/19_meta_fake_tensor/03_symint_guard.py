"""Meta Kernel & FakeTensor case study 3: SymInt and dynamic shapes.

Companion script for meta_fake_tensor/meta_fake_tensor.md. Covers:
  1. SymInt guards: how torch.compile tracks dynamic shapes
  2. Recompilation triggers: when guards fail
  3. Dynamic vs static compilation comparison

Run:
    python 03_symint_guard.py
"""

import sys

import torch


def exp_dynamo_explain_guards():
    print("=" * 60)
    print("1. Inspect Dynamo guards for a compiled function")
    print("=" * 60)

    def f(x, y):
        return (x @ y.t()).sum()

    try:
        from torch._dynamo import explain

        explanation = explain(f, torch.randn(4, 8), torch.randn(3, 8))
        guard_count = len(explanation.guards)
        print(f"  Total guards generated: {guard_count}")
        print(f"  Sample guards (first 5):")
        for guard in explanation.guards[:5]:
            print(f"    {guard}")
        if guard_count > 5:
            print(f"    ... and {guard_count - 5} more")
    except Exception as e:
        print(f"  torch._dynamo.explain not available: {e}")

    print(f"\n  Guards encode shape/dtype/device assumptions")
    print(f"  When an assumption is violated -> recompile")
    print()


def exp_dynamic_shapes():
    print("=" * 60)
    print("2. Dynamic shapes: recompile behavior")
    print("=" * 60)

    def f(x):
        return x.relu().sum()

    # Static compile: shapes are baked in
    compiled_static = torch.compile(f, dynamic=False)
    compiled_dynamic = torch.compile(f, dynamic=True)

    shapes_to_test = [(4, 8), (8, 8), (16, 8), (32, 8)]

    print(f"  Static mode (dynamic=False):")
    for shape in shapes_to_test:
        x = torch.randn(*shape)
        try:
            y = compiled_static(x)
            print(f"    shape={list(shape)}: OK, out={y.item():.3f}")
        except Exception as e:
            print(f"    shape={list(shape)}: ERROR {str(e)[:60]}")

    print(f"\n  Dynamic mode (dynamic=True):")
    for shape in shapes_to_test:
        x = torch.randn(*shape)
        y = compiled_dynamic(x)
        print(f"    shape={list(shape)}: OK, out={y.item():.3f}")
    print()


def exp_symint_demo():
    print("=" * 60)
    print("3. SymInt: shape as symbolic expressions")
    print("=" * 60)

    try:
        import torch.fx.experimental.symbolic_shapes as sym_shapes

        print(f"  SymInt is used internally by torch.compile for:")
        print(f"    - s0:  first dynamic dimension of first arg")
        print(f"    - s1:  second dynamic dimension of first arg")
        print(f"    - Expressions like s0*s1 for flattened size")

        # Simple check if SymInt is active
        # Dynamo internally creates SymInts when dynamic=True
    except ImportError as e:
        print(f"  symbolic_shapes module: {e}")

    def f(x, y):
        # Relu is elementwise -> output shape = input shape (dynamic)
        return torch.relu(x + y).sum()

    compiled = torch.compile(f, dynamic=True)
    x = torch.randn(4, 8)
    y = torch.randn(4, 8)

    out = compiled(x, y)
    print(f"  f(x,y) with dynamic=True: {out.item():.4f}")

    # Different shape should work (dynamic)
    x2 = torch.randn(8, 8)
    y2 = torch.randn(8, 8)
    out2 = compiled(x2, y2)
    print(f"  Same compiled function, different shape: {out2.item():.4f}")
    print(f"\n  SymInt allows compiled graph to adapt to different input shapes")
    print()


def exp_guard_failure():
    print("=" * 60)
    print("4. Simulate guard failure and recompilation")
    print("=" * 60)

    compile_count = [0]

    # Custom backend that counts recompilations
    def counting_backend(gm, example_inputs):
        compile_count[0] += 1
        return gm

    def f(x):
        return x.relu().sum()

    compiled = torch.compile(f, backend=counting_backend)

    shapes = [(4, 8), (4, 8), (8, 8), (8, 8), (16, 8)]
    for i, shape in enumerate(shapes):
        x = torch.randn(*shape)
        y = compiled(x)
        print(f"  Call {i+1}: shape={list(shape)}, total compilations={compile_count[0]}")

    print(f"\n  Total compilations: {compile_count[0]}")
    print(f"  Shape changes cause recompilation when guard fails")
    print()


EXPERIMENTS = {
    "guards": exp_dynamo_explain_guards,
    "dynamic": exp_dynamic_shapes,
    "symint": exp_symint_demo,
    "recompile": exp_guard_failure,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[meta_fake_tensor case 3] DONE")


if __name__ == "__main__":
    main()
