"""Meta Kernel & FakeTensor case study 5: SymInt with compile + inductor.

Companion script for meta_fake_tensor/meta_fake_tensor.md. Covers:
  1. SymInt guard behavior with torch.compile
  2. Dynamic=True compile behavior
  3. Inductor lowering with SymInt

Run:
    python 05_symint_inductor.py
"""

import sys

import torch


def exp_dynamic_compile_shapes():
    print("=" * 60)
    print("1. Dynamic shapes across multiple compile calls")
    print("=" * 60)

    def f(x):
        return x.relu().sum()

    compiled = torch.compile(f, dynamic=True)

    # Call with different batch sizes
    batch_sizes = [4, 8, 16, 32, 64]
    for bs in batch_sizes:
        x = torch.randn(bs, 128)
        y = compiled(x)
        print(f"  batch={bs:3d}: out={y.item():.4f}")

    print(f"\n  With dynamic=True, compiled graph adapts to varying batch size")
    print(f"  SymInt allows the graph to be parametric in batch dim")
    print()


def exp_static_vs_dynamic_perf():
    print("=" * 60)
    print("2. Static vs dynamic recompilation count")
    print("=" * 60)

    compile_count = [0]

    def counting_backend(gm, inputs):
        compile_count[0] += 1
        return gm

    def simple_model(x):
        return (x @ torch.eye(128)).sum()

    # Static compile
    compiled_static = torch.compile(simple_model, backend=counting_backend)
    shapes_static = [(4, 128), (8, 128), (16, 128), (4, 128)]
    compile_count[0] = 0

    for shape in shapes_static:
        x = torch.randn(*shape)
        try:
            compiled_static(x)
        except Exception:
            pass

    print(f"  Static compile recompiles: {compile_count[0]} (shapes: {shapes_static})")

    # Dynamic compile
    compiled_dynamic = torch.compile(simple_model, backend=counting_backend, dynamic=True)
    compile_count[0] = 0

    for shape in shapes_static:
        x = torch.randn(*shape)
        try:
            compiled_dynamic(x)
        except Exception:
            pass

    print(f"  Dynamic compile recompiles: {compile_count[0]} (shapes: {shapes_static})")
    print()


def exp_symint_in_inductor():
    print("=" * 60)
    print("3. How Inductor uses SymInt for codegen")
    print("=" * 60)

    print(f"  Inductor codegen with SymInt:")
    print(f"")
    print(f"  Static shape [4, 128]: generates")
    print(f"    for (int i = 0; i < 4 * 128; i++) {{")
    print(f"        out[i] = relu(in[i]);")
    print(f"    }}")
    print(f"")
    print(f"  Dynamic shape [s0, s1]: generates")
    print(f"    for (int i = 0; i < s0 * s1; i++) {{")
    print(f"        out[i] = relu(in[i]);")
    print(f"    }}")
    print(f"    // With guard: s0 == input.size(0), s1 == input.size(1)")
    print(f"")

    # Demonstrate with explicit compile + inspection
    @torch.compile(dynamic=True)
    def f(x):
        return x.relu().sum()

    x = torch.randn(32, 64)
    result = f(x)
    print(f"  Result: {result.item():.4f}")

    # Another call with different shape
    x2 = torch.randn(16, 64)
    result2 = f(x2)
    print(f"  Different batch: {result2.item():.4f}")
    print()


EXPERIMENTS = {
    "dynamic": exp_dynamic_compile_shapes,
    "perf": exp_static_vs_dynamic_perf,
    "inductor": exp_symint_in_inductor,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[meta_fake_tensor case 5] DONE")


if __name__ == "__main__":
    main()
