"""Custom Ops case study 4: TORCH_LIBRARY_IMPL and backend registration.

Companion script for custom_ops/custom_ops.md. Covers:
  1. Override existing backend keys
  2. IMPL vs DEF distinction
  3. CompositeImplicitAutograd as fallback

Run:
    python 04_backend_override.py
"""

import sys

import torch


def exp_override_existing():
    print("=" * 60)
    print("1. Override existing op with custom implementation")
    print("=" * 60)

    # Register a new IMPL for an existing aten op
    # This replaces the kernel for CompositeImplicitAutograd key
    lib = torch.library.Library("override_demo", "DEF")
    lib.define("my_override_op(Tensor x) -> Tensor")

    @torch.library.impl("override_demo::my_override_op", "CompositeImplicitAutograd")
    def composite_fallback(x):
        return x * 100

    @torch.library.impl("override_demo::my_override_op", "CPU")
    def cpu_override(x):
        return x * 200

    # CPU tensor: CPU key wins -> * 200
    x_cpu = torch.randn(3)
    y = torch.ops.override_demo.my_override_op(x_cpu)
    print(f"  CPU tensor result: {y.tolist()} (expect x*200)")

    # Meta tensor: no Meta key -> CompositeImplicitAutograd fallback
    x_meta = torch.empty(3, device="meta")
    y_meta = torch.ops.override_demo.my_override_op(x_meta)
    print(f"  Meta tensor result shape: {list(y_meta.shape)}")

    # Key insight: IMPL adds to existing op, DEF creates new
    print(f"\n  DEF = define new op")
    print(f"  IMPL = add kernel for existing op at a dispatch key")
    print()


def exp_backend_selection():
    print("=" * 60)
    print("2. Dispatch key selection priority order")
    print("=" * 60)

    lib = torch.library.Library("priority_demo", "DEF")
    lib.define("prio_test(Tensor x) -> Tensor")

    @torch.library.impl("priority_demo::prio_test", "CompositeImplicitAutograd")
    def lowest_prio(x):
        return x * 1

    @torch.library.impl("priority_demo::prio_test", "CPU")
    def cpu_prio(x):
        return x * 10

    # CPU > CompositeImplicitAutograd
    x = torch.randn(3)
    y = torch.ops.priority_demo.prio_test(x)
    print(f"  CPU result (expect x*10): {y.tolist()}")

    # Priority order (highest to lowest):
    priorities = [
        "Autograd (if requires_grad)",
        "AutocastCUDA (if autocast enabled)",
        "CPU/CUDA/Meta/XLA (backend-specific)",
        "CompositeExplicitAutograd",
        "CompositeImplicitAutograd (fallback)",
    ]
    print(f"\n  Dispatch key priority (high -> low):")
    for i, p in enumerate(priorities):
        print(f"    {i+1}. {p}")
    print()


def exp_multi_backend_op():
    print("=" * 60)
    print("3. Register op for multiple backends in Python")
    print("=" * 60)

    lib = torch.library.Library("multi_demo", "DEF")
    lib.define("multi_op(Tensor x) -> Tensor")

    @torch.library.impl("multi_demo::multi_op", "CPU")
    def cpu_fn(x):
        return x + 1

    @torch.library.impl("multi_demo::multi_op", "Meta")
    def meta_fn(x):
        return x.new_empty(x.shape)

    if torch.cuda.is_available():
        @torch.library.impl("multi_demo::multi_op", "CUDA")
        def cuda_fn(x):
            return x + 2

    # Test
    x_cpu = torch.randn(3)
    print(f"  CPU: {torch.ops.multi_demo.multi_op(x_cpu).tolist()}")

    x_meta = torch.empty(3, device="meta")
    print(f"  Meta: shape={list(torch.ops.multi_demo.multi_op(x_meta).shape)}")

    if torch.cuda.is_available():
        x_cuda = torch.randn(3, device="cuda")
        print(f"  CUDA: {torch.ops.multi_demo.multi_op(x_cuda).tolist()}")

    # Full compile support
    @torch.compile
    def compiled_fn(x):
        return torch.ops.multi_demo.multi_op(x).sum()

    result = compiled_fn(x_cpu)
    print(f"  Compile OK: {result:.4f}")
    print()


EXPERIMENTS = {
    "override": exp_override_existing,
    "selection": exp_backend_selection,
    "multi": exp_multi_backend_op,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[custom_ops case 4] DONE")


if __name__ == "__main__":
    main()
