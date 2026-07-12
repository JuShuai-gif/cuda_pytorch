"""torchgen case study 2: dispatch key routing and Composite* differences.

Companion script for torchgen/torchgen.md. Covers:
  1. CompositeImplicitAutograd vs CompositeExplicitAutograd behavior
  2. Custom op dispatch key routing
  3. dispatch key priority and fallback

Run:
    python 02_dispatch_key_routing.py
"""

import sys

import torch


def exp_composite_difference():
    print("=" * 60)
    print("1. CompositeImplicitAutograd vs CompositeExplicitAutograd")
    print("=" * 60)

    # add.Tensor uses CompositeImplicitAutograd
    # This means: no explicit autograd backward formula needed
    # The autograd is implicit from composing primitives

    x = torch.randn(3, requires_grad=True)
    y = torch.randn(3, requires_grad=True)

    # Both add and mul can trace through autograd
    z_add = x + y
    z_mul = x * y

    print(f"  add (CompositeImplicitAutograd): grad_fn={z_add.grad_fn}")
    print(f"  mul (CompositeExplicitAutograd): grad_fn={z_mul.grad_fn}")

    # Verify gradient works for both
    z_add.sum().backward(retain_graph=True)
    z_mul.sum().backward()
    print(f"  add backward: x.grad={x.grad}")
    print()

    # Key insight: CompositeImplicitAutograd ops don't need derivatives.yaml entries
    print(f"  CompositeImplicitAutograd: no derivatives.yaml entry needed")
    print(f"  Backward is composed from primitive op gradients")
    print()


def exp_custom_op_routing():
    print("=" * 60)
    print("2. Custom op dispatch key routing")
    print("=" * 60)

    lib = torch.library.Library("demo", "DEF")
    lib.define("my_scale(Tensor x, float alpha) -> Tensor")

    @torch.library.impl("demo::my_scale", "CPU")
    def my_scale_cpu(x, alpha):
        return x * alpha

    @torch.library.impl("demo::my_scale", "CompositeImplicitAutograd")
    def my_scale_composite(x, alpha):
        return x * alpha

    x = torch.randn(3)
    y = torch.ops.demo.my_scale(x, 2.0)
    print(f"  CPU tensor: my_scale(x, 2.0) result OK, shape={list(y.shape)}")

    # Check dispatch table
    table = torch._C._dispatch_dump_table("demo::my_scale")
    print(f"  Dispatch table:")
    for line in table.strip().split("\n"):
        if line.strip():
            print(f"    {line.strip()}")

    # CompositeImplicitAutograd provides fallback for all backends
    if torch.cuda.is_available():
        x_cuda = torch.randn(3, device="cuda")
        y_cuda = torch.ops.demo.my_scale(x_cuda, 3.0)
        print(f"\n  CUDA tensor: works via CompositeImplicitAutograd fallback")
        print(f"  Result device: {y_cuda.device}, shape={list(y_cuda.shape)}")
    print()


def exp_dispatch_priority():
    print("=" * 60)
    print("3. Dispatch key priority and override")
    print("=" * 60)

    lib = torch.library.Library("demo2", "DEF")
    lib.define("priority_test(Tensor x) -> Tensor")

    @torch.library.impl("demo2::priority_test", "CompositeImplicitAutograd")
    def fallback_impl(x):
        print(f"    -> CompositeImplicitAutograd fallback called")
        return x * 10

    @torch.library.impl("demo2::priority_test", "CPU")
    def cpu_impl(x):
        print(f"    -> CPU-specific kernel called")
        return x * 20

    x = torch.randn(3)
    print(f"  Calling demo2::priority_test on CPU tensor:")
    y = torch.ops.demo2.priority_test(x)
    print(f"  Result: {y}")
    # CPU key has higher priority than CompositeImplicitAutograd
    print(f"  -> CPU kernel wins (higher priority dispatch key)")

    if torch.cuda.is_available():
        x_cuda = torch.randn(3, device="cuda")
        print(f"\n  Calling on CUDA tensor (no CUDA kernel registered):")
        y_cuda = torch.ops.demo2.priority_test(x_cuda)
        print(f"  Result device: {y_cuda.device}")
        print(f"  -> CompositeImplicitAutograd fallback (no CUDA kernel)")
    print()


def exp_tls_exclude():
    print("=" * 60)
    print("4. TLS exclude: torch.no_grad's effect on dispatch")
    print("=" * 60)

    x = torch.randn(3, requires_grad=True)
    y = torch.randn(3, requires_grad=True)

    # no_grad excludes Autograd key
    with torch.no_grad():
        z = x + y
        print(f"  Inside no_grad: requires_grad={z.requires_grad}, grad_fn={z.grad_fn}")

    z = x + y
    print(f"  Outside no_grad: requires_grad={z.requires_grad}, grad_fn={z.grad_fn}")

    # This is TLS (Thread Local State) controlling dispatch keys
    print(f"\n  torch.no_grad excludes Autograd from DispatchKeySet")
    print(f"  torch.autocast includes AutocastCUDA in DispatchKeySet")
    print()


EXPERIMENTS = {
    "composite": exp_composite_difference,
    "custom": exp_custom_op_routing,
    "priority": exp_dispatch_priority,
    "tls": exp_tls_exclude,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[torchgen case 2] DONE")


if __name__ == "__main__":
    main()
