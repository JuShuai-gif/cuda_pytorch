"""Meta Kernel & FakeTensor case study 4: custom op meta kernel registration.

Companion script for meta_fake_tensor/meta_fake_tensor.md. Covers:
  1. Register meta kernel for custom op
  2. Test with torch.compile
  3. Complex shape inference for custom ops

Run:
    python 04_custom_op_meta.py
"""

import sys

import torch
from torch._subclasses.fake_tensor import FakeTensorMode


def exp_minimal_meta_kernel():
    print("=" * 60)
    print("1. Minimal meta kernel: just return empty tensor")
    print("=" * 60)

    lib = torch.library.Library("meta_demo", "DEF")
    lib.define("my_scale(Tensor x, float scale) -> Tensor")

    @torch.library.impl("meta_demo::my_scale", "CPU")
    def my_scale_cpu(x, scale):
        return x * scale

    # Meta kernel: return empty tensor of correct shape
    @torch.library.impl("meta_demo::my_scale", "Meta")
    def my_scale_meta(x, scale):
        return x.new_empty(x.shape)

    # Verify meta kernel works
    with FakeTensorMode():
        x = torch.randn(4, 8, device="cuda")
        y = torch.ops.meta_demo.my_scale(x, 2.0)
        print(f"  my_scale(x, 2.0) on FakeTensor:")
        print(f"    type={type(y).__name__}, shape={list(y.shape)}, device={y.device}")

    # Compile test
    @torch.compile
    def f(x):
        return torch.ops.meta_demo.my_scale(x, 3.0).sum()

    x = torch.randn(4, 8)
    result = f(x)
    print(f"  Compile OK: {result:.4f}")
    print()


def exp_shape_changing_op():
    print("=" * 60)
    print("2. Meta kernel for shape-changing custom op")
    print("=" * 60)

    lib = torch.library.Library("meta_demo2", "DEF")
    lib.define("my_matmul(Tensor a, Tensor b) -> Tensor")

    @torch.library.impl("meta_demo2::my_matmul", "CPU")
    def my_matmul_cpu(a, b):
        return a @ b

    @torch.library.impl("meta_demo2::my_matmul", "Meta")
    def my_matmul_meta(a, b):
        # Infer output shape: [M, K] @ [K, N] -> [M, N]
        return a.new_empty(a.size(0), b.size(1))

    # Test with different shapes
    shapes = [
        ((4, 8), (8, 3)),
        ((16, 32), (32, 64)),
        ((1, 5), (5, 10)),
    ]

    with FakeTensorMode():
        for sa, sb in shapes:
            a = torch.randn(*sa, device="cuda")
            b = torch.randn(*sb, device="cuda")
            c = torch.ops.meta_demo2.my_matmul(a, b)
            expected_shape = [sa[0], sb[1]]
            print(f"  {list(sa)} @ {list(sb)} -> {list(c.shape)} (expected {expected_shape})")
    print()


def exp_missing_meta_debug():
    print("=" * 60)
    print("3. Debug missing meta kernel systematically")
    print("=" * 60)

    lib = torch.library.Library("meta_debug", "DEF")
    lib.define("op1(Tensor x) -> Tensor")
    lib.define("op2(Tensor x) -> Tensor")
    lib.define("op3(Tensor x) -> Tensor")

    @torch.library.impl("meta_debug::op1", "CPU")
    def op1_cpu(x):
        return x * 2

    # op2: register meta
    @torch.library.impl("meta_debug::op2", "CPU")
    def op2_cpu(x):
        return x + 1

    @torch.library.impl("meta_debug::op2", "Meta")
    def op2_meta(x):
        return x.new_empty(x.shape)

    # op3: no meta
    @torch.library.impl("meta_debug::op3", "CPU")
    def op3_cpu(x):
        return x.relu()

    ops = ["meta_debug::op1", "meta_debug::op2", "meta_debug::op3"]
    for op_name in ops:
        has_meta = torch._C._dispatch_has_kernel_for_dispatch_key(op_name, "Meta")
        status = "OK" if has_meta else "MISSING"
        print(f"  {op_name:20s}: Meta kernel = {status}")
    print()


EXPERIMENTS = {
    "minimal": exp_minimal_meta_kernel,
    "shape": exp_shape_changing_op,
    "debug": exp_missing_meta_debug,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[meta_fake_tensor case 4] DONE")


if __name__ == "__main__":
    main()
