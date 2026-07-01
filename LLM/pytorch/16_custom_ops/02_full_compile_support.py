"""Custom Ops case study 2: full torch.compile support (meta + autograd).

Companion script for custom_ops/custom_ops.md. Covers:
  1. Register Meta kernel for compile support
  2. Register Autograd for backward
  3. Verify compile compatibility

Run:
    python 02_full_compile_support.py
"""

import sys

import torch


def exp_meta_kernel():
    print("=" * 60)
    print("1. Register Meta kernel for torch.compile")
    print("=" * 60)

    lib = torch.library.Library("compiledemo", "DEF")
    lib.define("my_silu(Tensor x) -> Tensor")

    @torch.library.impl("compiledemo::my_silu", "CPU")
    def my_silu_cpu(x):
        return x * torch.sigmoid(x)

    if torch.cuda.is_available():
        @torch.library.impl("compiledemo::my_silu", "CUDA")
        def my_silu_cuda(x):
            return x * torch.sigmoid(x)

    # Test 1: Without Meta kernel -> compile fails
    print(f"  Test 1: Without Meta kernel")

    @torch.compile
    def f_no_meta(x):
        return torch.ops.compiledemo.my_silu(x).sum()

    x = torch.randn(4, 8)
    try:
        result = f_no_meta(x)
        print(f"    Compile OK: {result}")
    except Exception as e:
        print(f"    Compile FAILED: {str(e)[:80]}")

    # Test 2: Register Meta kernel -> compile succeeds
    print(f"\n  Test 2: With Meta kernel registered")
    try:
        @torch.library.impl("compiledemo::my_silu", "Meta")
        def my_silu_meta(x):
            return x.new_empty(x.shape)

        @torch.compile
        def f_with_meta(x):
            return torch.ops.compiledemo.my_silu(x).sum()

        result = f_with_meta(x)
        print(f"    Compile OK: {result:.4f}")
    except Exception as e:
        print(f"    Error: {str(e)[:80]}")
    print()


def exp_autograd_kernel():
    print("=" * 60)
    print("2. Register Autograd kernel for backward")
    print("=" * 60)

    lib = torch.library.Library("graddemo", "DEF")
    lib.define("my_op(Tensor x) -> Tensor")

    @torch.library.impl("graddemo::my_op", "CPU")
    def my_op_cpu(x):
        return x * torch.sigmoid(x)

    @torch.library.impl("graddemo::my_op", "Autograd")
    def my_op_autograd(x):
        class MyOpFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x * torch.sigmoid(x)

            @staticmethod
            def backward(ctx, grad_output):
                x, = ctx.saved_tensors
                sig_x = torch.sigmoid(x)
                return grad_output * sig_x * (1 + x * (1 - sig_x))

        return MyOpFn.apply(x)

    x = torch.randn(4, 8, requires_grad=True)
    y = torch.ops.graddemo.my_op(x)
    y.sum().backward()

    print(f"  Forward: y.sum() = {y.sum().item():.4f}")
    print(f"  Backward: x.grad norm = {x.grad.norm().item():.4f}")

    # Verify against analytical gradient
    def silu_grad(x):
        s = torch.sigmoid(x)
        return s * (1 + x * (1 - s))

    expected_grad = silu_grad(x)
    max_diff = (x.grad - expected_grad).abs().max().item()
    print(f"  Gradient max diff: {max_diff:.2e}")
    print()


def exp_compile_full_pipeline():
    print("=" * 60)
    print("3. Full pipeline: custom op with meta + autograd + compile")
    print("=" * 60)

    lib = torch.library.Library("fulldemo", "DEF")
    lib.define("my_full_op(Tensor x) -> Tensor")

    # CPU
    @torch.library.impl("fulldemo::my_full_op", "CPU")
    def my_full_op_cpu(x):
        return torch.relu(x) * torch.sigmoid(x)

    # CUDA
    if torch.cuda.is_available():
        @torch.library.impl("fulldemo::my_full_op", "CUDA")
        def my_full_op_cuda(x):
            return torch.relu(x) * torch.sigmoid(x)

    # Meta (for compile)
    @torch.library.impl("fulldemo::my_full_op", "Meta")
    def my_full_op_meta(x):
        return x.new_empty(x.shape)

    # Autograd
    @torch.library.impl("fulldemo::my_full_op", "Autograd")
    def my_full_op_autograd(x):
        class MyFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return torch.ops.fulldemo.my_full_op(x)

            @staticmethod
            def backward(ctx, grad_output):
                x, = ctx.saved_tensors
                relu_x = torch.relu(x)
                sig_x = torch.sigmoid(x)
                grad = (x > 0).float() * sig_x + relu_x * sig_x * (1 - sig_x)
                return grad_output * grad
        return MyFn.apply(x)

    # Test: Compile + Autograd
    @torch.compile
    def train_step(x):
        return torch.ops.fulldemo.my_full_op(x).sum()

    x = torch.randn(4, 8, requires_grad=True)
    loss = train_step(x)
    loss.backward()

    print(f"  Loss: {loss.item():.4f}")
    print(f"  x.grad norm: {x.grad.norm().item():.4f}")
    print(f"  Full pipeline successful!")
    print(f"  -> Custom op works with torch.compile + autograd")
    print()


EXPERIMENTS = {
    "meta": exp_meta_kernel,
    "autograd": exp_autograd_kernel,
    "full": exp_compile_full_pipeline,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[custom_ops case 2] DONE")


if __name__ == "__main__":
    main()
