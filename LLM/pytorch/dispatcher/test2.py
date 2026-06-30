"""Dispatcher advanced: multiple backends, composite ops, autograd dispatch.

Companion script for dispatcher/dispatcher.md.
  1. multi-backend registration:  CPU + CUDA + Meta kernels
  2. composite op:                register a decomposition
  3. autograd custom:             register Autograd dispatch key
  4. dispatch key override:       use Meta tensor to test without hardware

Run:
    python test2.py               # full demo
    python test2.py multi          # multi-backend registration
    python test2.py composite      # composite/decomposition
    python test2.py autograd       # custom Autograd dispatch
    python test2.py meta           # Meta tensor dispatch (dry run)
"""

import sys
import torch


# ============ 1. Multi-backend registration ============
def exp_multi():
    print("=" * 60)
    print("1. Multi-backend: CPU + CUDA + Meta kernels")
    print("=" * 60)

    lib = torch.library.Library("demo2", "DEF")
    lib.define("square(Tensor a) -> Tensor")

    # CPU kernel
    @torch.library.impl("demo2::square", "CPU")
    def square_cpu(x):
        return x * x

    # CUDA kernel
    @torch.library.impl("demo2::square", "CUDA")
    def square_cuda(x):
        return x * x

    # Meta kernel (for shape inference without hardware)
    @torch.library.impl("demo2::square", "Meta")
    def square_meta(x):
        return torch.empty_like(x)

    # Test CPU
    cpu_x = torch.tensor([1.0, 2.0, 3.0])
    cpu_y = torch.ops.demo2.square(cpu_x)
    print(
        f"  CPU: {cpu_x.tolist()} -> {cpu_y.tolist()}, match={torch.allclose(cpu_y, cpu_x**2)}"
    )

    # Test CUDA
    if torch.cuda.is_available():
        cuda_x = torch.tensor([2.0, 3.0, 4.0], device="cuda")
        cuda_y = torch.ops.demo2.square(cuda_x)
        print(
            f"  CUDA: {cuda_x.tolist()} -> {cuda_y.tolist()}, match={torch.allclose(cuda_y, cuda_x**2)}"
        )

    # Test Meta (dry run, no memory allocation)
    meta_x = torch.empty(2, 4, device="meta")
    meta_y = torch.ops.demo2.square(meta_x)
    print(f"  Meta: shape={list(meta_x.shape)} -> {list(meta_y.shape)}")
    print("  -> Meta dispatch key allows shape inference with zero memory")
    print()


# ============ 2. Composite op ============
def exp_composite():
    print("=" * 60)
    print("2. CompositeImplicitAutograd: decomposition")
    print("=" * 60)

    lib2 = torch.library.Library("demo2", "DEF")
    lib2.define("myrelu(Tensor a) -> Tensor")

    # Register as composite: PyTorch automatically decomposes it
    @torch.library.impl("demo2::myrelu", "CompositeImplicitAutograd")
    def myrelu_impl(x):
        return torch.maximum(x, torch.tensor(0.0, dtype=x.dtype, device=x.device))

    # Works on CPU, CUDA, Meta — all via decomposition
    x = torch.tensor([-1.0, 0.0, 2.0])
    y = torch.ops.demo2.myrelu(x)
    print(f"  Composite: {x.tolist()} -> {y.tolist()}")
    print(f"  Match torch.relu: {torch.allclose(y, x.relu())}")

    # Also works on CUDA without a CUDA kernel!
    if torch.cuda.is_available():
        xc = torch.tensor([-1.0, 2.0, -3.0], device="cuda")
        yc = torch.ops.demo2.myrelu(xc)
        print(f"  CUDA (via composite): {xc.tolist()} -> {yc.tolist()}")

    # Meta works too
    mx = torch.empty(4, 4, device="meta")
    my = torch.ops.demo2.myrelu(mx)
    print(f"  Meta: shape={list(mx.shape)} -> {list(my.shape)}")
    print("  -> CompositeImplicitAutograd provides fallback for ALL backends")
    print()


# ============ 3. Custom Autograd dispatch ============
def exp_autograd():
    print("=" * 60)
    print("3. Autograd dispatch: custom backward for registered op")
    print("=" * 60)

    lib3 = torch.library.Library("demo3", "DEF")
    lib3.define("my_scale(Tensor a, float scale) -> Tensor")

    @torch.library.impl("demo3::my_scale", "CompositeImplicitAutograd")
    def my_scale_impl(x, scale):
        return x * scale

    # Register Autograd kernel for custom backward
    @torch.library.impl("demo3::my_scale", "Autograd")
    def my_scale_autograd(x, scale):
        return MyScaleAutograd.apply(x, scale)

    class MyScaleAutograd(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, scale):
            ctx.scale = scale
            return x * scale

        @staticmethod
        def backward(ctx, grad_output):
            # Demonstrate custom gradient: scale the gradient too
            return grad_output * ctx.scale, None  # No grad for scale

    # Test with autograd
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = torch.ops.demo3.my_scale(x, 2.0)
    y.sum().backward()

    print(f"  x: {x.tolist()}")
    print(f"  y = my_scale(x, 2.0): {y.tolist()}")
    print(f"  x.grad: {x.grad.tolist()}")
    print(f"  expected grad: [2.0, 2.0, 2.0]")
    print(f"  match: {torch.allclose(x.grad, torch.tensor([2.0, 2.0, 2.0]))}")
    print()


# ============ 4. Meta tensor dispatch (dry run) ============
def exp_meta():
    print("=" * 60)
    print("4. Meta tensor: shape inference without hardware")
    print("=" * 60)

    # Create model entirely on meta device
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10),
    ).to("meta")

    # Forward: no actual computation, just shape propagation
    x = torch.empty(4, 64, device="meta")
    with torch.no_grad():
        y = model(x)

    print(f"  Input shape:  {list(x.shape)}")
    print(f"  Output shape: {list(y.shape)}")

    # Parameter info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}")
    print(f"  FP16 memory:  {total_params * 2 / 1e6:.1f} MB")
    print(f"  FP32 memory:  {total_params * 4 / 1e6:.1f} MB")
    print("  -> Meta dispatch allows architecture analysis with zero memory")
    print()


EXPERIMENTS = {
    "multi": exp_multi,
    "composite": exp_composite,
    "autograd": exp_autograd,
    "meta": exp_meta,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[dispatcher test2] DONE")


if __name__ == "__main__":
    main()
