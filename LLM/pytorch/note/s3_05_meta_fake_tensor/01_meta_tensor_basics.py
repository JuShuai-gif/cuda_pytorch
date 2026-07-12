"""Meta Kernel & FakeTensor case study 1: shape inference without execution.

Companion script for meta_fake_tensor/meta_fake_tensor.md. Covers:
  1. Meta tensor basics: shape inference on complex models
  2. Compare meta tensor vs real tensor memory
  3. Check which ops lack meta kernel

Run:
    python 01_meta_tensor_basics.py
    python 01_meta_tensor_basics.py model
    python 01_meta_tensor_basics.py check
"""

import sys

import torch


def exp_meta_basics():
    print("=" * 60)
    print("1. Meta tensor: shape inference without data")
    print("=" * 60)

    # Create meta tensors
    x = torch.empty(2, 3, device="meta")
    y = torch.empty(3, 4, device="meta")

    print(f"  x: device={x.device}, shape={list(x.shape)}, dtype={x.dtype}")
    print(f"  x data_ptr: {x.data_ptr()} (0 on meta)")

    # Operations on meta tensor
    z = x @ y
    print(f"\n  z = x @ y:")
    print(f"    device={z.device}, shape={list(z.shape)}, stride={z.stride()}")

    # Can't access data
    try:
        z.numpy()
    except Exception as e:
        print(f"\n  z.numpy() ERROR: {str(e)[:80]}")
        print(f"  -> meta tensor has no storage, cannot access data")

    # Chain operations
    w = z.relu().sin().sum(dim=1)
    print(f"\n  w = relu(sin(z)).sum(1):")
    print(f"    shape={list(w.shape)}, device={w.device}")
    print()


def exp_model_on_meta():
    print("=" * 60)
    print("2. Run a full model on meta device")
    print("=" * 60)

    class SampleTransformer(torch.nn.Module):
        def __init__(self, hidden=256, layers=4):
            super().__init__()
            self.layers = torch.nn.ModuleList(
                [
                    torch.nn.Sequential(
                        torch.nn.Linear(hidden, hidden),
                        torch.nn.ReLU(),
                        torch.nn.LayerNorm(hidden),
                    )
                    for _ in range(layers)
                ]
            )
            self.head = torch.nn.Linear(hidden, 10)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x) + x  # residual
            return self.head(x)

    model = SampleTransformer()

    # Run on meta to inspect shapes without allocating GPU memory
    x = torch.empty(8, 256, device="meta")
    with torch.no_grad():
        y = model(x)
        print(f"  Input:  shape={list(x.shape)}")
        print(f"  Output: shape={list(y.shape)}")
        print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # This is how torch.compile pre-checks model shapes
    print(f"\n  torch.compile uses this internally to trace shapes")
    print(f"  without running actual CUDA kernels")
    print()


def exp_meta_kernel_check():
    print("=" * 60)
    print("3. Check which ops have/don't have Meta kernel")
    print("=" * 60)

    ops_to_check = [
        "add",
        "matmul",
        "layer_norm",
        "conv2d",
        "batch_norm",
        "max_pool2d",
        "silu",
        "gelu",
    ]

    print(f"  {'Op':20s} {'Meta':8s} {'CPU':8s} {'CUDA':8s}")
    print(f"  {'-'*40}")
    for op in ops_to_check:
        meta = torch._C._dispatch_has_kernel_for_dispatch_key(op, "Meta")
        cpu = torch._C._dispatch_has_kernel_for_dispatch_key(op, "CPU")
        cuda = torch._C._dispatch_has_kernel_for_dispatch_key(op, "CUDA")
        print(f"  {op:20s} {'YES' if meta else 'no':8s} {'YES' if cpu else 'no':8s} {'YES' if cuda else 'no':8s}")

    print(f"\n  Ops without Meta kernel will fail torch.compile")

    # Custom op without meta kernel
    try:
        lib = torch.library.Library("demo", "DEF")
        lib.define("no_meta_op(Tensor x) -> Tensor")

        @torch.library.impl("demo::no_meta_op", "CPU")
        def no_meta_cpu(x):
            return x * 2

        # Try meta tensor -> fails
        x = torch.empty(3, device="meta")
        try:
            y = torch.ops.demo.no_meta_op(x)
            print(f"  demo::no_meta_op on meta: succeeded (has Meta key)")
        except Exception as e:
            print(f"  demo::no_meta_op on meta: FAILED ({str(e)[:60]})")
    except Exception as e:
        print(f"  Custom op setup error: {e}")
    print()


EXPERIMENTS = {
    "basics": exp_meta_basics,
    "model": exp_model_on_meta,
    "check": exp_meta_kernel_check,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[meta_fake_tensor case 1] DONE")


if __name__ == "__main__":
    main()
