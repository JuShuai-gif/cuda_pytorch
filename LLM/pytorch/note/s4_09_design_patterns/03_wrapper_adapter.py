"""Design Patterns case study 3: wrapper/decorator pattern in PyTorch.

Companion script for 40_design_patterns/design_patterns.md.

Run:
    python 03_wrapper_adapter.py
"""

import sys
import torch


def exp_dispatch_wrapper():
    print("=" * 60)
    print("1. Wrapper: BatchedTensor wraps Tensor for vmap")
    print("=" * 60)
    from torch.func import vmap, grad

    def loss(w, x):
        return ((w * x).sum()).sin()

    w = torch.randn(4, requires_grad=True)
    xs = torch.randn(8, 4)

    # vmap wraps each input in BatchedTensor internally
    per_grad = vmap(grad(loss), in_dims=(None, 0))(w, xs)
    print(f"  vmap wraps tensors in BatchedTensor internally")
    print(f"  Per-sample grad shape: {list(per_grad.shape)} (8 samples, 4 params)")
    print(f"  Pattern: BatchedTensor = Wrapper(Tensor, batch_dim, batch_size)")


def exp_tensor_subclass():
    print("=" * 60)
    print("2. Decorator: Tensor subclass intercepts ops")
    print("=" * 60)

    class ScaleTensor(torch.Tensor):
        """Wrapper that scales all values by factor."""
        @staticmethod
        def __new__(cls, data, factor=2.0):
            t = torch.as_tensor(data).as_subclass(cls)
            t._factor = factor
            return t

        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            if kwargs is None:
                kwargs = {}
            args_flat = []
            for a in args:
                if isinstance(a, cls):
                    args_flat.append(a.as_subclass(torch.Tensor))
                else:
                    args_flat.append(a)
            result = func(*args_flat, **kwargs)
            return result

    x = ScaleTensor(torch.arange(6).view(2, 3).float(), factor=3.0)
    y = x * 2 + 1
    print(f"  ScaleTensor: {x}")
    print(f"  Computed: {y}  (dispatch intercepted by __torch_dispatch__)")


def exp_kernel_registry():
    print("=" * 60)
    print("3. Registry: TORCH_LIBRARY = self-registering singleton")
    print("=" * 60)
    lib = torch.library.Library("reg_demo", "DEF")
    lib.define("my_op(Tensor x) -> Tensor")

    @torch.library.impl("reg_demo::my_op", "CPU")
    def cpu_impl(x):
        return x * 10

    table = torch._C._dispatch_dump_table("reg_demo::my_op")
    print(f"  Registered to global dispatch table automatically")
    print(f"  -> Registry pattern: each TORCH_LIBRARY call auto-registers")


EXPERIMENTS = {"wrapper": exp_dispatch_wrapper, "decorator": exp_tensor_subclass, "registry": exp_kernel_registry}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}'")
            continue
        EXPERIMENTS[name]()
    print("[design_patterns case 3] DONE")


if __name__ == "__main__":
    main()
