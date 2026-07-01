"""Custom Ops case study 6: tensor subclass for custom dispatch.

Companion script for custom_ops/custom_ops.md. Covers:
  1. __torch_dispatch__ for tensor subclasses
  2. Custom tensor type with dispatch
  3. Integration with torch.compile

Run:
    python 06_tensor_subclass.py
"""

import sys

import torch


def exp_subclass_basics():
    print("=" * 60)
    print("1. Tensor subclass: override dispatch at tensor level")
    print("=" * 60)

    class LoggingTensor(torch.Tensor):
        """A tensor that logs every operation."""
        @staticmethod
        def __new__(cls, data):
            tensor = torch.as_tensor(data).as_subclass(cls)
            return tensor

        def __repr__(self):
            return f"LoggingTensor({super().__repr__()})"

        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            if kwargs is None:
                kwargs = {}
            print(f"    [dispatch] {func}")
            # Unwrap LoggingTensor inputs
            unwrapped_args = tuple(
                a.as_subclass(torch.Tensor) if isinstance(a, LoggingTensor) else a
                for a in args
            )
            result = func(*unwrapped_args, **kwargs)
            # Re-wrap if result is a tensor
            if isinstance(result, torch.Tensor) and result.dim() > 0:
                return result.as_subclass(cls)
            return result

    x = LoggingTensor(torch.randn(3))
    y = LoggingTensor(torch.randn(3))
    z = x + y
    print(f"  Result: {z}")
    print()


def exp_sparse_subclass():
    print("=" * 60)
    print("2. Custom sparse tensor with dispatch")
    print("=" * 60)

    # Tensor subclasses integrate with the dispatcher
    # They can intercept ops at the Python level before dispatch

    class MaskedTensor(torch.Tensor):
        @staticmethod
        def __new__(cls, data, mask):
            tensor = torch.as_tensor(data).as_subclass(cls)
            tensor.mask = mask.to(tensor.device)
            return tensor

        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            if kwargs is None:
                kwargs = {}
            if func == torch.ops.aten.add.Tensor:
                # Custom add: apply mask
                a, b = args[0], args[1]
                a_val = a.as_subclass(torch.Tensor) if isinstance(a, cls) else a
                b_val = b.as_subclass(torch.Tensor) if isinstance(b, cls) else b
                result = a_val + b_val
                # Combine masks
                new_mask = a.mask & (b.mask if isinstance(b, cls) else True)
                return cls(result, new_mask)
            return func(*args, **kwargs)

    x = MaskedTensor(torch.randn(3), torch.tensor([True, True, False]))
    y = MaskedTensor(torch.randn(3), torch.tensor([True, False, True]))
    z = x + y
    print(f"  x: {x}, mask={x.mask.tolist()}")
    print(f"  y: {y}, mask={y.mask.tolist()}")
    print(f"  z = x+y: mask={z.mask.tolist()} (AND of masks)")
    print()


def exp_compile_with_subclass():
    print("=" * 60)
    print("3. Tensor subclass + torch.compile")
    print("=" * 60)

    print(f"  Tensor subclasses and torch.compile:")
    print(f"    - __torch_dispatch__ is called from Dynamo trace")
    print(f"    - Subclass instances become graph inputs")
    print(f"    - Can break compile if dispatch logic is complex")
    print(f"")
    print(f"  Best practices:")
    print(f"    - Keep __torch_dispatch__ simple (forward to base)")
    print(f"    - Use __torch_function__ for non-dispatch interception")
    print(f"    - Register decomposition for custom behaviors")
    print()


EXPERIMENTS = {
    "logging": exp_subclass_basics,
    "sparse": exp_sparse_subclass,
    "compile": exp_compile_with_subclass,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[custom_ops case 6] DONE")


if __name__ == "__main__":
    main()
