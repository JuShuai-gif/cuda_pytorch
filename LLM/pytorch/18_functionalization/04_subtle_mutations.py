"""Functionalization case study 4: detect subtle mutation patterns.

Companion script for functionalization/functionalization.md. Covers:
  1. Parameter mutation detection
  2. nn.Module in-place pitfalls
  3. Mutation in custom Module

Run:
    python 04_subtle_mutations.py
"""

import sys

import torch
from torch._subclasses.functional_tensor import FunctionalTensorMode


def exp_param_mutation():
    print("=" * 60)
    print("1. Parameter in-place mutation detection")
    print("=" * 60)

    class MutatingModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, x):
            self.scale.data.add_(0.1)  # mutation of parameter!
            return x * self.scale

    model = MutatingModule()

    # Eager: successive calls accumulate
    x = torch.tensor([1.0, 2.0, 3.0])
    y1 = model(x)
    y2 = model(x)
    print(f"  Call 1: {y1.tolist()}  (scale={model.scale.item():.1f})")
    print(f"  Call 2: {y2.tolist()}  (scale={model.scale.item():.1f})")
    print(f"  -> Parameter was mutated (scale grows each call)")

    # FunctionalTensorMode: detects this pattern
    model2 = MutatingModule()
    with FunctionalTensorMode():
        try:
            y = model2(x)
            print(f"  Functional ok: {y.tolist()}")
        except Exception as e:
            print(f"  Functional error: {str(e)[:100]}")

    print(f"\n  torch.compile may behave differently if param is mutated")
    print()


def exp_silent_view_mutation():
    print("=" * 60)
    print("2. Silent mutation through view chain")
    print("=" * 60)

    # This pattern is dangerous: modify view -> corrupt original
    x = torch.ones(4, 4)
    y = x[::2, ::2]     # view with non-trivial strides
    y *= 2              # in-place on view: modifies x!

    print(f"  x (original): {x[0, 0].item()} {x[0, 1].item()} {x[0, 2].item()} {x[0, 3].item()}")
    print(f"                  {x[2, 0].item()} {x[2, 1].item()} {x[2, 2].item()} {x[2, 3].item()}")

    # Functionalization must detect all aliases and update them
    with FunctionalTensorMode():
        x2 = torch.ones(4, 4).clone()
        y2 = x2[::2, ::2]
        try:
            y2 *= 2
            print(f"\n  Functional mode: x2[0,0] = {x2[0,0].item()}")
        except Exception as e:
            print(f"\n  Functional mode error: {str(e)[:100]}")
    print()


def exp_custom_fn_mutation():
    print("=" * 60)
    print("3. Custom autograd.Function with internal mutation")
    print("=" * 60)

    class MyOp(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            out = x.clone()
            out.mul_(2)
            return out

        @staticmethod
        def backward(ctx, grad):
            x, = ctx.saved_tensors
            return grad * 2

    x = torch.randn(3, requires_grad=True)
    y = MyOp.apply(x)
    y.sum().backward()

    print(f"  Forward OK: {y.tolist()}")
    print(f"  Backward OK: grad={x.grad.tolist()}")

    # Custom Function with mutation: AOTAutograd needs decomposition
    def fn(x):
        return MyOp.apply(x).sum()

    try:
        compiled = torch.compile(fn)
        result = compiled(x)
        print(f"  Compile OK: {result:.4f}")
    except Exception as e:
        print(f"  Compile: {str(e)[:80]}")
    print()


EXPERIMENTS = {
    "param": exp_param_mutation,
    "view": exp_silent_view_mutation,
    "fn": exp_custom_fn_mutation,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[functionalization case 4] DONE")


if __name__ == "__main__":
    main()
