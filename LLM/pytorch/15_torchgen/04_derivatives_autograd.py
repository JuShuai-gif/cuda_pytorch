"""torchgen case study 4: backward formula via derivatives.yaml.

Companion script for torchgen/torchgen.md. Covers:
  1. How derivatives.yaml connects to torchgen
  2. Custom autograd formula impact
  3. Differentiability checking

Run:
    python 04_derivatives_autograd.py
"""

import sys

import torch


def exp_view_grad_formula():
    print("=" * 60)
    print("1. View operation autograd: derivatives.yaml role")
    print("=" * 60)

    # view in derivatives.yaml:
    # name: view(Tensor self, SymInt[] size) -> Tensor
    #   self: view_backward(grad, self.sizes())
    # view_backward = reshape of incoming grad to original shape

    x = torch.randn(2, 3, requires_grad=True)
    y = x.view(-1)  # view(2,3) -> view(6)
    z = y.sum()
    z.backward()

    print(f"  x.shape: {list(x.shape)}")
    print(f"  y = x.view(-1): shape={list(y.shape)}")
    print(f"  x.grad: {x.grad.tolist()}")
    print(f"  -> view_backward reshapes grad to original shape")
    print(f"  -> Defined in tools/autograd/derivatives.yaml")
    print()


def exp_index_grad():
    print("=" * 60)
    print("2. Indexing gradient formula in derivatives.yaml")
    print("=" * 60)

    # Indexing op gradient:
    # The gradient flows back only to indexed positions
    x = torch.randn(4, 4, requires_grad=True)
    idx = torch.tensor([0, 2])
    y = x[idx]  # select rows 0 and 2
    z = y.sum()
    z.backward()

    print(f"  x[idx].sum(), idx=[0,2]:")
    print(f"    x.grad[0]: {x.grad[0].tolist()}  (non-zero, selected)")
    print(f"    x.grad[1]: {x.grad[1].tolist()}  (zero, not selected)")
    print(f"    x.grad[2]: {x.grad[2].tolist()}  (non-zero, selected)")
    print(f"    x.grad[3]: {x.grad[3].tolist()}  (zero, not selected)")
    print()

    # Embedding gradient: similar to index_select
    emb = torch.nn.Embedding(10, 4)
    ids = torch.tensor([3, 7])
    out = emb(ids)
    out.sum().backward()

    print(f"  Embedding backward:")
    print(f"    emb.weight.grad[3]: {'non-zero' if emb.weight.grad[3].abs().sum() > 0 else 'zero'} (indexed)")
    print(f"    emb.weight.grad[5]: {'non-zero' if emb.weight.grad[5].abs().sum() > 0 else 'zero'} (not indexed)")
    print()


def exp_custom_backward_vs_compiled():
    print("=" * 60)
    print("3. Custom autograd.function vs compiled graph")
    print("=" * 60)

    class MyReLU(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x.clamp(min=0)

        @staticmethod
        def backward(ctx, grad_output):
            x, = ctx.saved_tensors
            return grad_output * (x > 0).float()

    # Eager custom backward
    x1 = torch.randn(4, 4, requires_grad=True)
    y1 = MyReLU.apply(x1)
    y1.sum().backward()
    g1 = x1.grad.clone()

    # Built-in relu
    x2 = torch.randn(4, 4, requires_grad=True)
    x2.data.copy_(x1.data)
    y2 = torch.relu(x2)
    y2.sum().backward()
    g2 = x2.grad.clone()

    print(f"  Custom backward vs built-in relu:")
    print(f"    Max diff: {(g1 - g2).abs().max().item():.2e}")

    # AOTAutograd trace: custom autograd.Function breaks tracing
    def f_custom(x):
        return MyReLU.apply(x).sum()

    def f_builtin(x):
        return torch.relu(x).sum()

    try:
        compiled_custom = torch.compile(f_custom)
        r = compiled_custom(torch.randn(4))
        print(f"    Compile custom: OK")
    except Exception as e:
        print(f"    Compile custom: {str(e)[:80]}")
    print()


def exp_check_differentiability():
    print("=" * 60)
    print("4. Check which ops are registered as differentiable")
    print("=" * 60)

    # Check if op has autograd support
    x = torch.randn(3, requires_grad=True)

    differentiate_ops = [
        ("+", lambda x: x + 1),
        ("*", lambda x: x * 2),
        ("abs", lambda x: x.abs()),
        ("sqrt", lambda x: (x**2).sqrt()),
        ("clamp", lambda x: x.clamp(0, 1)),
    ]

    for name, fn in differentiate_ops:
        try:
            y = fn(x)
            y.sum().backward(retain_graph=True)
            x.grad = None
            print(f"  {name:10s}: differentiable (grad OK)")
        except Exception as e:
            print(f"  {name:10s}: {str(e)[:50]}")

    print(f"\n  derivatives.yaml format:")
    print(f"    name: my_op(Tensor self) -> Tensor")
    print(f"      self: my_op_backward(grad, self)")
    print()


EXPERIMENTS = {
    "view": exp_view_grad_formula,
    "index": exp_index_grad,
    "custom": exp_custom_backward_vs_compiled,
    "diff": exp_check_differentiability,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[torchgen case 4] DONE")


if __name__ == "__main__":
    main()
