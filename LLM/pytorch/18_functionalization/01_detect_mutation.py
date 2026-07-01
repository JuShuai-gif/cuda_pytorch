"""Functionalization case study 1: detect in-place mutation with FunctionalTensorMode.

Companion script for functionalization/functionalization.md. Covers:
  1. FunctionalTensorMode: isolate mutation behavior
  2. Detect silent in-place operations
  3. Compare view vs view_copy semantics

Run:
    python 01_detect_mutation.py
"""

import sys

import torch


def exp_faketensor_mode():
    print("=" * 60)
    print("1. FunctionalTensorMode: mutation isolation")
    print("=" * 60)

    def bad_func(x):
        y = x.view(-1)
        y.add_(1)  # in-place mutation through view alias
        return x * 2

    x = torch.ones(2, 3)
    y = bad_func(x)
    print(f"  Eager mode:")
    print(f"    Input x:     {x}")
    print(f"    Output (x*2): {y}")
    print(f"    x modified:  {x.tolist()} (yes, view+inplace!)")

    # FunctionalTensorMode: prevents side-effects
    from torch._subclasses.functional_tensor import FunctionalTensorMode

    x2 = torch.ones(2, 3)
    with FunctionalTensorMode():
        try:
            y2 = bad_func(x2)
            print(f"\n  Functional mode output: {y2}")
        except Exception as e:
            print(f"\n  Functional mode error: {str(e)[:100]}")
    print()


def exp_detect_mutation():
    print("=" * 60)
    print("2. Detect hidden mutations in model code")
    print("=" * 60)

    class SusModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.buffer = torch.nn.Parameter(torch.zeros(3))

        def forward(self, x):
            # In-place modification of own parameter
            self.buffer.add_(x.mean(dim=0))
            return x + self.buffer

    model = SusModel()

    from torch._subclasses.functional_tensor import FunctionalTensorMode

    with FunctionalTensorMode():
        x = torch.randn(2, 3)
        try:
            y = model(x)
            print(f"  Model forward OK, output shape: {list(y.shape)}")
            print(f"  buffer value: {model.buffer.tolist()}")
        except Exception as e:
            print(f"  FunctionalTensorMode caught mutation issue: {str(e)[:100]}")


def exp_view_vs_copy():
    print("\n" + "=" * 60)
    print("3. View vs view_copy: compiler perspective")
    print("=" * 60)

    x = torch.arange(6).view(2, 3).float()

    # x is contiguous 2x3
    print(f"  x: shape={list(x.shape)}, stride={x.stride()}, contiguous={x.is_contiguous()}")

    # view: zero-copy
    y_view = x.view(-1)
    print(f"\n  view(-1):  shape={list(y_view.shape)}, same storage={x.storage().data_ptr() == y_view.storage().data_ptr()}")
    print(f"             data_ptr diff: {y_view.data_ptr() - x.data_ptr()}")

    # As compiler sees it: view -> alias -> not safe to reorder
    print(f"  Compiler issue: view creates alias -> cannot assume independent memory")

    # view_copy: copy
    y_copy = x.contiguous().view(-1)
    print(f"\n  view_copy: contiguous().view(-1)")
    print(f"             same storage={x.storage().data_ptr() == y_copy.storage().data_ptr()}")
    print(f"  Compiler benefit: independent memory -> safe to reorder/fuse")

    # After transpose: view fails
    x_t = x.t()
    print(f"\n  After transpose: contiguous={x_t.is_contiguous()}")
    try:
        x_t.view(-1)
    except RuntimeError as e:
        print(f"  x.t().view(-1) ERROR: {str(e)[:80]}")

    # reshape handles it
    z = x_t.reshape(-1)
    print(f"  x.t().reshape(-1): OK, first creates contiguous copy internally")
    print()


EXPERIMENTS = {
    "fake_mode": exp_faketensor_mode,
    "mutation": exp_detect_mutation,
    "view_copy": exp_view_vs_copy,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[functionalization case 1] DONE")


if __name__ == "__main__":
    main()
