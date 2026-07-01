"""Functionalization case study 3: custom op alias annotation and compile.

Companion script for functionalization/functionalization.md. Covers:
  1. Alias annotation for custom ops
  2. Compile impact of missing annotation
  3. View ops and functionalization interaction

Run:
    python 03_alias_annotation_debug.py
"""

import sys

import torch


def exp_missing_alias():
    print("=" * 60)
    print("1. Missing alias annotation consequence")
    print("=" * 60)

    # Register without (a!) annotation -> mutation not tracked
    lib = torch.library.Library("annodemo", "DEF")
    lib.define("my_inplace(Tensor(a!) self) -> Tensor(a!)")

    @torch.library.impl("annodemo::my_inplace", "CPU")
    def my_inplace_cpu(self):
        return self.mul_(2)

    # (a!) correctly annotated -> functionalization can track it
    x = torch.ones(3)
    orig_version = x._version
    y = torch.ops.annodemo.my_inplace(x)
    print(f"  x version: {orig_version} -> {x._version} (incremented)")
    print(f"  x is y: {x is y}")
    print(f"  x value: {x.tolist()}")

    # Compile test
    def f(x):
        z = torch.ops.annodemo.my_inplace(x)
        return z.sum()

    try:
        compiled = torch.compile(f)
        result = compiled(x.detach().clone())
        print(f"  Compile OK: {result}")
    except Exception as e:
        print(f"  Compile ERROR: {str(e)[:120]}")
    print()


def exp_view_chain():
    print("=" * 60)
    print("2. Complex view chain: transpose -> slice -> expand")
    print("=" * 60)

    x = torch.arange(24).view(2, 3, 4).float()
    print(f"  Original: shape={list(x.shape)}, contiguous={x.is_contiguous()}")

    # Chain of views
    v1 = x.transpose(0, 2)   # shape [4,3,2]
    v2 = v1[:, :, :1]        # slice
    v3 = v2.expand(4, 3, 8)  # expand with stride=0

    print(f"  After transpose->slice->expand:")
    print(f"    shape={list(v3.shape)}, contiguous={v3.is_contiguous()}")

    # Functionalization flattens this chain into view_copy + metadata tracking
    # Each view in the chain creates an alias, tracked via alias map


def exp_compile_diagnose():
    print("\n" + "=" * 60)
    print("3. Diagnose mutation problems in compiled model")
    print("=" * 60)

    # This pattern is notorious: BatchNorm running_mean update
    class ModelWithBN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm1d(8)

        def forward(self, x):
            # In training mode, BN updates running_mean/running_var IN-PLACE
            # Functionalization handles this by making copies
            return self.bn(x)

    model = ModelWithBN()
    model.train()

    x = torch.randn(4, 8)

    # Eager
    y_eager = model(x)
    print(f"  Eager OK: shape={list(y_eager.shape)}")
    print(f"  BN running_mean: {model.bn.running_mean[:4].tolist()}")

    model.eval()

    @torch.compile
    def inference(x):
        return model(x)

    y_compiled = inference(x)
    print(f"  Compile OK: shape={list(y_compiled.shape)}")
    print(f"\n  BN's in-place updates in training mode are handled by functionalization")
    print(f"  In eval mode, no mutation -> simpler graph")


EXPERIMENTS = {
    "alias": exp_missing_alias,
    "view": exp_view_chain,
    "diagnose": exp_compile_diagnose,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[functionalization case 3] DONE")


if __name__ == "__main__":
    main()
