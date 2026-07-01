"""AOTAutograd case study 3: decomposition and precision.

Companion script for aot_autograd/aot_autograd.md. Covers:
  1. Check which ops have decompositions
  2. Measure precision difference from decomposition
  3. Compare compiled vs eager numerical accuracy

Run:
    python 03_decomposition_precision.py
"""

import sys

import torch


def exp_check_decomps():
    print("=" * 60)
    print("1. Check which ops have decompositions registered")
    print("=" * 60)

    from torch._decomp import get_decompositions

    common_ops = [
        torch.ops.aten.layer_norm,
        torch.ops.aten.batch_norm,
        torch.ops.aten.gelu,
        torch.ops.aten.silu,
        torch.ops.aten.softmax.int,
    ]

    for op in common_ops:
        try:
            decomps = get_decompositions([op])
            name = str(op).split(".")[-1]
            if decomps:
                print(f"  {name:20s}: has decomposition ({len(decomps)} funcs)")
            else:
                print(f"  {name:20s}: no decomposition")
        except Exception as e:
            print(f"  {str(op)[:30]}: ERROR {str(e)[:50]}")
    print()


def exp_precision_comparison():
    print("=" * 60)
    print("2. Eager vs compiled numerical precision")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return

    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(128, 256),
                torch.nn.GELU(),
                torch.nn.Linear(256, 128),
            )

        def forward(self, x):
            return self.layers(x)

    model = TestModel().cuda()
    x = torch.randn(32, 128, device="cuda")

    # Eager
    with torch.no_grad():
        y_eager = model(x)

    # Compiled
    compiled = torch.compile(model)
    with torch.no_grad():
        y_compiled = compiled(x)

    diff = (y_eager - y_compiled).abs().max().item()
    relative = (diff / y_eager.abs().max().item()) if y_eager.abs().max() > 0 else 0.0

    print(f"  Eager vs Compiled:")
    print(f"    Max absolute diff: {diff:.2e}")
    print(f"    Max relative diff: {relative:.2e}")
    print(f"    Mean output eager:   {y_eager.mean().item():.6f}")
    print(f"    Mean output compiled: {y_compiled.mean().item():.6f}")

    tolerance = 5e-5 if x.dtype == torch.float32 else 1e-2
    if diff < tolerance:
        print(f"    PASS (diff < {tolerance})")
    else:
        print(f"    NOTE: diff exceeds {tolerance}, check decomposition")
    print()


def exp_backward_precision():
    print("=" * 60)
    print("3. Eager vs compiled backward (gradient) precision")
    print("=" * 60)

    if not torch.cuda.is_available():
        return

    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(64, 64)

        def forward(self, x):
            return torch.nn.functional.gelu(self.linear(x))

    model1 = TestModel().cuda()
    model2 = TestModel().cuda()
    model2.load_state_dict(model1.state_dict())  # Same initial weights

    x1 = torch.randn(16, 64, device="cuda", requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)

    # Eager
    loss1 = model1(x1).sum()
    loss1.backward()

    # Compiled
    compiled = torch.compile(model2)
    loss2 = compiled(x2).sum()
    loss2.backward()

    grad_diff = (x1.grad - x2.grad).abs().max().item()
    print(f"  Max gradient diff: {grad_diff:.2e}")

    # Weight gradient
    param_names = [n for n, _ in model1.named_parameters()]
    for n1, p1 in model1.named_parameters():
        p2 = model2.get_parameter(n1)
        if p1.grad is not None and p2.grad is not None:
            w_diff = (p1.grad - p2.grad).abs().max().item()
            print(f"  {n1}: grad max diff = {w_diff:.2e}")

    print()


EXPERIMENTS = {
    "decomps": exp_check_decomps,
    "precision": exp_precision_comparison,
    "backward": exp_backward_precision,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[aot_autograd case 3] DONE")


if __name__ == "__main__":
    main()
