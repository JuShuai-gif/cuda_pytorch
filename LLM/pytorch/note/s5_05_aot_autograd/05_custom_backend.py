"""AOTAutograd case study 5: custom backend and inductor integration.

Companion script for aot_autograd/aot_autograd.md. Covers:
  1. Register custom backend for AOTAutograd
  2. Trace through different backends
  3. Compare backend outputs

Run:
    python 05_custom_backend.py
"""

import sys

import torch
from torch._functorch.aot_autograd import aot_function


def exp_custom_backend():
    print("=" * 60)
    print("1. Register custom fake backend for AOTAutograd")
    print("=" * 60)

    def my_fw_backend(gm, example_inputs):
        print(f"  [my_backend] FW graph: {len(list(gm.graph.nodes))} nodes")
        return gm

    def my_bw_backend(gm, example_inputs):
        print(f"  [my_backend] BW graph: {len(list(gm.graph.nodes))} nodes")
        return gm

    def model(x, w):
        return (x @ w).relu().sum()

    x = torch.randn(4, 8, requires_grad=True)
    w = torch.randn(8, 3, requires_grad=True)

    aot_fn = aot_function(model, my_fw_backend, my_bw_backend)
    loss = aot_fn(x, w)
    loss.backward()
    print()


def exp_trace_backend_comparison():
    print("=" * 60)
    print("2. Compare eager vs aot vs compiled outputs")
    print("=" * 60)

    class SmallMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(16, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 16),
            )

        def forward(self, x):
            return self.net(x)

    model1 = SmallMLP()
    model2 = SmallMLP()
    model2.load_state_dict(model1.state_dict())
    model3 = SmallMLP()
    model3.load_state_dict(model1.state_dict())

    x = torch.randn(8, 16, requires_grad=True)
    x2 = x.clone().detach().requires_grad_(True)
    x3 = x.clone().detach().requires_grad_(True)

    # Eager
    y1 = model1(x)
    y1.sum().backward()

    # AOT
    def aot_wrapper(m, xi):
        aot_fn = aot_function(m.forward, lambda gm, _: gm, lambda gm, _: gm)
        return aot_fn(xi).sum()

    # Compile
    compiled = torch.compile(model3)

    y3 = compiled(x3)
    y3.sum().backward()

    print(f"  Eager loss:   {y1.sum().item():.4f}")
    print(f"  Compile loss: {y3.sum().item():.4f}")
    print(f"  Match: {torch.allclose(y1.sum(), y3.sum())}")

    # Compare gradients
    w1_grad = model1.net[0].weight.grad
    w3_grad = model3.net[0].weight.grad
    if w1_grad is not None and w3_grad is not None:
        grad_diff = (w1_grad - w3_grad).abs().max().item()
        print(f"  Grad max diff: {grad_diff:.2e}")
    print()


def exp_multi_output_model():
    print("=" * 60)
    print("3. AOTAutograd with multiple outputs")
    print("=" * 60)

    def multi_output(x):
        y = x.relu()
        z = x.sin()
        return y.sum(), z.sum()

    x = torch.randn(4, 8, requires_grad=True)

    def printer(gm, inputs):
        print(f"  Graph outputs: {len(list(gm.graph.nodes))} nodes")
        return gm

    aot_fn = aot_function(multi_output, printer, printer)
    out1, out2 = aot_fn(x)
    total = out1 + out2
    total.backward()

    print(f"  out1: {out1.item():.4f}")
    print(f"  out2: {out2.item():.4f}")
    print(f"  x.grad exists: {x.grad is not None}")
    print()


EXPERIMENTS = {
    "backend": exp_custom_backend,
    "compare": exp_trace_backend_comparison,
    "multi": exp_multi_output_model,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[aot_autograd case 5] DONE")


if __name__ == "__main__":
    main()
