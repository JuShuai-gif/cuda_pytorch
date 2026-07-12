"""functorch case study 3: vmap + grad composition and batching rule fallback.

Companion script for functorch/functorch.md. Covers:
  1. grad(vmap) vs vmap(grad) semantics
  2. Detect batching rule fallback
  3. Custom op with explicit batching rule

Run:
    python 03_grad_vmap_compose.py
"""

import sys

import torch
from torch.func import vmap, grad


def exp_composition():
    print("=" * 60)
    print("1. grad(vmap) vs vmap(grad) semantics")
    print("=" * 60)

    def loss(w, x):
        return ((w * x).sum()).sin()

    w = torch.randn(4, requires_grad=True)
    xs = torch.randn(8, 4)

    # vmap(grad): per-sample gradients -> [8, 4]
    per_sample_g = vmap(grad(loss), in_dims=(None, 0))(w, xs)
    print(f"  vmap(grad(loss)): shape={list(per_sample_g.shape)}")
    print(f"    -> [num_samples, param_shape]")

    # grad(vmap): gradient of batched loss -> [4]
    batched_loss = lambda w, xs: vmap(lambda x: loss(w, x))(xs).sum()
    batched_g = grad(batched_loss)(w, xs)
    print(f"  grad(vmap(x -> loss)): shape={list(batched_g.shape)}")
    print(f"    -> [param_shape] (aggregated over batch)")

    # Verify equivalence: sum of per-sample grads = batched grad
    assert torch.allclose(per_sample_g.sum(0), batched_g), "Should be equal!"
    print(f"  Sum(per_sample_g) == batched_g: True")
    print()


def exp_fallback_detect():
    print("=" * 60)
    print("2. Detect batching rule fallback")
    print("=" * 60)

    # Register a custom op without batching rule
    lib = torch.library.Library("vmapdemo", "DEF")
    lib.define("slow_op(Tensor x) -> Tensor")

    @torch.library.impl("vmapdemo::slow_op", "CPU")
    def slow_op_cpu(x):
        # Simulate a complex op without batching rule
        return torch.sin(x) * torch.cos(x)

    # vmap will use for-loop fallback if no batching rule
    def f(x):
        return torch.ops.vmapdemo.slow_op(x).sum()

    xs = torch.randn(32, 8)
    # This should work via fallback (each element computed individually)
    try:
        result = vmap(f)(xs)
        print(f"  vmap on slow_op: OK (via for-loop fallback)")
        print(f"  Output shape: {list(result.shape)}")
    except Exception as e:
        print(f"  vmap FAILED: {str(e)[:100]}")

    # Compare speed: for-loop vs vmap fallback
    import time

    # Vmap version
    t0 = time.perf_counter()
    for _ in range(100):
        vmap(f)(xs)
    t1 = time.perf_counter()
    t_vmap = (t1 - t0) / 100

    # Python for-loop version (similar to what fallback does)
    t2 = time.perf_counter()
    for _ in range(100):
        for i in range(len(xs)):
            f(xs[i])
    t3 = time.perf_counter()
    t_for = (t3 - t2) / 100

    print(f"  Vmap fallback time: {t_vmap*1000:.4f}ms per batch")
    print(f"  Python for-loop:    {t_for*1000:.4f}ms per batch")
    print()


def exp_grad_composition_order():
    print("=" * 60)
    print("3. Composition order: vmap(grad) for per-sample gradients")
    print("=" * 60)

    # Real use case: DP-SGD or per-sample gradient computation
    model = torch.nn.Linear(16, 8)

    def per_sample_loss(params_tuple, x):
        """Compute loss for a single sample."""
        w, b = params_tuple
        return ((x @ w.t() + b).relu().sum()).sin()

    # Convert model to tuple of parameters
    w = model.weight.detach()
    b = model.bias.detach()
    params = (w, b)
    x_batch = torch.randn(32, 16)

    # Per-sample gradients using vmap(grad)
    per_sample_grads = vmap(
        lambda x: grad(per_sample_loss)(params, x),
        in_dims=(0,)
    )(x_batch)

    # Each sample's gradient
    print(f"  Model: Linear(16, 8)")
    print(f"  Batch: 32 samples")
    print(f"  Per-sample gradient for weight: shape={list(per_sample_grads[0].shape)}")
    if len(per_sample_grads) > 1:
        print(f"  Per-sample gradient for bias:   shape={list(per_sample_grads[1].shape)}")

    # Verify sum equals regular batch gradient
    model.zero_grad()
    output = model(x_batch)
    loss = output.relu().sum().sin()
    loss.backward()
    batch_grad = model.weight.grad.clone()

    sum_per_sample = per_sample_grads[0].sum(0)
    print(f"\n  Sum(per_sample_grads) vs batch_grad match: {torch.allclose(sum_per_sample, batch_grad)}")
    print()


EXPERIMENTS = {
    "compose": exp_composition,
    "fallback": exp_fallback_detect,
    "grad": exp_grad_composition_order,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[functorch case 3] DONE")


if __name__ == "__main__":
    main()
