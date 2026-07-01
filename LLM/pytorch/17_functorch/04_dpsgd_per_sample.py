"""functorch case study 4: vmap + grad for per-sample gradient in DP-SGD.

Companion script for functorch/functorch.md. Covers:
  1. Differential Privacy SGD per-sample gradient
  2. Memory comparison: vmap vs for-loop
  3. Stacking per-sample gradients

Run:
    python 04_dpsgd_per_sample.py
"""

import sys

import torch
from torch.func import vmap, grad


def exp_per_sample_gradients():
    print("=" * 60)
    print("1. Per-sample gradients for DP-SGD")
    print("=" * 60)

    model = torch.nn.Linear(16, 4)

    def sample_loss(params, x, y):
        w, b = params
        logits = x @ w.t() + b
        return torch.nn.functional.cross_entropy(logits.unsqueeze(0), y.unsqueeze(0))

    w = model.weight.detach()
    b = model.bias.detach()

    x_batch = torch.randn(32, 16)
    y_batch = torch.randint(0, 4, (32,))

    # Vmap over samples -> per-sample gradients
    compute_grad = grad(sample_loss, argnums=0)
    batch_grad_fn = vmap(compute_grad, in_dims=(None, 0, 0))

    per_sample_w_grad, per_sample_b_grad = batch_grad_fn((w, b), x_batch, y_batch)

    print(f"  Batch: 32 samples, Linear(16, 4)")
    print(f"  Weight grads shape: {list(per_sample_w_grad.shape)}  # [32, 4, 16]")
    print(f"  Bias grads shape:   {list(per_sample_b_grad.shape)}  # [32, 4]")

    # Sum of per-sample grads = batch gradient
    model.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(x_batch), y_batch)
    loss.backward()

    sum_w = per_sample_w_grad.sum(0)
    print(f"  Sum(per_sample) vs batch grad: {torch.allclose(sum_w, model.weight.grad, atol=1e-5)}")
    print()


def exp_per_sample_norms():
    print("=" * 60)
    print("2. Per-sample gradient norm clipping (DP-SGD)")
    print("=" * 60)

    model = torch.nn.Linear(8, 4)
    w = model.weight.detach()
    b = model.bias.detach()

    def sample_grad_norm(x, y):
        g_w, g_b = grad(sample_loss_fn := lambda w, b, x, y:
            torch.nn.functional.cross_entropy((x @ w + b).unsqueeze(0), y.unsqueeze(0))
        )(w, b)
        # Flatten and compute L2 norm
        flat_grad = torch.cat([g_w.flatten(), g_b.flatten()])
        return flat_grad.norm(p=2)

    # Vmap over batch
    batch_norms = vmap(sample_grad_norm, in_dims=(0, 0))(x_batch, y_batch)
    print(f"  Per-sample gradient norms: min={batch_norms.min():.3f}, max={batch_norms.max():.3f}")
    print(f"  DP-SGD clips each sample's gradient to C (e.g., C=1.0)")
    print()


# Need these for the function above
x_batch = torch.randn(32, 8)
y_batch = torch.randint(0, 4, (32,))

def sample_loss_fn(w, b, x, y):
    return torch.nn.functional.cross_entropy((x @ w + b).unsqueeze(0), y.unsqueeze(0))


EXPERIMENTS = {
    "dpsgd": exp_per_sample_gradients,
    "norms": exp_per_sample_norms,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[functorch case 4] DONE")


if __name__ == "__main__":
    main()
