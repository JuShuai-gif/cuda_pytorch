"""functorch case study 5: model ensembling with vmap.

Companion script for functorch/functorch.md. Covers:
  1. Ensemble of models via vmap
  2. Stacked parameters pattern
  3. vmap + nn.Module

Run:
    python 05_ensemble_vmap.py
"""

import sys

import torch
from torch.func import vmap, grad


def exp_ensemble_forward():
    print("=" * 60)
    print("1. Ensemble inference with vmap")
    print("=" * 60)

    # Create 5 independent models
    n_models = 5
    models = [torch.nn.Linear(8, 4) for _ in range(n_models)]

    # Stack weights: shape [5, 4, 8] and [5, 4]
    stacked_w = torch.stack([m.weight.data for m in models])
    stacked_b = torch.stack([m.bias.data for m in models])

    # Naive for-loop ensemble
    x = torch.randn(3, 8)
    outputs_loop = torch.stack([m(x) for m in models])  # [5, 3, 4]

    # Vmap ensemble: treat stacked params as batched
    def predict(weight, bias, x):
        return x @ weight.t() + bias

    outputs_vmap = vmap(predict, in_dims=(0, 0, None))(stacked_w, stacked_b, x)

    print(f"  {n_models} models, Linear(8,4), batch=3")
    print(f"  For-loop output shape: {list(outputs_loop.shape)}")
    print(f"  Vmap output shape:     {list(outputs_vmap.shape)}")
    print(f"  Match: {torch.allclose(outputs_loop, outputs_vmap)}")

    # Ensemble average
    avg_loop = outputs_loop.mean(0)
    avg_vmap = outputs_vmap.mean(0)
    print(f"  Ensemble avg match: {torch.allclose(avg_loop, avg_vmap)}")
    print()


def exp_ensemble_grad():
    print("=" * 60)
    print("2. Ensemble gradient: gradient through all models")
    print("=" * 60)

    n_models = 3
    stacked_w = torch.stack([torch.randn(4, 8) for _ in range(n_models)])
    stacked_b = torch.stack([torch.randn(4) for _ in range(n_models)])
    stacked_w.requires_grad_(True)
    stacked_b.requires_grad_(True)

    x = torch.randn(16, 8)

    def ensemble_loss(weight, bias, x):
        return (x @ weight.t() + bias).pow(2).sum()

    loss = vmap(ensemble_loss, in_dims=(0, 0, None))(stacked_w, stacked_b, x).sum()
    loss.backward()

    print(f"  Total loss: {loss.item():.4f}")
    print(f"  Stacked weight grad shape: {stacked_w.grad.shape}")
    print(f"    -> Each model gets independent gradient")
    print()


def exp_ensemble_batchnorm():
    print("=" * 60)
    print("3. Ensemble of BatchNorm models")
    print("=" * 60)

    # vmap over nn.Module requires model params to be batched
    # For BatchNorm: running stats are problematic
    # Use torch.func.functional_call for this

    print(f"  Ensemble BatchNorm challenge:")
    print(f"    - Each model has own running_mean/var")
    print(f"    - vmap over nn.Module needs all params batched")
    print(f"")
    print(f"  Solution: torch.func.functional_call")
    print(f"    from torch.func import functional_call")
    print(f"    params = {k: torch.stack([m.state_dict()[k] for m in models])}")
    print(f"    outputs = vmap(partial(functional_call, model), in_dims=(0, None))(params, x)")
    print()


EXPERIMENTS = {
    "forward": exp_ensemble_forward,
    "grad": exp_ensemble_grad,
    "bn": exp_ensemble_batchnorm,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[functorch case 5] DONE")


if __name__ == "__main__":
    main()
