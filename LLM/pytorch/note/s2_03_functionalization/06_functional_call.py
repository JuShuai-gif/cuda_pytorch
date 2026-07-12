"""Functionalization case study 6: functional_call and parameter freezing.

Companion script for functionalization/functionalization.md. Covers:
  1. torch.func.functional_call for parameter control
  2. Parameter freezing through functionalization
  3. stateless module patterns

Run:
    python 06_functional_call.py
"""

import sys

import torch


def exp_functional_call():
    print("=" * 60)
    print("1. functional_call: stateless module execution")
    print("=" * 60)

    import torch.nn as nn

    model = nn.Linear(4, 3)
    x = torch.randn(2, 4)

    # Normal stateful call
    y_stateful = model(x)

    # Functional call: explicitly pass parameters
    from torch.func import functional_call
    params = dict(model.named_parameters())
    y_functional = functional_call(model, params, x)

    print(f"  Stateful:   {y_stateful.tolist()}")
    print(f"  Functional: {y_functional.tolist()}")
    print(f"  Match: {torch.allclose(y_stateful, y_functional)}")

    # Swap parameters: use different weights
    new_params = {k: v * 0.5 for k, v in params.items()}
    y_swapped = functional_call(model, new_params, x)
    print(f"\n  With swapped params (weight*0.5): {y_swapped.tolist()}")
    print()


def exp_parameter_freezing():
    print("=" * 60)
    print("2. Parameter freezing via functionalization")
    print("=" * 60)

    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 4),
    )

    # Freeze first Linear
    for name, param in model.named_parameters():
        if "0." in name:  # first layer
            param.requires_grad_(False)
            print(f"  Frozen: {name}")

    x = torch.randn(4, 8)
    y = model(x)
    y.sum().backward()

    # Check which params got gradients
    for name, param in model.named_parameters():
        status = "grad" if param.grad is not None else "no grad"
        print(f"  {name:20s}: requires_grad={param.requires_grad}, {status}")

    print(f"\n  torch.compile also respects requires_grad:")
    compiled = torch.compile(model)
    y_c = compiled(x.clone())
    print(f"  Compile OK: {y_c.sum().item():.4f}")
    print()


def exp_vmap_functional():
    print("=" * 60)
    print("3. vmap + functional_call for ensemble")
    print("=" * 60)

    from torch.func import vmap, functional_call

    model = torch.nn.Linear(4, 3)
    n_models = 5

    # Create 5 sets of parameters
    params = dict(model.named_parameters())
    batched_params = {k: torch.stack([v + torch.randn_like(v) * 0.1 for _ in range(n_models)])
                      for k, v in params.items()}

    x = torch.randn(2, 4)

    # Vmap over batched parameters
    ensemble_fn = vmap(
        lambda p: functional_call(model, p, x),
        in_dims=({k: 0 for k in batched_params},),
    )
    outputs = ensemble_fn(batched_params)
    print(f"  {n_models} models, Linear(4,3):")
    print(f"  Ensemble output shape: {list(outputs.shape)}  (5 models, 2 samples, 3 outputs)")
    print()


EXPERIMENTS = {
    "func_call": exp_functional_call,
    "freeze": exp_parameter_freezing,
    "vmap": exp_vmap_functional,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[functionalization case 6] DONE")


if __name__ == "__main__":
    main()
