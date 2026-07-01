"""functorch case study 7: MAML and meta-learning with functional_call + vmap.

Companion script for functorch/functorch.md. Covers:
  1. Model-Agnostic Meta-Learning (MAML) inner loop
  2. Higher-order gradients
  3. Per-task adaptation via vmap

Run:
    python 07_meta_learning.py
"""

import sys

import torch
from torch.func import vmap, grad, functional_call


def exp_maml_inner_loop():
    print("=" * 60)
    print("1. MAML inner loop: gradient through gradient")
    print("=" * 60)

    model = torch.nn.Linear(2, 1)
    params = dict(model.named_parameters())

    def inner_loss(params, x, y):
        pred = functional_call(model, params, x)
        return ((pred - y) ** 2).mean()

    def adapted_params(params, x, y, inner_lr=0.1):
        """One step of SGD in inner loop."""
        grads = grad(inner_loss)(params, x, y)
        return {k: v - inner_lr * grads.get(k, torch.zeros_like(v))
                for k, v in params.items()}

    x_task = torch.randn(8, 2)
    y_task = torch.randn(8, 1)

    adapted = adapted_params(params, x_task, y_task)

    for k in params:
        diff = (adapted[k] - params[k]).norm().item()
        print(f"  {k}: param updated, diff norm = {diff:.6f}")

    print(f"\n  MAML inner loop = gradient descent on task data")
    print(f"  Outer loop = gradient through inner loop (meta-gradient)")
    print()


def exp_meta_gradient():
    print("=" * 60)
    print("2. Meta-gradient: outer loop differentiates through inner loop")
    print("=" * 60)

    model = torch.nn.Linear(2, 1)
    params = dict(model.named_parameters())

    def meta_loss(params, task_x, task_y, query_x, query_y):
        # Inner loop: adapt
        inner_grads = grad(lambda p: functional_call(model, p, task_x).sub(task_y).pow(2).mean())(params)
        adapted = {k: v - 0.1 * inner_grads.get(k, torch.zeros_like(v)) for k, v in params.items()}
        # Outer loop: evaluate on query set
        pred = functional_call(model, adapted, query_x)
        return ((pred - query_y) ** 2).mean()

    task_x, task_y = torch.randn(8, 2), torch.randn(8, 1)
    query_x, query_y = torch.randn(4, 2), torch.randn(4, 1)

    meta_grads = grad(meta_loss)(params, task_x, task_y, query_x, query_y)

    print(f"  Meta-gradients:")
    for k in meta_grads:
        if meta_grads[k] is not None:
            print(f"    {k}: norm = {meta_grads[k].norm().item():.6f}")
    print(f"  -> Second-order derivatives (grad of grad)")
    print()


EXPERIMENTS = {
    "inner": exp_maml_inner_loop,
    "meta": exp_meta_gradient,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[functorch case 7] DONE")


if __name__ == "__main__":
    main()
