"""functorch case study 6: vjp and jvp for efficient Jacobian products.

Companion script for functorch/functorch.md. Covers:
  1. vjp (vector-Jacobian product) = reverse-mode
  2. jvp (Jacobian-vector product) = forward-mode
  3. When to use each

Run:
    python 06_vjp_jvp.py
"""

import sys

import torch
from torch.func import vjp, jvp, grad


def exp_vjp_basics():
    print("=" * 60)
    print("1. vjp: vector-Jacobian product (reverse mode)")
    print("=" * 60)

    def f(x):
        return torch.stack([x.sum(), x.prod()])

    x = torch.randn(4)
    v = torch.randn(2)  # vector in output space

    # vjp: d/dx (v^T f(x))
    _, vjp_fn = vjp(f, x)
    result = vjp_fn(v)[0]
    print(f"  f(x): {f(x).tolist()}")
    print(f"  vjp(f, x)(v): {result.tolist()}")

    # Equivalent: sum(grad(f_i, x) * v_i)
    grads = []
    for i in range(2):
        g, = torch.autograd.grad(f(x)[i], x, create_graph=True, retain_graph=True)
        grads.append(g)
    manual = sum(g * v_i for g, v_i in zip(grads, v))
    print(f"  Manual: {manual.tolist()}")
    print(f"  Match: {torch.allclose(result, manual)}")
    print()


def exp_jvp_basics():
    print("=" * 60)
    print("2. jvp: Jacobian-vector product (forward mode)")
    print("=" * 60)

    def f(x):
        return torch.stack([x.sum(), x.prod()])

    x = torch.randn(4)
    u = torch.randn(4)  # tangent vector

    # jvp: d/depsilon f(x + epsilon * u) at epsilon=0
    _, result = jvp(f, (x,), (u,))

    print(f"  f(x):     {f(x).tolist()}")
    print(f"  direction: {u.tolist()}")
    print(f"  jvp(f, x)(u): {result.tolist()}")
    print(f"  -> Rate of change of f along direction u")
    print()


def exp_vjp_jvp_neural_net():
    print("=" * 60)
    print("3. vjp/jvp in neural network context")
    print("=" * 60)

    model = torch.nn.Linear(8, 4)

    def neural_net(params_vec, x):
        w, b = torch.split(params_vec, [32, 4])
        w = w.view(4, 8)
        return (x @ w.t() + b).relu().sum()

    params = torch.cat([model.weight.flatten(), model.bias])
    x = torch.randn(2, 8)

    # vjp: d/dparams (grad_output^T * f(params, x)) -> gradient
    v = torch.tensor(1.0)
    _, vjp_fn = vjp(lambda p: neural_net(p, x), params)
    gradient = vjp_fn(v)[0]

    print(f"  Neural net gradient norm (vjp): {gradient.norm().item():.4f}")
    print(f"  vjp is the mathematical basis for backpropagation")
    print()


EXPERIMENTS = {
    "vjp": exp_vjp_basics,
    "jvp": exp_jvp_basics,
    "neural": exp_vjp_jvp_neural_net,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[functorch case 6] DONE")


if __name__ == "__main__":
    main()
