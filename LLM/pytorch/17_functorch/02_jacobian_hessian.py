"""functorch case study 2: Jacobian / Hessian with jacfwd and jacrev.

Companion script for functorch/functorch.md. Covers:
  1. Forward-mode Jacobian (jacfwd) for output >> input
  2. Reverse-mode Jacobian (jacrev) for input >> output
  3. Hessian via jacfwd(jacrev)

Run:
    python 02_jacobian_hessian.py
"""

import sys

import torch
from torch.func import jacfwd, jacrev


def f(x):
    return torch.sin(x)


def exp_jacobian_modes():
    print("=" * 60)
    print("1. Forward vs Reverse Jacobian")
    print("=" * 60)

    x = torch.randn(3, requires_grad=True)

    J_fwd = jacfwd(f)(x)
    J_rev = jacrev(f)(x)

    print(f"  f(x) = sin(x), x shape={list(x.shape)}")
    print(f"  jacfwd shape: {list(J_fwd.shape)}  (forward-mode)")
    print(f"  jacrev shape: {list(J_rev.shape)}  (reverse-mode)")
    print(f"  Match: {torch.allclose(J_fwd, J_rev)}")
    print()


def exp_model_jacobian():
    print("=" * 60)
    print("2. Jacobian of a small neural network")
    print("=" * 60)

    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 2),
    )

    x = torch.randn(4)

    # Output dim = 2, input dim = 4
    # jacfwd is efficient when output_dim < input_dim -> fwd mode
    # jacrev is efficient when output_dim > input_dim -> rev mode
    # Here output_dim (2) < input_dim (4) -> jacfwd is better

    def f_model(x):
        return model(x)

    t0 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    J_fwd = jacfwd(f_model)(x)
    J_rev = jacrev(f_model)(x)

    print(f"  Model: Linear(4,8) -> Tanh -> Linear(8,2)")
    print(f"  Input dim={4}, Output dim={2}")
    print(f"  jacfwd shape: {list(J_fwd.shape)} (forward, efficient for 2 < 4)")
    print(f"  jacrev shape: {list(J_rev.shape)} (reverse, better when output > input)")
    print()


def exp_hessian():
    print("=" * 60)
    print("3. Hessian via jacfwd(jacrev)")
    print("=" * 60)

    def loss(params):
        return torch.sin(params).sum()

    params = torch.randn(4, requires_grad=True)
    # hessian = d/dx (d loss/dx)^T = jacfwd(jacrev(loss))
    # jacrev(loss) -> gradient (4,)
    # jacfwd(gradient) -> (4, 4) Hessian
    hessian = jacfwd(jacrev(loss))(params)

    print(f"  Hessian shape: {list(hessian.shape)}")
    print(f"  Hessian (first 2x2):")
    print(f"    {hessian[:2, :2].tolist()}")

    # Check symmetry: Hessian should be symmetric
    is_symmetric = torch.allclose(hessian, hessian.t())
    print(f"  Symmetric: {is_symmetric}")
    print()


EXPERIMENTS = {
    "jacobian": exp_jacobian_modes,
    "model": exp_model_jacobian,
    "hessian": exp_hessian,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[functorch case 2] DONE")


if __name__ == "__main__":
    main()
