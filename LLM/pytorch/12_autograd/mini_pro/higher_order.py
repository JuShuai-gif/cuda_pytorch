"""
Autograd Higher-Order: 二阶导、Hessian、grad of grad
======================================================
演示 PyTorch 的 create_graph=True 和 double backward

运行: python 07_higher_order.py
"""

import sys

import torch


def demo_second_order():
    """二阶导数: d^2y/dx^2"""
    print("=" * 60)
    print("1. 二阶导数: create_graph=True")
    print("=" * 60)

    x = torch.tensor(2.0, requires_grad=True)
    y = x**3     # y = x^3

    # 一阶: dy/dx = 3x^2 = 12
    grad1, = torch.autograd.grad(y, x, create_graph=True)
    print(f"  dy/dx = {grad1.item()}  (期望: 3*2^2 = 12)")

    # 二阶: d^2y/dx^2 = 6x = 12
    grad2, = torch.autograd.grad(grad1, x)
    print(f"  d^2y/dx^2 = {grad2.item()}  (期望: 6*2 = 12)")

    # 没有 create_graph=True 会报错
    x2 = torch.tensor(2.0, requires_grad=True)
    y2 = x2**3
    grad1_no_graph, = torch.autograd.grad(y2, x2, create_graph=False)
    try:
        torch.autograd.grad(grad1_no_graph, x2)
    except RuntimeError as e:
        print(f"\n  无 create_graph 的二阶失败: {str(e)[:60]}")


def demo_hessian_vector():
    """Hessian-vector product: v^T H v"""
    print("\n" + "=" * 60)
    print("2. Hessian-vector product")
    print("=" * 60)

    def f(x):
        return torch.sin(x).sum()

    x = torch.randn(4, requires_grad=True)
    v = torch.randn(4)

    # HVP = d/depsilon grad(f)(x + epsilon*v) | epsilon=0
    grad_f = torch.autograd.grad(f(x), x, create_graph=True)[0]
    hvp = torch.autograd.grad(grad_f, x, grad_outputs=v)[0]

    print(f"  f(x) = sin(x).sum()")
    print(f"  grad: {grad_f.tolist()}")
    print(f"  HVP (shape={list(hvp.shape)}): {hvp.tolist()}")


def demo_grad_penalty():
    """实战: WGAN-GP 中的梯度惩罚 (二阶导)"""
    print("\n" + "=" * 60)
    print("3. 实战: WGAN-GP gradient penalty")
    print("=" * 60)

    # WGAN-GP: E[(||grad_x D(x_hat)||_2 - 1)^2]
    # 需要对判别器输出的梯度再求导
    disc = torch.nn.Linear(4, 1)
    x_real = torch.randn(8, 4)
    x_fake = torch.randn(8, 4)

    # 插值
    eps = torch.rand(8, 1)
    x_hat = eps * x_real + (1 - eps) * x_fake
    x_hat.requires_grad_(True)

    # 判别器输出
    d_hat = disc(x_hat)

    # 梯度
    grads = torch.autograd.grad(
        outputs=d_hat, inputs=x_hat,
        grad_outputs=torch.ones_like(d_hat),
        create_graph=True, retain_graph=True
    )[0]

    grad_penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    print(f"  Gradient penalty: {grad_penalty.item():.4f}")
    print(f"  -> 需要 create_graph=True 来计算梯度的梯度")


EXPERIMENTS = {
    "second": demo_second_order,
    "hvp": demo_hessian_vector,
    "wgan": demo_grad_penalty,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}'")
            continue
        EXPERIMENTS[name]()
    print("[autograd higher-order] DONE")


if __name__ == "__main__":
    main()
