"""02 - 张量运算：形状、梯度、grad_fn、叶子节点与梯度累加。

运行方式:
    cd mini_autograd
    python examples/02_tensor_operations.py
"""

import numpy as np

import mini_autograd.ops as ops
from mini_autograd import Tensor


def main() -> None:
    print("=" * 60)
    print("示例 02: 张量运算")
    print("=" * 60)

    # --- 逐元素运算 --------------------------------------------------------- #
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    y = Tensor(np.ones((2, 2)), requires_grad=True)
    z = (x * y + x**2).sum()
    z.backward()
    print("\nz = sum(x*y + x^2)")
    print("x.grad =", x.grad)  # = y + 2x = [[3,5],[7,9]]
    print("y.grad =", y.grad)  # = x = [[1,2],[3,4]]

    # --- 矩阵乘法 ------------------------------------------------------------ #
    x = Tensor(np.random.randn(3, 4), requires_grad=True)
    w = Tensor(np.random.randn(4, 2), requires_grad=True)
    y = x @ w
    y.sum().backward()
    print("\nx @ w -> 输出形状", y.shape)
    print("x.grad 形状:", x.grad.shape, " w.grad 形状:", w.grad.shape)

    # --- 多条路径上的梯度累加 ------------------------------------------------- #
    a = Tensor(3.0, requires_grad=True)
    q = a * a + a  # a 被使用两次：dy/da = 2a + 1 = 7
    q.backward()
    print("\ny = a*a + a   (a 出现了两次)")
    print("a.grad =", a.grad, " (期望 7)")

    # --- 非叶子节点与叶子节点的语义 ------------------------------------------- #
    x = Tensor(2.0, requires_grad=True)
    h = x * 2.0
    print("\nx.is_leaf =", x.is_leaf, ", h.is_leaf =", h.is_leaf)
    print("x.grad_fn =", x.grad_fn, ", h.grad_fn =", h.grad_fn)

    # --- 非标量 backward，需要显式传入梯度 ------------------------------------ #
    x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    y = x * x
    y.backward(np.array([0.5, 0.5, 0.5]))
    print("\n非标量 y.backward([.5,.5,.5]) -> x.grad =", x.grad)


if __name__ == "__main__":
    main()
