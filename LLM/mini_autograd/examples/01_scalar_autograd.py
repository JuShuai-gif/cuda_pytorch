"""01 - 标量自动微分：完整走一遍 c = a*b + a 的计算图。

运行方式:
    cd mini_autograd
    python examples/01_scalar_autograd.py
"""

import numpy as np

from mini_autograd import Tensor
from mini_autograd.tensor import _reverse_topological_order


def main() -> None:
    print("=" * 60)
    print("示例 01: 标量自动微分")
    print("=" * 60)

    # --- 构建一个微小的计算图：c = a * b + a ------------------------------ #
    a = Tensor(2.0, requires_grad=True)
    b = Tensor(3.0, requires_grad=True)

    t = a * b  # t 是中间结果，grad_fn = <Mul>
    c = t + a  # c 是输出，      grad_fn = <Add>

    print(f"\na = {a.data}, b = {b.data}")
    print(f"t = a*b = {t.data}   grad_fn = {t.grad_fn}")
    print(f"c = t+a = {c.data}   grad_fn = {c.grad_fn}")

    # --- 在 backward 之前观察计算图 ----------------------------------------- #
    print("\n[前向执行时构建的计算图]")
    order = _reverse_topological_order(c)
    for fn in order:
        names = [f"{i.grad_fn or 'leaf'}" for i in fn.inputs]
        print(f"  {fn}  (输入: {names})")

    print("\n链式法则:  dc/da = dc/dt * dt/da + dc/da 直接 = 1*b + 1 = 4")
    print("           dc/db = dc/dt * dt/db = 1*a = 2")

    # --- 反向传播 ------------------------------------------------------------ #
    c.backward()
    print(f"\na.grad = {a.grad}   (期望 4)")
    print(f"b.grad = {b.grad}   (期望 2)")
    print(f"t.grad（非叶子） = {t.grad}   (期望 1)")

    # --- 用 numpy 做有限差分验证 --------------------------------------------- #
    eps = 1e-6
    fd_a = (
        (Tensor(a.data + eps) * b + Tensor(a.data + eps)).item()
        - (Tensor(a.data - eps) * b + Tensor(a.data - eps)).item()
    ) / (2 * eps)
    fd_b = (
        (a * Tensor(b.data + eps) + a).item() - (a * Tensor(b.data - eps) + a).item()
    ) / (2 * eps)
    print(f"\n有限差分验证: dc/da={fd_a:.6f}, dc/db={fd_b:.6f}")
    assert np.isclose(a.grad, 4.0) and np.isclose(b.grad, 2.0), "梯度计算错误！"
    print("\n所有检查通过。")


if __name__ == "__main__":
    main()
