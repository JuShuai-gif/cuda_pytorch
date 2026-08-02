"""06 - 多个算子在 mini_autograd 与 PyTorch 之间逐项对比。

运行方式:
    cd mini_autograd
    python examples/06_compare_with_pytorch.py
"""

import numpy as np
import torch

import mini_autograd.ops as ops
from mini_autograd import Tensor


def compare(name, our, theirs):
    np.testing.assert_allclose(our, theirs, rtol=1e-5, atol=1e-6)
    print(f"[通过] {name}")


def main() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(4, 3))
    w0 = rng.normal(size=(3, 2))
    b0 = rng.normal(size=(3,))

    print("=" * 60)
    print("示例 06: mini_autograd vs PyTorch")
    print("=" * 60)

    # --- 矩阵乘法 ------------------------------------------------------------ #
    mx = Tensor(x0.copy(), requires_grad=True)
    ops.matmul(mx, Tensor(w0.copy())).sum().backward()
    tx = torch.tensor(x0.copy(), dtype=torch.float64, requires_grad=True)
    torch.matmul(tx, torch.tensor(w0, dtype=torch.float64)).sum().backward()
    compare("matmul 梯度", mx.grad, tx.grad.numpy())

    # --- 广播 ---------------------------------------------------------------- #
    mx = Tensor(x0.copy(), requires_grad=True)
    mb = Tensor(b0.copy(), requires_grad=True)
    (mx + mb).sum().backward()
    tx = torch.tensor(x0.copy(), dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(b0.copy(), dtype=torch.float64, requires_grad=True)
    (tx + tb).sum().backward()
    compare("广播梯度 (x)", mx.grad, tx.grad.numpy())
    compare("广播梯度 (b)", mb.grad, tb.grad.numpy())

    # --- relu / sigmoid / tanh ----------------------------------------------- #
    for name, fn_m, fn_t in [
        ("relu", ops.relu, torch.relu),
        ("sigmoid", ops.sigmoid, torch.sigmoid),
        ("tanh", ops.tanh, torch.tanh),
    ]:
        mx = Tensor(x0.copy(), requires_grad=True)
        fn_m(mx).sum().backward()
        tx = torch.tensor(x0.copy(), dtype=torch.float64, requires_grad=True)
        fn_t(tx).sum().backward()
        compare(f"{name} 梯度", mx.grad, tx.grad.numpy())

    # --- 多次使用同一变量时的梯度累加 ------------------------------------------ #
    ma = Tensor(3.0, requires_grad=True)
    (ma * ma + ma).backward()
    ta = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
    (ta * ta + ta).backward()
    compare("累加梯度 (2a+1)", ma.grad, ta.grad.numpy())

    print("\n所有对比通过。")


if __name__ == "__main__":
    main()
