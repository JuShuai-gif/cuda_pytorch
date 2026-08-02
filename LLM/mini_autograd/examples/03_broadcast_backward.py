"""03 - 广播：为什么反向时梯度要"求和"还原回较小的形状。

运行方式:
    cd mini_autograd
    python examples/03_broadcast_backward.py
"""

import numpy as np

from mini_autograd import Tensor


def main() -> None:
    print("=" * 60)
    print("示例 03: 广播的反向传播")
    print("=" * 60)

    # y = x + b  其中 x:(4,3), b:(3,)
    x = Tensor(np.arange(12.0).reshape(4, 3), requires_grad=True)
    b = Tensor(np.array([10.0, 20.0, 30.0]), requires_grad=True)

    print("\nx.shape =", x.shape)
    print("b.shape =", b.shape)
    print("x + b   ->  形状", (x + b).shape, "(numpy 广播)")

    y = x + b
    y.sum().backward()

    print("\nloss = sum(x + b) = 6 + (10+20+30)*4 = 246")
    print("x.grad.shape =", x.grad.shape, "(与 x 相同)")
    print("b.grad.shape =", b.grad.shape, "-> 还原回 (3,)")

    # 为什么？x 的每一行都加上了同一个 b，所以每个 b[j] 影响了 4 个输出。
    print("b.grad =", b.grad, " (期望 [4, 4, 4])")
    print("\n解释: b[j] 被广播到了 4 行，所以它的梯度是这 4 个贡献的")
    print("求和（每个贡献来自一行）。")

    # --- 标量广播 ------------------------------------------------------------ #
    x = Tensor(np.ones((2, 3)), requires_grad=True)
    s = Tensor(5.0, requires_grad=True)
    (x * s).sum().backward()
    print("\n标量 s: 在 (2,3) 上做 x*s，s.grad =", s.grad, " (期望 6)")

    # --- (1,C,1,1) 在 (N,C,H,W) 上的广播 ------------------------------------- #
    n, c, h, w = 2, 3, 4, 4
    x = Tensor(np.random.randn(n, c, h, w), requires_grad=True)
    g = Tensor(np.random.randn(1, c, 1, 1), requires_grad=True)
    (x * g).sum().backward()
    expected = x.data.sum(axis=(0, 2, 3), keepdims=True)
    print("\n(1,C,1,1) 缩放: g.grad.shape =", g.grad.shape)
    assert np.allclose(g.grad, expected)
    print("g.grad 等于沿 (N,H,W) 轴求和 -> OK")


if __name__ == "__main__":
    main()
