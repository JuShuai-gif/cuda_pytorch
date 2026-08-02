"""单个算子及其反向传播的测试。"""

import numpy as np
import pytest

import mini_autograd.ops as ops
from mini_autograd import Tensor


def finite_diff_grad(f, x0, eps=1e-6):
    """用有限差分计算标量函数 f 对数组 x0 的数值梯度。"""
    x = Tensor(x0.copy(), requires_grad=True)
    f(x).sum().backward()
    grad = x.grad.copy()
    num = np.zeros_like(x0)
    for idx in np.ndindex(x0.shape):
        xp, xm = x0.copy(), x0.copy()
        xp[idx] += eps
        xm[idx] -= eps
        num[idx] = (f(Tensor(xp)).sum().data - f(Tensor(xm)).sum().data) / (2 * eps)
    np.testing.assert_allclose(grad, num, rtol=1e-5, atol=1e-6)


def test_add():
    """加法梯度：dz/da = dz/db = 1"""
    a = Tensor([1.0, 2.0], requires_grad=True)
    b = Tensor([3.0, 4.0], requires_grad=True)
    (a + b).sum().backward()
    np.testing.assert_allclose(a.grad, [1, 1])
    np.testing.assert_allclose(b.grad, [1, 1])


def test_sub():
    """减法梯度：dz/da = 1，dz/db = -1"""
    a = Tensor([5.0, 6.0], requires_grad=True)
    b = Tensor([3.0, 4.0], requires_grad=True)
    (a - b).sum().backward()
    np.testing.assert_allclose(a.grad, [1, 1])
    np.testing.assert_allclose(b.grad, [-1, -1])


def test_mul():
    """乘法梯度：dz/da = b，dz/db = a"""
    a = Tensor([2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0], requires_grad=True)
    (a * b).sum().backward()
    np.testing.assert_allclose(a.grad, [4, 5])
    np.testing.assert_allclose(b.grad, [2, 3])


def test_div():
    """除法梯度：dz/da = 1/b，dz/db = -a/b^2"""
    a = Tensor([8.0, 9.0], requires_grad=True)
    b = Tensor([2.0, 3.0], requires_grad=True)
    (a / b).sum().backward()
    np.testing.assert_allclose(a.grad, [1 / 2, 1 / 3])
    np.testing.assert_allclose(b.grad, [-8 / 4, -9 / 9])


def test_neg():
    """取负梯度：dz/dx = -1"""
    x = Tensor([1.0, -2.0], requires_grad=True)
    (-x).sum().backward()
    np.testing.assert_allclose(x.grad, [-1, -1])


def test_pow():
    """幂梯度：dz/dx = p * x^(p-1)"""
    x = Tensor([2.0, 3.0], requires_grad=True)
    (x**2).sum().backward()
    np.testing.assert_allclose(x.grad, [4.0, 6.0])


def test_reverse_operators():
    """反向运算符：2-x、2/x、2*x"""
    x = Tensor(3.0, requires_grad=True)
    (2.0 - x).backward()
    np.testing.assert_allclose(x.grad, -1.0)

    x = Tensor(3.0, requires_grad=True)
    (2.0 / x).backward()
    np.testing.assert_allclose(x.grad, -2.0 / 9.0)

    x = Tensor(2.0, requires_grad=True)
    (2.0 * x).backward()
    np.testing.assert_allclose(x.grad, 2.0)


def test_matmul_gradient():
    """矩阵乘梯度：dz/da = grad @ b.T，dz/db = a.T @ grad"""
    rng = np.random.default_rng(1)
    a0, b0 = rng.normal(size=(3, 4)), rng.normal(size=(4, 5))
    a = Tensor(a0.copy(), requires_grad=True)
    b = Tensor(b0.copy(), requires_grad=True)
    ops.matmul(a, b).sum().backward()
    np.testing.assert_allclose(a.grad, np.ones((3, 5)) @ b0.T, rtol=1e-8)
    np.testing.assert_allclose(b.grad, a0.T @ np.ones((3, 5)), rtol=1e-8)


def test_matmul_vector_matrix():
    """向量与矩阵相乘的梯度。"""
    rng = np.random.default_rng(2)
    v0 = rng.normal(size=4)
    m0 = rng.normal(size=(4, 3))
    # 向量 @ 矩阵: (4,) @ (4,3) -> (3,)
    v = Tensor(v0.copy(), requires_grad=True)
    m = Tensor(m0.copy(), requires_grad=True)
    ops.matmul(v, m).sum().backward()
    np.testing.assert_allclose(v.grad, np.ones(3) @ m0.T, rtol=1e-8)
    np.testing.assert_allclose(m.grad, np.outer(v0, np.ones(3)), rtol=1e-8)

    # 矩阵 @ 向量: (4,3) @ (3,) -> (4,)
    v2 = rng.normal(size=3)
    v = Tensor(v2.copy(), requires_grad=True)
    m = Tensor(m0.copy(), requires_grad=True)
    ops.matmul(m, v).sum().backward()
    np.testing.assert_allclose(m.grad, np.outer(np.ones(4), v2), rtol=1e-8)
    np.testing.assert_allclose(v.grad, m0.T @ np.ones(4), rtol=1e-8)


def test_matmul_batched_broadcast():
    """批量矩阵乘的广播梯度。"""
    rng = np.random.default_rng(3)
    a0 = rng.normal(size=(1, 3, 4))
    b0 = rng.normal(size=(2, 4, 5))
    a = Tensor(a0.copy(), requires_grad=True)
    b = Tensor(b0.copy(), requires_grad=True)
    ops.matmul(a, b).sum().backward()
    # 对两个输入分别做有限差分验证
    eps = 1e-6
    fa = lambda t: ops.matmul(t, Tensor(b0))
    finite_diff_grad(fa, a0, eps)
    fb = lambda t: ops.matmul(Tensor(a0), t)
    finite_diff_grad(fb, b0, eps)


def test_sum():
    """sum 的梯度：把 1 广播回输入形状。"""
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    x.sum().backward()
    np.testing.assert_allclose(x.grad, np.ones((2, 2)))

    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    x.sum(axis=0).backward(np.ones(2))  # 非标量输出需要传入梯度
    np.testing.assert_allclose(x.grad, np.ones((2, 2)))


def test_mean():
    """mean 的梯度：把 1/count 广播回输入形状。"""
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    x.mean().backward()
    np.testing.assert_allclose(x.grad, np.full((2, 2), 1.0 / 4))

    x = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    x.mean(axis=0).backward()
    np.testing.assert_allclose(x.grad, np.full(4, 0.25))


def test_reshape():
    """reshape 的梯度：把梯度 reshape 回输入形状。"""
    x = Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
    y = x.reshape(3, 2)
    y.sum().backward()
    np.testing.assert_allclose(x.grad, np.ones((2, 3)))


def test_transpose():
    """transpose 的梯度：用逆置换转置梯度。"""
    x = Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
    t = x.transpose()
    t.sum().backward()
    np.testing.assert_allclose(x.grad, np.ones((2, 3)))
    assert t.shape == (3, 2)


def test_exp_log():
    """exp 与 log 的梯度。"""
    x = Tensor([0.5, 1.5], requires_grad=True)
    ops.exp(x).sum().backward()
    np.testing.assert_allclose(x.grad, np.exp([0.5, 1.5]))

    x = Tensor([1.0, 2.0], requires_grad=True)
    ops.log(x).sum().backward()
    np.testing.assert_allclose(x.grad, [1.0, 0.5])


def test_relu():
    """ReLU 的梯度：x>0 时为 1，否则为 0。"""
    x = Tensor([1.0, -2.0, 0.5], requires_grad=True)
    ops.relu(x).sum().backward()
    np.testing.assert_allclose(x.grad, [1.0, 0.0, 1.0])


def test_sigmoid():
    """sigmoid 的梯度：z*(1-z)。"""
    x = Tensor([0.0, 1.0], requires_grad=True)
    ops.sigmoid(x).sum().backward()
    s = 1.0 / (1.0 + np.exp(-np.array([0.0, 1.0])))
    np.testing.assert_allclose(x.grad, s * (1 - s))


def test_tanh():
    """tanh 的梯度：1 - z^2。"""
    x = Tensor([0.0, 1.0], requires_grad=True)
    ops.tanh(x).sum().backward()
    t = np.tanh([0.0, 1.0])
    np.testing.assert_allclose(x.grad, 1 - t**2)


def test_all_elementwise_match_finite_diff():
    """所有逐元素算子的梯度都与有限差分一致。"""
    rng = np.random.default_rng(4)
    x0 = rng.uniform(0.5, 2.0, (2, 3))
    for name, f in [
        ("sigmoid", ops.sigmoid),
        ("tanh", ops.tanh),
        ("exp", ops.exp),
        ("log", ops.log),
    ]:
        finite_diff_grad(lambda t, f=f: f(t), x0)


def test_composite_chain_rule():
    """复合链式法则：z = (x*y + x)^2 => dz/dx = 2*(x*y + x)*(y+1)"""
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)
    z = (x * y + x) ** 2
    z.backward()
    val = 2 * (2 * 3 + 2)
    np.testing.assert_allclose(x.grad, val * (3 + 1))
    np.testing.assert_allclose(y.grad, val * 2)
