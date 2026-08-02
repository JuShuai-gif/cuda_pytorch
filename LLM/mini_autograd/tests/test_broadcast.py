"""广播及其在反向中的逆运算（梯度求和）的测试。"""

import numpy as np

from mini_autograd import Tensor
import mini_autograd.ops as ops


def test_scalar_and_matrix():
    """标量与矩阵运算的广播。"""
    x = Tensor(np.ones((3, 2)), requires_grad=True)
    s = Tensor(5.0, requires_grad=True)
    (x + s).sum().backward()
    np.testing.assert_allclose(s.grad, np.array(6.0))
    np.testing.assert_allclose(x.grad, np.ones((3, 2)))


def test_matrix_and_vector_bias():
    """(batch, feature) 与 (feature,) 的广播。"""
    # y = x + b 其中 x:(4,3), b:(3,)
    x = Tensor(np.ones((4, 3)), requires_grad=True)
    b = Tensor(np.ones(3), requires_grad=True)
    y = x + b
    y.sum().backward()
    assert b.grad.shape == (3,)
    np.testing.assert_allclose(b.grad, np.full(3, 4.0))
    np.testing.assert_allclose(x.grad, np.ones((4, 3)))


def test_batch_channel_hw():
    """(N,C,H,W) 与 (1,C,1,1) 的广播：b 的梯度沿 batch 和空间轴求和。"""
    rng = np.random.default_rng(0)
    x = Tensor(rng.normal(size=(2, 3, 4, 4)), requires_grad=True)
    b = Tensor(rng.normal(size=(1, 3, 1, 1)), requires_grad=True)
    (x * b).sum().backward()
    np.testing.assert_allclose(b.grad, x.data.sum(axis=(0, 2, 3), keepdims=True))


def test_multi_step_broadcast():
    """连续多个广播操作串联。"""
    # ((x + b) * c) 其中 b:(3,)，c 为标量
    x = Tensor(np.arange(12.0).reshape(4, 3), requires_grad=True)
    b = Tensor(np.ones(3), requires_grad=True)
    c = Tensor(2.0, requires_grad=True)
    y = (x + b) * c
    y.sum().backward()
    np.testing.assert_allclose(x.grad, np.full((4, 3), 2.0))
    np.testing.assert_allclose(b.grad, np.full(3, 8.0))
    np.testing.assert_allclose(c.grad, (x.data + b.data).sum())


def test_matmul_batch_broadcast_grads():
    """a:(1,3,4)、b:(2,4,5)：a 的梯度必须沿广播的 batch 维求和。"""
    rng = np.random.default_rng(1)
    a0 = rng.normal(size=(1, 3, 4))
    b0 = rng.normal(size=(2, 4, 5))
    a = Tensor(a0.copy(), requires_grad=True)
    b = Tensor(b0.copy(), requires_grad=True)
    ops.matmul(a, b).sum().backward()
    # 每个 batch 的梯度：ones(2,3,5) @ b0^T 再沿 batch 轴求和
    ones = np.ones((2, 3, 5))
    expected = np.einsum("bij,bkj->bik", ones, b0).sum(axis=0, keepdims=True)
    np.testing.assert_allclose(a.grad, expected)
    # b 的梯度不受 a 的广播维影响
    expected_b = np.einsum("bik,bij->bkj", a0, ones)
    np.testing.assert_allclose(b.grad, expected_b)


def test_mul_row_and_column_vectors():
    """行向量与列向量的广播。"""
    # x:(4,3) * v:(1,3) 与 x:(4,3) * w:(4,1)
    x = Tensor(np.ones((4, 3)), requires_grad=True)
    v = Tensor(np.arange(3.0), requires_grad=True)
    (x * v).sum().backward()
    np.testing.assert_allclose(v.grad, np.full(3, 4.0))

    x = Tensor(np.ones((4, 3)), requires_grad=True)
    w = Tensor(np.arange(4.0), requires_grad=True)
    (x * w.reshape(4, 1)).sum().backward()
    np.testing.assert_allclose(w.grad, np.full(4, 3.0))
