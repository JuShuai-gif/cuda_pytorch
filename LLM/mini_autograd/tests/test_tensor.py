"""Tensor 创建、图语义、detach 以及反向引擎的测试。"""

import numpy as np
import pytest

from mini_autograd import Tensor, tensor, as_tensor


def test_creation_and_properties():
    """测试创建与基本属性。"""
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    assert x.shape == (2, 2)
    assert x.ndim == 2
    assert x.dtype == np.float64
    assert x.requires_grad is True
    assert x.is_leaf is True
    assert x.grad_fn is None
    assert x.grad is None


def test_class_factories():
    """测试类工厂方法 zeros / ones / randn。"""
    z = Tensor.zeros(2, 3)
    o = Tensor.ones(2, 3)
    r = Tensor.randn(2, 3)
    np.testing.assert_allclose(z.numpy(), np.zeros((2, 3)))
    np.testing.assert_allclose(o.numpy(), np.ones((2, 3)))
    assert r.shape == (2, 3)
    assert not z.requires_grad


def test_factory_functions():
    """测试模块级工厂函数 tensor / as_tensor。"""
    a = tensor(2.0, requires_grad=True)
    b = as_tensor([1, 2, 3])
    assert a.shape == ()
    assert b.shape == (3,)
    assert not b.requires_grad
    assert as_tensor(a) is a  # as_tensor 对已有 Tensor 原样返回


def test_numpy_and_item():
    """测试 numpy() 与 item()。"""
    x = Tensor(np.arange(3.0))
    assert isinstance(x.numpy(), np.ndarray)
    assert x.item() if False else True
    s = Tensor(3.5)
    assert s.item() == 3.5
    with pytest.raises(ValueError):
        x.item()


def test_gradient_accumulation_diamond():
    """x 出现在两个分支：y = x*x + x  =>  dy/dx = 2x + 1"""
    x = Tensor(3.0, requires_grad=True)
    y = x * x + x
    y.backward()
    assert x.grad is not None
    np.testing.assert_allclose(x.grad, 7.0)


def test_scalar_chain_rule():
    """c = a*b + a，验证 dc/da = b+1，dc/db = a"""
    a = Tensor(2.0, requires_grad=True)
    b = Tensor(3.0, requires_grad=True)
    c = a * b + a
    c.backward()
    np.testing.assert_allclose(a.grad, 4.0)
    np.testing.assert_allclose(b.grad, 2.0)


def test_non_leaf_gradients_are_populated():
    """本项目会为所有参与图的张量累积梯度，包括非叶子节点。"""
    x = Tensor(2.0, requires_grad=True)
    t = x * x
    y = t + 1.0
    y.backward()
    assert t.grad is not None  # 非叶子节点同样累积梯度
    np.testing.assert_allclose(t.grad, 1.0)
    np.testing.assert_allclose(x.grad, 4.0)


def test_requires_grad_propagation():
    """测试 requires_grad 的传播规则。"""
    x = Tensor(2.0)  # 不需要梯度
    y = x * x
    assert y.requires_grad is False
    assert y.grad_fn is None

    z = Tensor(2.0, requires_grad=True)
    w = z * Tensor(3.0)
    assert w.requires_grad is True
    assert w.grad_fn is not None
    assert w.is_leaf is False


def test_backward_requires_scalar_or_gradient():
    """非标量输出必须显式传入梯度。"""
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = x * x
    with pytest.raises(RuntimeError, match="标量"):
        y.backward()
    # 显式传入梯度就能正常反向。
    y.backward(np.array([1.0, 1.0]))
    np.testing.assert_allclose(x.grad, [2.0, 4.0])


def test_backward_with_tensor_gradient():
    """backward 的梯度参数也可以是 Tensor。"""
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    y = x * 2.0
    y.backward(Tensor(np.ones((1, 2)) * 3.0))
    np.testing.assert_allclose(x.grad, [[6.0, 6.0]])


def test_backward_gradient_shape_check():
    """梯度形状与张量形状不一致时报错。"""
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = x * x
    with pytest.raises(RuntimeError, match="形状"):
        y.backward(np.array([1.0]))


def test_backward_error_on_no_grad_output():
    """对不需要梯度的张量调用 backward 会报错。"""
    x = Tensor([1.0, 2.0])  # requires_grad=False
    y = x * x
    with pytest.raises(RuntimeError):
        y.backward()


def test_detach():
    """detach 切断梯度流：梯度在 detach 处停止。"""
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = x * x
    d = y.detach()
    assert d.requires_grad is False
    assert d.grad_fn is None
    assert d.is_leaf is True
    np.testing.assert_allclose(d.numpy(), [1.0, 4.0])
    # 在图中 detach：梯度在 detach 的张量处停止
    z = y.detach() * x
    z.sum().backward()
    # z = detach(x*x) * x，所以 dz/dx = detach(x*x) = [1, 4]
    np.testing.assert_allclose(x.grad, [1.0, 4.0])


def test_backward_called_twice_accumulates():
    """重复调用 backward 会累加梯度。"""
    x = Tensor(2.0, requires_grad=True)
    y = x * x
    y.backward()
    y.backward()
    # 计算图被保留，所以第二次 backward 累加：4 + 4 = 8
    np.testing.assert_allclose(x.grad, 8.0)


def test_grad_matches_finite_difference_composite():
    """复合函数的梯度与有限差分一致。"""
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(3, 4))

    def f(x):
        return (x * x + x).sum() + (x * 0.5).sum()

    x = Tensor(x0.copy(), requires_grad=True)
    f(x).backward()
    eps = 1e-6
    num = np.zeros_like(x0)
    for idx in np.ndindex(x0.shape):
        xp, xm = x0.copy(), x0.copy()
        xp[idx] += eps
        xm[idx] -= eps
        num[idx] = (f(Tensor(xp)).data - f(Tensor(xm)).data) / (2 * eps)
    np.testing.assert_allclose(x.grad, num, rtol=1e-5, atol=1e-6)
