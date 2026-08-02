"""no_grad、enable_grad、set_grad_enabled 以及 detach 的测试。"""

import numpy as np

from mini_autograd import Tensor, no_grad, enable_grad, set_grad_enabled


def test_no_grad_blocks_graph():
    """no_grad 内部不构建计算图。"""
    x = Tensor(2.0, requires_grad=True)
    with no_grad():
        y = x * x
    assert y.requires_grad is False
    assert y.grad_fn is None
    assert y.is_leaf is True


def test_no_grad_is_nested():
    """no_grad 支持嵌套。"""
    x = Tensor(2.0, requires_grad=True)
    with no_grad():
        y = x * x
        with no_grad():
            z = y + 1.0
        w = z * 2.0
    assert z.requires_grad is False
    assert w.requires_grad is False


def test_grad_reenabled_after_no_grad():
    """离开 no_grad 后梯度重新开启。"""
    x = Tensor(2.0, requires_grad=True)
    with no_grad():
        pass
    y = x * x
    assert y.requires_grad is True
    y.backward()
    np.testing.assert_allclose(x.grad, 4.0)


def test_enable_grad_inside_no_grad():
    """在 no_grad 内部用 enable_grad 重新开启。"""
    x = Tensor(2.0, requires_grad=True)
    with no_grad():
        with enable_grad():
            y = x * x
        z = x * x
    assert y.requires_grad is True
    assert z.requires_grad is False


def test_set_grad_enabled():
    """set_grad_enabled 全局开关。"""
    x = Tensor(2.0, requires_grad=True)
    set_grad_enabled(False)
    try:
        y = x * x
        assert y.requires_grad is False
    finally:
        set_grad_enabled(True)
    y2 = x * x
    assert y2.requires_grad is True


def test_parameter_creation_in_no_grad():
    """在 no_grad 内显式创建的 Parameter 仍然需要梯度。"""
    # 显式的 requires_grad=True 在 no_grad 内部依然生效。
    with no_grad():
        from mini_autograd.nn import Parameter

        p = Parameter(np.array([1.0, 2.0]))
    assert p.requires_grad is True
    # 但在 no_grad 内对它的运算不构建计算图。
    with no_grad():
        q = p * 2.0
    assert q.requires_grad is False
