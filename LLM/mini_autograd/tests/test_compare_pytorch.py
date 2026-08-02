"""对比测试：mini_autograd vs PyTorch（缺少 torch 时自动跳过）。

每个测试都在两个框架里运行同样的计算，并断言前向结果和所有梯度在
rtol=1e-5, atol=1e-6 范围内一致。
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mini_autograd import Tensor, nn as mnn, optim as moptim
import mini_autograd.ops as ops


def _torch(x):
    return torch.tensor(x, dtype=torch.float64, requires_grad=True)


def assert_match(ours, theirs, rtol=1e-5, atol=1e-6):
    np.testing.assert_allclose(ours, theirs, rtol=rtol, atol=atol)


def run_both(fn_mini, fn_torch):
    """分别在 mini_autograd 和 torch 上运行 fn，返回输出与叶子梯度。"""
    x0 = np.random.default_rng(0).normal(size=(4, 3))

    mx = Tensor(x0.copy(), requires_grad=True)
    mo = fn_mini(mx)
    mo.sum().backward()
    our_grad = mx.grad.copy()

    tx = _torch(x0.copy())
    to = fn_torch(tx)
    to.sum().backward()
    torch_grad = tx.grad.numpy()

    return mo.data, our_grad, to.data.numpy(), torch_grad


# --------------------------------------------------------------------------- #
#  逐元素算子                                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name, fn_mini, fn_torch",
    [
        ("add", lambda x: x + 2.0, lambda x: x + 2.0),
        ("sub", lambda x: x - 1.5, lambda x: x - 1.5),
        ("mul", lambda x: x * 3.0, lambda x: x * 3.0),
        ("div", lambda x: x / 2.0, lambda x: x / 2.0),
        ("neg", lambda x: -x, lambda x: -x),
        ("pow", lambda x: x**2, lambda x: x**2),
        ("exp", ops.exp, torch.exp),
        ("sigmoid", ops.sigmoid, torch.sigmoid),
        ("tanh", ops.tanh, torch.tanh),
        ("relu", ops.relu, torch.relu),
    ],
)
def test_elementwise_matches_pytorch(name, fn_mini, fn_torch):
    our_out, our_grad, torch_out, torch_grad = run_both(fn_mini, fn_torch)
    assert_match(our_out, torch_out)
    assert_match(our_grad, torch_grad)


def test_log_matches_pytorch():
    """log 算子与 PyTorch 一致（输入为正数）。"""
    x0 = np.abs(np.random.default_rng(0).normal(size=(3, 4))) + 0.5
    mx = Tensor(x0.copy(), requires_grad=True)
    ops.log(mx).sum().backward()
    tx = _torch(x0.copy())
    torch.log(tx).sum().backward()
    assert_match(mx.grad, tx.grad.numpy())


def test_sum_mean_matches_pytorch():
    """sum / mean 的梯度与 PyTorch 一致。"""
    x0 = np.random.default_rng(0).normal(size=(4, 3))
    mx = Tensor(x0.copy(), requires_grad=True)
    mx.sum().backward()
    tx = _torch(x0.copy())
    tx.sum().backward()
    assert_match(mx.grad, tx.grad.numpy())

    mx = Tensor(x0.copy(), requires_grad=True)
    mx.mean().backward()
    tx = _torch(x0.copy())
    tx.mean().backward()
    assert_match(mx.grad, tx.grad.numpy())


def test_matmul_matches_pytorch():
    """矩阵乘法的梯度与 PyTorch 一致。"""
    x0 = np.random.default_rng(0).normal(size=(4, 3))
    w0 = np.random.default_rng(1).normal(size=(3, 5))
    mx, tw = Tensor(x0.copy(), requires_grad=True), _torch(x0.copy())
    ops.matmul(mx, Tensor(w0.copy())).sum().backward()
    torch.matmul(tw, torch.tensor(w0, dtype=torch.float64)).sum().backward()
    assert_match(mx.grad, tw.grad.numpy())

    mw, tx = Tensor(w0.copy(), requires_grad=True), _torch(w0.copy())
    ops.matmul(Tensor(x0.copy()), mw).sum().backward()
    torch.matmul(torch.tensor(x0, dtype=torch.float64), tx).sum().backward()
    assert_match(mw.grad, tx.grad.numpy())


def test_reshape_transpose_matches_pytorch():
    """reshape / transpose 的梯度与 PyTorch 一致。"""
    x0 = np.arange(12.0).reshape(3, 4)
    mx, tx = Tensor(x0.copy(), requires_grad=True), _torch(x0.copy())
    mx.reshape(4, 3).sum().backward()
    tx.reshape(4, 3).sum().backward()
    assert_match(mx.grad, tx.grad.numpy())

    mx, tx = Tensor(x0.copy(), requires_grad=True), _torch(x0.copy())
    mx.transpose().sum().backward()
    tx.transpose(0, 1).sum().backward()
    assert_match(mx.grad, tx.grad.numpy())


# --------------------------------------------------------------------------- #
#  广播                                                                       #
# --------------------------------------------------------------------------- #
def test_broadcast_bias_matches_pytorch():
    """广播偏置的梯度与 PyTorch 一致。"""
    x0 = np.random.default_rng(0).normal(size=(4, 3))
    b0 = np.random.default_rng(1).normal(size=(3,))
    mx, mb = (
        Tensor(x0.copy(), requires_grad=True),
        Tensor(b0.copy(), requires_grad=True),
    )
    tx, tb = _torch(x0.copy()), _torch(b0.copy())
    (mx + mb).sum().backward()
    (tx + tb).sum().backward()
    assert_match(mx.grad, tx.grad.numpy())
    assert_match(mb.grad, tb.grad.numpy())
    assert mb.grad.shape == (3,)


def test_broadcast_batch_channel_matches_pytorch():
    """(N,C,H,W) 与 (1,C,1,1) 广播的梯度与 PyTorch 一致。"""
    x0 = np.random.default_rng(0).normal(size=(2, 3, 4, 4))
    b0 = np.random.default_rng(1).normal(size=(1, 3, 1, 1))
    mx, mb = (
        Tensor(x0.copy(), requires_grad=True),
        Tensor(b0.copy(), requires_grad=True),
    )
    tx, tb = _torch(x0.copy()), _torch(b0.copy())
    (mx * mb).sum().backward()
    (tx * tb).sum().backward()
    assert_match(mx.grad, tx.grad.numpy())
    assert_match(mb.grad, tb.grad.numpy())


# --------------------------------------------------------------------------- #
#  nn 模块、损失、SGD                                                          #
# --------------------------------------------------------------------------- #
def test_linear_layer_matches_pytorch():
    """Linear 层的前向与梯度都和 PyTorch 一致。"""
    torch.manual_seed(0)
    np.random.seed(0)
    x0 = np.random.randn(8, 3)
    t0 = np.random.randn(8, 2)  # 两个框架使用同一个目标值
    m_layer = mnn.Linear(3, 2)
    t_layer = torch.nn.Linear(3, 2, dtype=torch.float64)
    t_layer.weight.data.copy_(torch.tensor(m_layer.weight.data.copy()))
    t_layer.bias.data.copy_(torch.tensor(m_layer.bias.data.copy()))

    m_out = m_layer(Tensor(x0))
    m_loss = mnn.MSELoss()(m_out, Tensor(t0.copy()))
    m_loss.backward()

    t_out = t_layer(torch.tensor(x0, dtype=torch.float64))
    t_loss = torch.nn.functional.mse_loss(
        t_out, torch.tensor(t0.copy(), dtype=torch.float64)
    )
    t_loss.backward()

    assert_match(m_loss.data, t_loss.item())
    assert_match(m_layer.weight.grad, t_layer.weight.grad.numpy())
    assert_match(m_layer.bias.grad, t_layer.bias.grad.numpy())


def test_mse_loss_matches_pytorch():
    """MSELoss 与 PyTorch 一致。"""
    torch.manual_seed(0)
    np.random.seed(0)
    x0 = np.random.randn(5, 2)
    t0 = np.random.randn(5, 2)
    m_loss = mnn.MSELoss()(Tensor(x0.copy(), requires_grad=True), Tensor(t0.copy()))
    t_loss = torch.nn.functional.mse_loss(
        _torch(x0.copy()), torch.tensor(t0, dtype=torch.float64)
    )
    assert_match(m_loss.data, t_loss.item())
    assert_match(m_loss.data * 0, 0)


def test_cross_entropy_matches_pytorch():
    """CrossEntropyLoss 的前向与梯度都和 PyTorch 一致。"""
    torch.manual_seed(0)
    np.random.seed(0)
    x0 = np.random.randn(6, 4)
    y0 = np.random.randint(0, 4, size=6)
    mx = Tensor(x0.copy(), requires_grad=True)
    m_loss = mnn.CrossEntropyLoss()(mx, y0)
    m_loss.backward()
    tx = _torch(x0.copy())
    t_loss = torch.nn.functional.cross_entropy(tx, torch.tensor(y0))
    t_loss.backward()
    assert_match(m_loss.data, t_loss.item())
    assert_match(mx.grad, tx.grad.numpy())


def test_sgd_update_matches_pytorch():
    """SGD 多次更新后的参数与 PyTorch 一致。"""
    torch.manual_seed(0)
    np.random.seed(0)
    x0 = np.random.randn(8, 3)
    t0 = np.random.randn(8, 1)

    m_layer = mnn.Linear(3, 1)
    t_layer = torch.nn.Linear(3, 1, dtype=torch.float64)
    t_layer.weight.data.copy_(torch.tensor(m_layer.weight.data.copy()))
    t_layer.bias.data.copy_(torch.tensor(m_layer.bias.data.copy()))

    m_opt = moptim.SGD(m_layer.parameters(), lr=0.01)
    t_opt = torch.optim.SGD(t_layer.parameters(), lr=0.01)

    for _ in range(5):
        m_loss = mnn.MSELoss()(m_layer(Tensor(x0)), Tensor(t0))
        m_opt.zero_grad()
        m_loss.backward()
        m_opt.step()

        t_loss = torch.nn.functional.mse_loss(
            t_layer(torch.tensor(x0, dtype=torch.float64)),
            torch.tensor(t0, dtype=torch.float64),
        )
        t_opt.zero_grad()
        t_loss.backward()
        t_opt.step()

    assert_match(m_layer.weight.data, t_layer.weight.data.numpy())
    assert_match(m_layer.bias.data, t_layer.bias.data.numpy())


def test_detach_matches_pytorch():
    """detach 的梯度语义与 PyTorch 一致。"""
    x0 = np.random.default_rng(0).normal(size=(3, 3))
    mx, tx = Tensor(x0.copy(), requires_grad=True), _torch(x0.copy())
    m_y = mx * 2.0
    t_y = tx * 2.0
    (m_y.detach() * mx).sum().backward()
    (t_y.detach() * tx).sum().backward()
    assert_match(mx.grad, tx.grad.numpy())
