"""激活函数、损失函数、Linear、Module 和 SGD 的测试。"""

import numpy as np
import pytest

import mini_autograd.ops as ops
from mini_autograd import Tensor, nn, optim, no_grad


# --------------------------------------------------------------------------- #
#  激活函数                                                                   #
# --------------------------------------------------------------------------- #
def test_activation_modules_forward():
    """激活函数模块的前向结果。"""
    x = Tensor([0.0, 1.0, -1.0])
    np.testing.assert_allclose(nn.ReLU()(x).numpy(), [0.0, 1.0, 0.0])
    np.testing.assert_allclose(nn.Tanh()(x).numpy(), np.tanh(x.numpy()))
    np.testing.assert_allclose(
        nn.Sigmoid()(x).numpy(), 1.0 / (1.0 + np.exp(-x.numpy()))
    )


# --------------------------------------------------------------------------- #
#  损失函数                                                                   #
# --------------------------------------------------------------------------- #
def test_mse_loss_mean_and_sum():
    """MSELoss 的 mean 与 sum 两种归约方式。"""
    pred = Tensor([[2.0], [4.0]])
    target = Tensor([[1.0], [3.0]])
    mse = nn.MSELoss()
    np.testing.assert_allclose(mse(pred, target).numpy(), 1.0)
    np.testing.assert_allclose(nn.MSELoss(reduction="sum")(pred, target).numpy(), 2.0)


def test_mse_loss_backward():
    """MSELoss 的梯度。"""
    pred = Tensor([2.0, 4.0], requires_grad=True)
    target = Tensor([1.0, 3.0])
    loss = nn.MSELoss()(pred, target)
    loss.backward()
    # d/dpred = 2*(pred-target)/N = [2*1/2, 2*1/2]
    np.testing.assert_allclose(pred.grad, [1.0, 1.0])


def test_cross_entropy_loss_forward():
    """CrossEntropyLoss 的前向结果（与手写稳定 softmax 一致）。"""
    logits = Tensor(np.array([[1.0, 2.0, 0.5], [0.5, 0.1, 2.0]]))
    targets = np.array([1, 2])
    loss = nn.CrossEntropyLoss()(logits, targets)
    # 手写的稳定 softmax
    exp = np.exp(logits.numpy() - logits.numpy().max(axis=1, keepdims=True))
    p = exp / exp.sum(axis=1, keepdims=True)
    expected = -np.mean(np.log(p[np.arange(2), targets]))
    np.testing.assert_allclose(loss.numpy(), expected)


def test_cross_entropy_loss_backward():
    """CrossEntropyLoss 的梯度：d/dlogits = (p - y)/N，N=1"""
    logits = Tensor(np.array([[1.0, 2.0, 0.5]]), requires_grad=True)
    targets = np.array([1])
    loss = nn.CrossEntropyLoss()(logits, targets)
    loss.backward()
    exp = np.exp(logits.numpy() - logits.numpy().max(axis=1, keepdims=True))
    softmax = exp / exp.sum(axis=1, keepdims=True)
    onehot = np.eye(3)[[1]]
    expected = softmax - onehot
    np.testing.assert_allclose(logits.grad, expected, atol=1e-8)


def test_cross_entropy_loss_backward_batched():
    """批量情况下：d/dlogits = (softmax - onehot) / N"""
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(6, 4))
    y0 = np.random.randint(0, 4, size=6)
    logits = Tensor(x0.copy(), requires_grad=True)
    loss = nn.CrossEntropyLoss()(logits, y0)
    loss.backward()
    exp = np.exp(x0 - x0.max(axis=1, keepdims=True))
    softmax = exp / exp.sum(axis=1, keepdims=True)
    onehot = np.eye(4)[y0]
    expected = (softmax - onehot) / 6
    np.testing.assert_allclose(logits.grad, expected, atol=1e-8)


# --------------------------------------------------------------------------- #
#  Linear                                                                     #
# --------------------------------------------------------------------------- #
def test_linear_forward_shape():
    """Linear 前向的输出形状。"""
    layer = nn.Linear(3, 5)
    x = Tensor(np.zeros((4, 3)))
    assert layer(x).shape == (4, 5)


def test_linear_no_bias():
    """不带 bias 的 Linear。"""
    layer = nn.Linear(3, 5, bias=False)
    assert layer.bias is None
    x = Tensor(np.ones((2, 3)))
    assert layer(x).shape == (2, 5)


def test_linear_backward_updates_params():
    """Linear 的权重和偏置都能收到梯度。"""
    rng = np.random.default_rng(0)
    layer = nn.Linear(3, 2)
    x = Tensor(rng.normal(size=(8, 3)))
    target = Tensor(rng.normal(size=(8, 2)))
    loss = nn.MSELoss()(layer(x), target)
    loss.backward()
    assert (
        layer.weight.grad is not None and layer.weight.grad.shape == layer.weight.shape
    )
    assert layer.bias.grad is not None and layer.bias.grad.shape == layer.bias.shape


def test_linear_grad_matches_pytorch_formula():
    """(N,2) 输出上 MSE 的 dloss/dW = 2*(pred-t)^T @ x / (N*2)"""
    rng = np.random.default_rng(1)
    layer = nn.Linear(3, 2)
    x0 = rng.normal(size=(8, 3))
    t0 = rng.normal(size=(8, 2))
    pred0 = x0 @ layer.weight.data.T + layer.bias.data
    expected_w = (pred0 - t0).T @ x0 / (8 * 2) * 2
    expected_b = (pred0 - t0).sum(axis=0) / (8 * 2) * 2

    x = Tensor(x0)
    loss = nn.MSELoss()(layer(x), Tensor(t0))
    loss.backward()
    np.testing.assert_allclose(layer.weight.grad, expected_w, rtol=1e-8)
    np.testing.assert_allclose(layer.bias.grad, expected_b, rtol=1e-8)


# --------------------------------------------------------------------------- #
#  Module                                                                     #
# --------------------------------------------------------------------------- #
class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def test_module_parameters_registration():
    """Module 能递归收集所有参数。"""
    model = _MLP()
    params = list(model.parameters())
    assert len(params) == 4  # fc1.w, fc1.b, fc2.w, fc2.b
    assert all(p.requires_grad for p in params)


def test_module_named_parameters():
    """named_parameters 使用 模块.路径 命名。"""
    model = _MLP()
    names = [n for n, _ in model.named_parameters()]
    assert names == ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]


def test_module_zero_grad():
    """zero_grad 清空所有参数梯度。"""
    model = _MLP()
    loss = model(Tensor(np.zeros((3, 2)))).sum()
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())
    model.zero_grad()
    assert all(p.grad is None for p in model.parameters())


def test_module_train_eval_flag():
    """train/eval 递归切换 training 标志。"""
    model = _MLP()
    assert model.training is True
    model.eval()
    assert model.training is False
    assert model.fc1.training is False
    model.train()
    assert model.training is True


def test_module_call_dispatches_to_forward():
    """module(x) 转发到 forward(x)。"""
    model = _MLP()
    x = Tensor(np.zeros((2, 2)))
    assert model(x).shape == (2, 1)


# --------------------------------------------------------------------------- #
#  SGD                                                                        #
# --------------------------------------------------------------------------- #
def test_sgd_steps_update_parameters():
    """SGD.step 会更新参数。"""
    rng = np.random.default_rng(0)
    layer = nn.Linear(3, 1)
    before_w = layer.weight.data.copy()
    x = Tensor(rng.normal(size=(4, 3)))
    loss = nn.MSELoss()(layer(x), Tensor(np.ones((4, 1))))
    opt = optim.SGD(layer.parameters(), lr=0.01)
    loss.backward()
    opt.step()
    assert not np.allclose(layer.weight.data, before_w)


def test_sgd_linear_regression_converges():
    """SGD 训练线性回归能收敛到真实参数。"""
    np.random.seed(0)
    model = nn.Linear(1, 1)
    x0 = np.random.uniform(-1, 1, (64, 1))
    y0 = 3.0 * x0 + 2.0
    opt = optim.SGD(model.parameters(), lr=0.05)
    for _ in range(3000):
        loss = nn.MSELoss()(model(Tensor(x0)), Tensor(y0))
        opt.zero_grad()
        loss.backward()
        opt.step()
    np.testing.assert_allclose(model.weight.data.ravel(), [3.0], atol=1e-2)
    np.testing.assert_allclose(model.bias.data.ravel(), [2.0], atol=1e-2)


def test_sgd_momentum_and_weight_decay_run():
    """带 momentum 和 weight_decay 的 SGD 至少能正常执行。"""
    rng = np.random.default_rng(0)
    layer = nn.Linear(3, 1)
    opt = optim.SGD(layer.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    x = Tensor(rng.normal(size=(4, 3)))
    loss = nn.MSELoss()(layer(x), Tensor(np.ones((4, 1))))
    loss.backward()
    opt.step()  # 不应抛异常
    assert layer.weight.data is not None


def test_mlp_classification_trains():
    """小 MLP 在两类数据上训练后准确率达标。"""
    np.random.seed(0)

    def make_data(n=100):
        r = np.random.RandomState(0)
        c1 = r.randn(n // 2, 2) + np.array([1.5, 1.5])
        c2 = r.randn(n // 2, 2) + np.array([-1.5, -1.5])
        x = np.vstack([c1, c2])
        y = np.array([0] * (n // 2) + [1] * (n // 2))
        return x, y

    x0, y0 = make_data()

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2, 4)
            self.act = nn.ReLU()
            self.fc2 = nn.Linear(4, 2)

        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))

    net = Net()
    opt = optim.SGD(net.parameters(), lr=0.1)
    ce = nn.CrossEntropyLoss()
    for _ in range(500):
        logits = net(Tensor(x0))
        loss = ce(logits, y0)
        opt.zero_grad()
        loss.backward()
        opt.step()
    logits = net(Tensor(x0)).numpy()
    acc = (logits.argmax(axis=1) == y0).mean()
    assert acc > 0.9
