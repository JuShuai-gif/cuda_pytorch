"""05 - 两层 MLP 在两个高斯团数据上的分类。

运行方式:
    cd mini_autograd
    python examples/05_mlp_classification.py
"""

import numpy as np

from mini_autograd import Tensor, nn, optim


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def make_data(n_per_class: int = 100, seed: int = 0):
    rng = np.random.RandomState(seed)
    c0 = rng.randn(n_per_class, 2) + np.array([1.5, 1.5])
    c1 = rng.randn(n_per_class, 2) + np.array([-1.5, -1.5])
    x = np.vstack([c0, c1]).astype(np.float64)
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return x, y


def main() -> None:
    np.random.seed(0)
    print("=" * 60)
    print("示例 05: 两层 MLP 分类")
    print("=" * 60)

    x0, y0 = make_data()
    model = MLP()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    print(f"\n数据集: {x0.shape[0]} 个样本，2 个类别")
    print(f"模型:   {model}")

    for step in range(1000):
        logits = model(Tensor(x0))  # 前向传播
        loss = loss_fn(logits, y0)  # 交叉熵损失

        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 梯度下降

        if step % 200 == 0:
            acc = (logits.data.argmax(axis=1) == y0).mean()
            print(f"step {step:4d}: loss = {loss.item():.4f}, acc = {acc:.3f}")

    logits = model(Tensor(x0)).data
    acc = (logits.argmax(axis=1) == y0).mean()
    print(f"\n训练集上的最终准确率: {acc:.3f}")
    assert acc > 0.95
    print("完成。")


if __name__ == "__main__":
    main()
