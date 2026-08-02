"""04 - 线性回归：用单个 Linear 层 + SGD 拟合 y = 3x + 2。

运行方式:
    cd mini_autograd
    python examples/04_linear_regression.py
"""

import numpy as np

from mini_autograd import Tensor, nn, optim


def main() -> None:
    np.random.seed(0)
    print("=" * 60)
    print("示例 04: 线性回归  y = 3x + 2")
    print("=" * 60)

    # --- 数据 ---------------------------------------------------------------- #
    x0 = np.random.uniform(-2.0, 2.0, (128, 1))
    y_true = 3.0 * x0 + 2.0

    # --- 模型 ---------------------------------------------------------------- #
    model = nn.Linear(1, 1)  # 一个权重 + 一个偏置
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    print(
        f"\n初始: w = {model.weight.data.item():.4f}, b = {model.bias.data.item():.4f}"
    )

    # --- 训练循环 ------------------------------------------------------------- #
    for step in range(3000):
        pred = model(Tensor(x0))
        loss = loss_fn(pred, Tensor(y_true))

        optimizer.zero_grad()  # 清空旧的梯度
        loss.backward()  # 执行反向自动微分
        optimizer.step()  # w -= lr * dw, b -= lr * db

        if step % 500 == 0:
            print(f"step {step:4d}: loss = {loss.item():.6f}")

    # --- 汇报结果 ------------------------------------------------------------- #
    w_learned = model.weight.data.item()
    b_learned = model.bias.data.item()
    print(f"\n学习结果: w = {w_learned:.4f} (目标 3.0), b = {b_learned:.4f} (目标 2.0)")
    print(f"误差:  dw = {abs(w_learned - 3.0):.6f}, db = {abs(b_learned - 2.0):.6f}")
    assert abs(w_learned - 3.0) < 1e-2 and abs(b_learned - 2.0) < 1e-2
    print("已收敛。")


if __name__ == "__main__":
    main()
