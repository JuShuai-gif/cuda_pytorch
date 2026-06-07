"""
从零实现的 AdamW 优化器。

AdamW 将 weight decay 与基于梯度的更新解耦，这与原始 Adam 不同，
原始 Adam 中 weight decay 是通过 L2 regularization 实现的。
更新规则如下：

    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    m_hat = m_t / (1 - beta1^t)
    v_hat = v_t / (1 - beta2^t)
    theta_t = theta_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * theta_{t-1})
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    """
    带有解耦 weight decay 的 AdamW 优化器。

    参数:
        params: 待优化的参数的可迭代对象。
        lr: 学习率。
        betas: 用于计算 gradient 及其平方的指数移动平均的系数。
        eps: 为提高数值稳定性而添加的项。
        weight_decay: weight decay 系数（与 gradient 解耦）。
        amsgrad: 是否使用 AMSGrad 变体。
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults: dict = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
        }
        super().__init__(params, defaults)  # type: ignore[arg-type]

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """
        执行单步优化。

        参数:
            closure: 可选闭包，用于重新评估模型并返回 loss。

        返回:
            如果提供了 closure，则返回 loss 值；否则返回 None。
        """
        loss: Optional[float] = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            beta1, beta2 = group["betas"]
            eps: float = group["eps"]
            weight_decay: float = group["weight_decay"]
            amsgrad: bool = group.get("amsgrad", False)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad: torch.Tensor = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                # 状态初始化
                state: dict = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    # 梯度值的 exponential moving average
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # 梯度平方值的 exponential moving average
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if amsgrad:
                        # 维护所有 exp_avg_sq 值中的最大值
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                exp_avg: torch.Tensor = state["exp_avg"]
                exp_avg_sq: torch.Tensor = state["exp_avg_sq"]
                state["step"] += 1
                step: int = state["step"]

                # 偏差校正
                bias_correction1: float = 1.0 - beta1**step
                bias_correction2: float = 1.0 - beta2**step

                # 解耦的 weight decay（直接应用于参数）
                if weight_decay != 0:
                    p.data.mul_(1.0 - lr * weight_decay)

                # 更新有偏的一阶和二阶矩估计
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # 计算步长
                if amsgrad:
                    max_exp_avg_sq: torch.Tensor = state["max_exp_avg_sq"]
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom: torch.Tensor = (
                        max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
                    ).add_(eps)
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                step_size: float = lr / bias_correction1

                # 应用更新
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


# 快速测试
if __name__ == "__main__":
    # 使用简单线性回归进行测试
    torch.manual_seed(42)

    # 创建一个简单模型
    model = torch.nn.Linear(10, 1)
    optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=0.01)

    # 生成一些数据
    X = torch.randn(32, 10)
    y = torch.randn(32, 1)

    initial_loss: float | None = None
    for i in range(100):

        def closure() -> torch.Tensor:
            pred = model(X)
            loss = torch.nn.functional.mse_loss(pred, y)
            return loss

        optimizer.zero_grad()
        pred = model(X)
        loss = torch.nn.functional.mse_loss(pred, y)
        loss.backward()
        opt_loss = optimizer.step(closure)

        if i == 0:
            initial_loss = loss.item()

    final_loss: float = torch.nn.functional.mse_loss(model(X), y).item()
    assert final_loss < initial_loss, "Loss should decrease during training"
    print(f"AdamW test passed! Loss: {initial_loss:.4f} -> {final_loss:.4f}")
