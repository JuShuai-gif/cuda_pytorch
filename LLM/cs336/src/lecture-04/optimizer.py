"""
第04讲 — 训练：从零实现 AdamW 优化器。

实现带解耦 weight decay、bias correction 和可选 AMSGrad 的 AdamW 算法。
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch.optim import Optimizer


# ---------------------------------------------------------------------------
# AdamW 优化器
# ---------------------------------------------------------------------------


class AdamW(Optimizer):
    """带解耦 weight decay 的 AdamW 优化器。

    实现了 *Decoupled Weight Decay Regularization*
    （Loshchilov & Hutter, 2019）中描述的算法。

    参数
    ----------
    params : 参数或参数组的可迭代对象。
    lr : 学习率。
    betas : (β₁, β₂) 用于滑动平均的系数。
    eps : 为数值稳定性添加的小量。
    weight_decay : 解耦 weight decay 系数。
    amsgrad : 是否使用 AMSGrad 变体。
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")

        defaults: Dict[str, Any] = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """执行单步优化。"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            amsgrad = group["amsgrad"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]

                # 状态初始化（惰性）
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                exp_avg: torch.Tensor = state["exp_avg"]
                exp_avg_sq: torch.Tensor = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                # 解耦 weight decay
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # 更新有偏动量
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # bias correction
                bias1 = 1.0 - beta1**t
                bias2 = 1.0 - beta2**t
                step_size = lr / bias1

                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().div_(math.sqrt(bias2)).add_(eps)
                else:
                    denom = exp_avg_sq.sqrt().div_(math.sqrt(bias2)).add_(eps)

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # 用线性回归问题进行简单测试
    w = torch.nn.Parameter(torch.randn(10, 1, requires_grad=True))
    b = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
    X = torch.randn(100, 10)
    y = X[:, :1] * 3 + 2 + torch.randn(100, 1) * 0.1

    optimizer = AdamW([w, b], lr=0.01, weight_decay=0.01)

    for step in range(10):
        optimizer.zero_grad()
        pred = X @ w + b
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        optimizer.step()

    print(f"Final loss: {loss.item():.6f}")
    print(f"True w≈3.0, fitted: {w[0, 0].item():.4f}")
    print(f"True b≈2.0, fitted: {b.item():.4f}")

    # 验证状态键
    assert "exp_avg" in optimizer.state[w], "Missing exp_avg"
    assert "exp_avg_sq" in optimizer.state[w], "Missing exp_avg_sq"
    print("AdamW state initialised correctly  ✓")
    print("\nAll checks passed.")
