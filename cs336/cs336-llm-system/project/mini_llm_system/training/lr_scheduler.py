"""
带余弦衰减和线性预热的 learning rate 调度器。

调度包含两个阶段：
1. 预热 (warmup)：LR 在 warmup_steps 内从 0 线性增加到 peak_lr。
2. 余弦衰减 (cosine decay)：剩余步数内 LR 沿余弦曲线从 peak_lr 衰减到 min_lr。
"""

from __future__ import annotations

import math
from typing import Optional

import torch


class CosineWarmupScheduler:
    """
    带线性预热的余弦 learning rate 调度器。

    Args:
        optimizer: PyTorch optimizer。
        warmup_steps: 线性预热的步数。
        total_steps: 总训练步数。
        min_lr_ratio: 最小 LR，以 peak LR 的分数表示（默认：0.1 = 10%）。
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
    ) -> None:
        self.optimizer: torch.optim.Optimizer = optimizer
        self.warmup_steps: int = warmup_steps
        self.total_steps: int = total_steps
        self.min_lr_ratio: float = min_lr_ratio
        self.current_step: int = 0

        # 存储每个参数组的基准（峰值）learning rate
        self.base_lrs: list[float] = [group["lr"] for group in optimizer.param_groups]

    def get_lr(self, step: int) -> float:
        """
        计算指定步数的 learning rate。

        Args:
            step: 当前训练步数（从 0 开始）。

        Returns:
            learning rate 乘数（需要与每个参数组的 base_lr 相乘）。
        """
        if step < self.warmup_steps:
            # 线性预热：0 -> 1.0
            return float(step) / float(max(1, self.warmup_steps))
        else:
            # 余弦衰减：1.0 -> min_lr_ratio
            progress: float = float(step - self.warmup_steps) / float(
                max(1, self.total_steps - self.warmup_steps)
            )
            cosine_decay: float = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay

    def step(self) -> float:
        """
        将调度器前进一步，并更新 optimizer 中各参数组的 LR。

        Returns:
            当前 learning rate 乘数。
        """
        lr_multiplier: float = self.get_lr(self.current_step)
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group["lr"] = base_lr * lr_multiplier
        self.current_step += 1
        return lr_multiplier

    def state_dict(self) -> dict:
        """返回调度器状态，用于 checkpoint 保存。"""
        return {
            "current_step": self.current_step,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """从 checkpoint 加载调度器状态。"""
        self.current_step = state_dict["current_step"]
        self.base_lrs = state_dict["base_lrs"]


# 快速测试
if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")  # 非交互式后端

    # 创建一个虚拟 optimizer
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

    total_steps: int = 1000
    warmup_steps: int = 100

    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=0.1,
    )

    lrs: list[float] = []
    for step in range(total_steps):
        lr = scheduler.get_lr(step)
        lrs.append(lr)

    # 验证预热阶段从 0 到 1
    assert abs(lrs[0]) < 0.01, f"First LR should be ~0, got {lrs[0]}"
    assert abs(lrs[warmup_steps - 1] - 1.0) < 0.02, (
        f"LR after warmup should be ~1.0, got {lrs[warmup_steps - 1]}"
    )

    # 验证最终 LR 等于 min_lr_ratio
    assert abs(lrs[-1] - 0.1) < 0.02, f"Final LR should be ~0.1, got {lrs[-1]}"

    print("CosineWarmupScheduler test passed!")
    print(f"  LR[0]: {lrs[0]:.4f}")
    print(f"  LR[{warmup_steps - 1}]: {lrs[warmup_steps - 1]:.4f}")
    print(f"  LR[{total_steps // 2}]: {lrs[total_steps // 2]:.4f}")
    print(f"  LR[{total_steps - 1}]: {lrs[-1]:.4f}")
