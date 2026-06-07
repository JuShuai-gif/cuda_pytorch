"""
第04讲 — 训练：学习率调度器。

实现：
  - cosine schedule（带可选的 warmup）
  - 线性 warmup
  - WSD (Warmup-Stable-Decay) schedule
"""

from __future__ import annotations

import math
from typing import List, Optional


# ---------------------------------------------------------------------------
# Cosine schedule
# ---------------------------------------------------------------------------


def cosine_schedule(
    step: int,
    total_steps: int,
    lr_max: float,
    lr_min: float = 0.0,
    warmup_steps: int = 0,
) -> float:
    """Cosine decay 学习率调度。

    当 warmup_steps > 0 时，学习率在前 ``warmup_steps`` 步内从 0 线性增长到
    lr_max，然后按 cosine decay 衰减到 lr_min。

    参数
    ----------
    step : int
        当前训练步数（从 0 开始编号）。
    total_steps : int
        训练总步数。
    lr_max : float
        峰值学习率。
    lr_min : float
        最小 / 最终学习率。
    warmup_steps : int
        warmup 步数。

    返回
    -------
    lr : float
    """
    if warmup_steps > 0 and step < warmup_steps:
        # 线性 warmup
        return lr_max * (step + 1) / max(warmup_steps, 1)

    # Cosine decay
    decay_steps = total_steps - warmup_steps
    t = max(step - warmup_steps, 0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * t / max(decay_steps, 1)))
    return lr_min + (lr_max - lr_min) * cosine


# ---------------------------------------------------------------------------
# 线性 warmup（独立版本）
# ---------------------------------------------------------------------------


def linear_warmup(step: int, warmup_steps: int, lr_max: float) -> float:
    """在 warmup_steps 内将学习率从 0 线性增长到 lr_max。"""
    if step >= warmup_steps:
        return lr_max
    return lr_max * (step + 1) / max(warmup_steps, 1)


# ---------------------------------------------------------------------------
# WSD schedule（Warmup – Stable – Decay）
# ---------------------------------------------------------------------------


def wsd_schedule(
    step: int,
    total_steps: int,
    lr_max: float,
    lr_min: float = 0.0,
    warmup_steps: int = 0,
    stable_ratio: float = 0.9,
) -> float:
    """Warmup-Stable-Decay (WSD) 学习率调度。

    - 前 ``warmup_steps`` 步：线性 warmup，从 0 → lr_max。
    - 接下来 ``stable_ratio * total_steps`` 步：保持在 lr_max。
    - 剩余步数：cosine decay，从 lr_max → lr_min。
    """
    # Warmup 阶段
    if step < warmup_steps:
        return lr_max * (step + 1) / max(warmup_steps, 1)

    stable_end = int(stable_ratio * total_steps)

    # Stable 阶段
    if step < stable_end:
        return lr_max

    # Decay 阶段
    decay_total = total_steps - stable_end
    t_decay = step - stable_end
    cosine = 0.5 * (1.0 + math.cos(math.pi * t_decay / max(decay_total, 1)))
    return lr_min + (lr_max - lr_min) * cosine


# ---------------------------------------------------------------------------
# 生成 schedule 轨迹
# ---------------------------------------------------------------------------


def generate_lr_trace(
    total_steps: int,
    lr_max: float = 1e-3,
    lr_min: float = 1e-5,
    warmup_steps: int = 100,
    schedule_fn=cosine_schedule,
) -> List[float]:
    """返回步数 0 .. total_steps-1 对应的学习率列表。"""
    return [
        schedule_fn(s, total_steps, lr_max, lr_min, warmup_steps)
        for s in range(total_steps)
    ]


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    total = 1000
    warmup = 200
    lr_max = 1e-3
    lr_min = 1e-5

    print("Cosine schedule (first 5 and last 5 steps):")
    for s in [0, 1, 199, 200, 500, 995, 996, 997, 998, 999]:
        lr = cosine_schedule(s, total, lr_max, lr_min, warmup)
        print(f"  step {s:4d}: lr={lr:.6e}")

    print(f"\nWSD schedule (stable_ratio=0.8):")
    stable_ratio = 0.8
    for s in [0, 1, 199, 200, 500, 795, 800, 950, 999]:
        lr = wsd_schedule(s, total, lr_max, lr_min, warmup, stable_ratio)
        print(f"  step {s:4d}: lr={lr:.6e}")

    # 验证性质
    assert (
        cosine_schedule(0, total, lr_max, lr_min, warmup_steps=0) == lr_max
    )  # 无 warmup，从最大值开始
    final = cosine_schedule(total - 1, total, lr_max, lr_min, warmup_steps=0)
    assert abs(final - lr_min) < 1e-8, f"Final LR {final} != {lr_min}"

    # warmup 单调递增
    prev = -1.0
    for s in range(warmup):
        cur = cosine_schedule(s, total, lr_max, lr_min, warmup)
        assert cur > prev, f"Warmup not monotonic at step {s}"
        prev = cur

    print("\nAll checks passed.")
