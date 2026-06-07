"""
Lab 04: Scaling Laws — 起始代码

完成以下内容:
  - fit_power_law: 拟合 L(N) = a * N^(-alpha) + b
  - plot_scaling_law: 可视化拟合结果
  - iso_flop_curves: 计算并绘制 IsoFLOP 曲线
  - compute_optimal_allocation: 为每个 FLOP budget 找到最优 N, D
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────
# 合成数据（模拟真实的 scaling law 实验）
# ──────────────────────────────────────────────────────────────────────

# 参数量 N (百万) 及对应的 loss
# 格式: (N_params, loss)
KAPLAN_DATA: List[Tuple[float, float]] = [
    (0.001, 7.50),
    (0.003, 6.20),
    (0.01, 5.30),
    (0.03, 4.60),
    (0.10, 4.10),
    (0.30, 3.55),
    (1.0, 3.20),
    (3.0, 2.90),
    (10.0, 2.70),
    (30.0, 2.45),
    (100.0, 2.30),
    (300.0, 2.10),
    (1000.0, 2.00),
]

# 固定模型下 D (十亿) 与对应 loss 的数据
FIXED_MODEL_DATA: List[Tuple[float, float]] = [
    (0.01, 6.00),
    (0.03, 5.00),
    (0.10, 4.20),
    (0.30, 3.70),
    (1.0, 3.30),
    (3.0, 3.00),
    (10.0, 2.75),
    (30.0, 2.55),
    (100.0, 2.42),
    (300.0, 2.33),
]

# 用于 IsoFLOP 曲线的 (N, D, loss) 合成数据
ISOFLOPS_DATA: List[Tuple[float, float, float]] = [
    # 格式: (N_params_in_M, D_tokens_in_B, loss)
    (10, 100, 3.40),
    (30, 100, 3.10),
    (100, 100, 2.90),
    (300, 100, 2.75),
    (10, 300, 3.00),
    (30, 300, 2.75),
    (100, 300, 2.55),
    (300, 300, 2.40),
    (10, 1000, 2.65),
    (30, 1000, 2.42),
    (100, 1000, 2.22),
    (300, 1000, 2.10),
    (10, 3000, 2.40),
    (30, 3000, 2.18),
    (100, 3000, 2.02),
    (300, 3000, 1.95),
]


# ──────────────────────────────────────────────────────────────────────
# 任务 2: 幂律拟合
# ──────────────────────────────────────────────────────────────────────


def power_law(x: np.ndarray, a: float, alpha: float, b: float) -> np.ndarray:
    """L(x) = a * x^(-alpha) + b"""
    return a * np.power(x, -alpha) + b


def fit_power_law(
    data: List[Tuple[float, float]],
    p0: Tuple[float, float, float] = (1.0, 0.1, 1.0),
) -> Tuple[float, float, float]:
    """将 L(N) = a * N^(-alpha) + b 拟合到数据。

    使用 scipy.optimize.curve_fit 或手动最小二乘法。

    Args:
        data: (x, y) 对列表，x 为 N（或 D），y 为 loss。
        p0: (a, alpha, b) 的初始猜测值。

    Returns:
        拟合参数 (a, alpha, b) 元组。
    """
    # TODO: 实现幂律拟合
    raise NotImplementedError("fit_power_law() not implemented")


def plot_scaling_law(
    data: List[Tuple[float, float]],
    a: float,
    alpha: float,
    b: float,
    label: str = "Model",
    save_path: str = "/tmp/scaling_law.png",
) -> None:
    """绘制数据点和拟合曲线。

    X 轴: log 刻度, Y 轴: 线性刻度 loss。
    """
    # TODO: 创建图表
    raise NotImplementedError("plot_scaling_law() not implemented")


# ──────────────────────────────────────────────────────────────────────
# 任务 3: IsoFLOP 曲线
# ──────────────────────────────────────────────────────────────────────


def fit_chinchilla_loss(
    data: List[Tuple[float, float, float]],
) -> Tuple[float, float, float, float, float]:
    """拟合 L(N, D) = E + A * N^(-alpha) + B * D^(-beta)。

    Args:
        data: (N, D, loss) 元组列表。
              N 单位为百万，D 单位为十亿，loss 为标量。

    Returns:
        拟合参数 (E, A, alpha, B, beta)。
    """
    # TODO: 拟合 Chinchilla loss 函数
    raise NotImplementedError("fit_chinchilla_loss() not implemented")


def compute_optimal_allocation(
    E: float,
    A: float,
    alpha: float,
    B: float,
    beta: float,
    flops_budgets: List[float],
) -> List[Tuple[float, float, float]]:
    """为每个 FLOP budget 找到计算最优的 (N, D)。

    在约束 C = 6 * N * D 下，找到使 L(N, D) 最小的 N。

    Args:
        E, A, alpha, B, beta: 拟合的 Chinchilla 参数。
        flops_budgets: FLOP 预算列表（如 [1e18, 1e19, ...]）。

    Returns:
        每个预算下的 (N_opt_in_M, D_opt_in_B, min_loss) 列表。
    """
    # TODO: 计算最优分配
    raise NotImplementedError("compute_optimal_allocation() not implemented")


def plot_isoflops(
    data: List[Tuple[float, float, float]],
    E: float,
    A: float,
    alpha: float,
    B: float,
    beta: float,
    optimal_points: List[Tuple[float, float, float]],
    save_path: str = "/tmp/isoflops.png",
) -> None:
    """绘制 IsoFLOP 等高线和最优点。"""
    # TODO: 创建 IsoFLOP 图表
    raise NotImplementedError("plot_isoflops() not implemented")


# ──────────────────────────────────────────────────────────────────────
# 任务 1: 知识问答
# ──────────────────────────────────────────────────────────────────────


def answer_scaling_law_questions() -> str:
    return """
Q1: Kaplan (2020) 的核心结论是什么？L(N) 和 L(D) 的幂律指数分别是多少？

YOUR ANSWER HERE

Q2: Chinchilla (2022) 如何修正了 Kaplan 的结论？

YOUR ANSWER HERE

Q3: 什么是 "compute-optimal" training？Kaplan 和 Chinchilla 给出的 optimal compute budget 分配有什么不同？

YOUR ANSWER HERE

Q4: 为什么 Chinchilla 的训练 token 数远大于 Kaplan 的预测？

YOUR ANSWER HERE
"""


if __name__ == "__main__":
    print("Lab 04 starter — 拟合 scaling laws 并绘制 IsoFLOP 曲线。")
