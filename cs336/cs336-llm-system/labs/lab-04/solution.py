"""
Lab 04 解答: Scaling Laws

完整的幂律拟合、IsoFLOP 曲线和计算最优分析。
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ──────────────────────────────────────────────────────────────────────
# 合成数据
# ──────────────────────────────────────────────────────────────────────

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

ISOFLOPS_DATA: List[Tuple[float, float, float]] = [
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


# ══════════════════════════════════════════════════════════════════════
# 任务 2: 幂律拟合
# ══════════════════════════════════════════════════════════════════════


def power_law(x: np.ndarray, a: float, alpha: float, b: float) -> np.ndarray:
    return a * np.power(x, -alpha) + b


def fit_power_law(
    data: List[Tuple[float, float]],
    p0: Tuple[float, float, float] = (1.0, 0.1, 1.0),
) -> Tuple[float, float, float]:
    x_vals = np.array([d[0] for d in data])
    y_vals = np.array([d[1] for d in data])

    popt, pcov = curve_fit(power_law, x_vals, y_vals, p0=p0, maxfev=10000)
    a, alpha, b = popt
    return float(a), float(alpha), float(b)


def plot_scaling_law(
    data: List[Tuple[float, float]],
    a: float,
    alpha: float,
    b: float,
    label: str = "Model",
    save_path: str = "/tmp/scaling_law.png",
) -> None:
    x_data = np.array([d[0] for d in data])
    y_data = np.array([d[1] for d in data])

    x_fit = np.logspace(math.log10(x_data.min()), math.log10(x_data.max()), 100)
    y_fit = power_law(x_fit, a, alpha, b)

    # 计算 R²
    y_pred = power_law(x_data, a, alpha, b)
    ss_res = np.sum((y_data - y_pred) ** 2)
    ss_tot = np.sum((y_data - y_data.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x_data, y_data, label="Data", color="blue", zorder=5)
    ax.plot(
        x_fit, y_fit, label=f"Fit: L = {a:.2f}·N^(-{alpha:.3f}) + {b:.3f}", color="red"
    )
    ax.set_xscale("log")
    ax.set_xlabel("Parameters N (millions)")
    ax.set_ylabel("Loss")
    ax.set_title(f"Scaling Law: {label}  (R² = {r2:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {save_path}")


# ══════════════════════════════════════════════════════════════════════
# 任务 3: IsoFLOP 曲线
# ══════════════════════════════════════════════════════════════════════


def chinchilla_loss(
    ND: np.ndarray, E: float, A: float, alpha: float, B: float, beta: float
) -> np.ndarray:
    """L(N, D) = E + A * N^(-alpha) + B * D^(-beta)。

    ND: (n_points, 2) 数组，列为 [N, D]
    """
    N = ND[:, 0]
    D = ND[:, 1]
    return E + A * np.power(N, -alpha) + B * np.power(D, -beta)


def fit_chinchilla_loss(
    data: List[Tuple[float, float, float]],
) -> Tuple[float, float, float, float, float]:
    ND_array = np.array([[d[0], d[1]] for d in data])
    losses = np.array([d[2] for d in data])

    # 初始猜测
    # E ~ 最小 loss, A, B ~ 范围, alpha, beta ~ 0.3
    p0 = (1.5, 5.0, 0.3, 5.0, 0.25)
    bounds = ([0, 0, 0.01, 0, 0.01], [10, 100, 2.0, 100, 2.0])

    popt, _ = curve_fit(
        chinchilla_loss, ND_array, losses, p0=p0, bounds=bounds, maxfev=10000
    )
    E, A, alpha, B, beta = popt
    return float(E), float(A), float(alpha), float(B), float(beta)


def compute_optimal_allocation(
    E: float,
    A: float,
    alpha: float,
    B: float,
    beta: float,
    flops_budgets: List[float],
) -> List[Tuple[float, float, float]]:
    """在 N 上进行网格搜索，为每个 FLOP budget 找到最优 (N, D)。

    约束: C = 6 * N * D（N 以参数个数计，非百万；D 以 token 数计，非十亿）。
    注意: 我们的数据中 N 以百万计，D 以十亿计，所以 FLOPs = 6 * N*1e6 * D*1e9
    """
    results = []
    for C in flops_budgets:
        best_loss = float("inf")
        best_N_m = 0.0
        best_D_b = 0.0

        # 在 0.01M 到 1000M 之间搜索 N（log 刻度）
        for N_m in np.logspace(-1, 4, 200):
            # 由 C = 6 * N * D, D = C / (6 * N)
            D_tokens = C / (6 * N_m * 1e6)
            D_b = D_tokens / 1e9

            if D_b < 0.001 or D_b > 1e5:
                continue

            loss = E + A * (N_m ** (-alpha)) + B * (D_b ** (-beta))
            if loss < best_loss:
                best_loss = loss
                best_N_m = N_m
                best_D_b = D_b

        results.append((best_N_m, best_D_b, best_loss))

    return results


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
    # 创建网格
    N_grid = np.logspace(0.5, 3.5, 80)  # 3M 到 3000M
    D_grid = np.logspace(0.5, 4.5, 80)  # 3B 到 30000B

    NN, DD = np.meshgrid(N_grid, D_grid)
    Z = E + A * NN ** (-alpha) + B * DD ** (-beta)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左: IsoFLOP 等高线
    ax = axes[0]
    levels = np.linspace(Z.min(), Z.max(), 12)
    cs = ax.contour(NN, DD, Z, levels=levels, cmap="viridis")
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")

    # 绘制数据点
    N_data = [d[0] for d in data]
    D_data = [d[1] for d in data]
    ax.scatter(N_data, D_data, c="red", s=20, zorder=5, alpha=0.6)

    # 绘制最优点
    N_opt = [p[0] for p in optimal_points]
    D_opt = [p[1] for p in optimal_points]
    ax.plot(N_opt, D_opt, "r--o", markersize=8, label="Optimal (N, D)", zorder=10)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Parameters N (M)")
    ax.set_ylabel("Tokens D (B)")
    ax.set_title("IsoFLOP Curves: Loss Contours")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 右: 最优 N vs FLOPs
    ax2 = axes[1]
    flops_values = [6 * p[0] * 1e6 * p[1] * 1e9 for p in optimal_points]
    N_values = [p[0] for p in optimal_points]
    D_values = [p[1] for p in optimal_points]

    ax2.loglog(flops_values, N_values, "b-o", label="N_opt (params)")
    ax2.loglog(flops_values, D_values, "g-s", label="D_opt (tokens)")
    ax2.set_xlabel("FLOPs")
    ax2.set_ylabel("Optimal N / D")
    ax2.set_title("Optimal Allocation: N_opt, D_opt vs FLOPs")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {save_path}")


# ══════════════════════════════════════════════════════════════════════
# 任务 1: 答案
# ══════════════════════════════════════════════════════════════════════


def answer_scaling_law_questions() -> str:
    return """
Q1: Kaplan (2020) 的核心结论是什么？L(N) 和 L(D) 的幂律指数分别是多少？
────────────────────────────────────────────────────────────────────────
Answer:
Kaplan 的核心结论是：语言模型的 loss 随参数量 N、数据量 D 和计算量 C
以幂律形式下降。具体来说：

  L(N) ∝ N^(-0.076)  — 参数量每增加 10x，loss 下降约 0.016
  L(D) ∝ D^(-0.095)  — 数据量每增加 10x，loss 下降约 0.020
  L(C) ∝ C^(-0.050)  — 计算量每增加 10x，loss 下降约 0.011

Kaplan 建议在给定 compute budget 下：
  N_opt ∝ C^0.73  (增大模型)
  D_opt ∝ C^0.27  (数据增长较慢)

Q2: Chinchilla (2022) 如何修正了 Kaplan 的结论？
─────────────────────────────────────────────────
Answer:
Chinchilla 通过更严谨的实验设计修正了 Kaplan 的三个关键问题：

1. 方法论改进：在固定 FLOP budget 下系统性扫描 (N, D) 组合，保证了 fair comparison
2. 使用统一的 cosine LR schedule：Kaplan 中不同模型使用了不同的 LR schedule
3. 不同的结论：

   Chinchilla: N_opt ∝ C^0.50, D_opt ∝ C^0.50
   (参数和数据应该等比例增长)

   vs Kaplan: N_opt ∝ C^0.73, D_opt ∝ C^0.27
   (应更偏向增大参数)

Q3: 什么是 "compute-optimal" training？
    Kaplan 和 Chinchilla 给出的 optimal compute budget 分配有什么不同？
────────────────────────────────────────────────────────────────────────
Answer:
"Compute-optimal" training 是指在给定 compute budget C 下，
选择 (N, D) 使得最终 loss L(N, D) 最小化。

形式化表达：
  min_{N, D} L(N, D)  s.t.  C = 6ND

分配差异：
  - Kaplan:   对于 10^21 FLOPs, 建议 N≈10B, D≈17B tokens
  - Chinchilla: 对于 10^21 FLOPs, 建议 N≈1.5B, D≈100B tokens

Chinchilla 的结论是：大多数大模型都训练不足 (undertrained)，
应当使用更多的训练数据。

Q4: 为什么 Chinchilla 的训练 token 数远大于 Kaplan 的预测？
─────────────────────────────────────────────────────────────
Answer:
根本原因是实验设计差异：

1. Kaplan 的实验设定中，大模型训练的 token 数本来就比小模型多。
   因此参数增长看起来比数据增长更有效——但实际上可能是因为
   大模型"恰巧"训练了更多数据。

2. Kaplan 用学习率 schedule 决定停止时机，而不是统一训练到收敛。
   这导致不同模型的训练充分程度不同。

3. Chinchilla 的实验控制了这些变量：在固定 FLOPs 下 scan (N,D)，
   每个组合都用完整的 cosine schedule 训练。

4. 结果表明，之前被认为"最有效"的那种偏向于增大参数的策略，
   实际上只是因为没有给模型足够的数据。
"""
    return answers


# ══════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Lab 04 解答验证 ===\n")

    # --- Kaplan 风格拟合 ---
    a, alpha, b = fit_power_law(KAPLAN_DATA, p0=(5.0, 0.076, 1.5))
    print(f"Kaplan fit: L(N) = {a:.2f} * N^(-{alpha:.3f}) + {b:.3f}")

    a2, alpha2, b2 = fit_power_law(FIXED_MODEL_DATA, p0=(5.0, 0.095, 1.5))
    print(f"Data fit:   L(D) = {a2:.2f} * D^(-{alpha2:.3f}) + {b2:.3f}")

    plot_scaling_law(KAPLAN_DATA, a, alpha, b, label="Kaplan L(N)")

    # --- Chinchilla 拟合 ---
    E, A_c, alpha_c, B_c, beta_c = fit_chinchilla_loss(ISOFLOPS_DATA)
    print(
        f"\nChinchilla fit: L(N,D) = {E:.3f} + {A_c:.2f}·N^(-{alpha_c:.3f}) + {B_c:.2f}·D^(-{beta_c:.3f})"
    )

    # 计算最优分配
    flops_budgets = [1e18, 1e19, 1e20, 1e21, 1e22]
    optimal = compute_optimal_allocation(E, A_c, alpha_c, B_c, beta_c, flops_budgets)
    print("\nCompute-Optimal Allocation:")
    for i, (C, (N_opt, D_opt, loss)) in enumerate(zip(flops_budgets, optimal)):
        print(
            f"  C = {C:.0e}: N_opt = {N_opt:.1f}M, D_opt = {D_opt:.1f}B, Loss = {loss:.3f}"
        )

    plot_isoflops(ISOFLOPS_DATA, E, A_c, alpha_c, B_c, beta_c, optimal)

    # --- 知识问答 ---
    print("\n" + "=" * 60)
    print(answer_scaling_law_questions())
