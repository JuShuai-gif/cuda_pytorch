"""
scaling_visualization.py — 缩放定律可视化

创建出版级别的图表:
  1. L(N), L(D), L(C) — 带有数据点的 Kaplan 独立拟合。
  2. Chinchilla loss 曲面 — (N, D) 空间中的等高线/热力图。
  3. IsoFLOP 曲线 — 固定 compute budget 下的 loss vs N。
  4. Compute-optimal N/D 比率趋势。

图表保存为当前目录下的 PNG 文件。
用法:
    python scaling_visualization.py
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import matplotlib

matplotlib.use("Agg")  # 非交互式后端，用于无头环境
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation, ScalarFormatter
from typing import Tuple

# 从同级模块导入工具函数
from scaling_law_fit import (
    generate_kaplan_data,
    generate_chinchilla_grid,
    generate_isflop_data,
    fit_kaplan,
    fit_chinchilla,
    find_isflop_optimal,
    kaplan_fn,
    chinchilla_fn,
    GT_KAPLAN,
    GT_CHINCHILLA,
)
from compute_optimal import (
    compute_optimal_analytical,
    compute_optimal_numerical,
    compute_optimal_ratio,
    CHINCHILLA_PARAMS,
)


# ---------------------------------------------------------------------------
# 全局 matplotlib 样式
# ---------------------------------------------------------------------------


def set_style() -> None:
    """应用简洁、出版级别的 matplotlib 样式。"""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 9,
            "figure.figsize": (8, 5.5),
            "lines.linewidth": 1.8,
            "lines.markersize": 3.5,
        }
    )


# ---------------------------------------------------------------------------
# 图 1: Kaplan 独立拟合  L(N), L(D), L(C)
# ---------------------------------------------------------------------------


def plot_kaplan_fits(save_path: str = "kaplan_fits.png") -> None:
    """绘制三条独立 Kaplan 幂律的拟合曲线与数据点。"""
    (N_data, L_N), (D_data, L_D), (C_data, L_C) = generate_kaplan_data()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle("Kaplan-Style Isolated Scaling Laws", fontsize=14, y=1.02)

    configs = [
        (axes[0], N_data, L_N, "N", "Model Parameters  N"),
        (axes[1], D_data, L_D, "D", "Dataset Size  D  (tokens)"),
        (axes[2], C_data, L_C, "C", "Compute  C  (PF-days)"),
    ]

    for ax, x, y, prefix, label in configs:
        # 拟合
        x_c_hat, a_hat, L_inf_hat, _ = fit_kaplan(x, y, prefix)

        # 绘制数据与拟合曲线
        ax.scatter(x, y, s=6, alpha=0.5, color="steelblue", label="Synthetic data")
        x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 300)
        y_fit = kaplan_fn(x_fit, x_c_hat, a_hat, L_inf_hat)
        ax.plot(
            x_fit,
            y_fit,
            color="crimson",
            linewidth=2,
            label=(
                rf"Fit: $L_\infty$={L_inf_hat:.3f}, "
                rf"$\alpha$={a_hat:.4f}"
            ),
        )

        # 真实值（ground truth）
        gt = GT_KAPLAN[prefix]
        gt_xc = gt[f"{prefix}_c"]
        gt_a = gt["alpha"]
        gt_Linf = gt["L_inf"]
        y_true = kaplan_fn(x_fit, gt_xc, gt_a, gt_Linf)
        ax.plot(x_fit, y_true, "k--", linewidth=1, alpha=0.7, label="Ground truth")

        ax.set_xscale("log")
        ax.set_xlabel(label)
        ax.set_ylabel("Loss  L")
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"[Saved] {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图 2: Chinchilla loss 曲面 — 等高线 / 热力图
# ---------------------------------------------------------------------------


def plot_chinchilla_surface(save_path: str = "chinchilla_surface.png") -> None:
    """Chinchilla loss 在 (N, D) 空间中的 2D 等高线 + 热力图。"""
    Nch, Dch, Lch = generate_chinchilla_grid(noise_std=0.0)  # 曲面不需要噪声
    N_vals = np.unique(Nch)
    D_vals = np.unique(Dch)
    L_grid = Lch.reshape(len(D_vals), len(N_vals))  # D rows, N cols

    fig, ax = plt.subplots(figsize=(8, 6))

    # 热力图
    levels = np.linspace(L_grid.min(), L_grid.max(), 30)
    cf = ax.contourf(N_vals, D_vals, L_grid, levels=levels, cmap="YlOrRd")
    cbar = fig.colorbar(cf, ax=ax, label="Loss  L(N, D)")

    # 叠加等高线
    ax.contour(
        N_vals, D_vals, L_grid, levels=15, colors="black", linewidths=0.4, alpha=0.6
    )

    # Compute-optimal 线（解析解）
    p = GT_CHINCHILLA
    C_range = np.logspace(14, 21, 200)
    N_opts = []
    D_opts = []
    for C in C_range:
        No, Do, _ = compute_optimal_analytical(
            C,
            p["E"],
            p["A"],
            p["alpha"],
            p["B"],
            p["beta"],
        )
        N_opts.append(No)
        D_opts.append(Do)
    ax.plot(N_opts, D_opts, "b-", linewidth=2, label="Compute-optimal (C = 6ND)")
    ax.scatter(N_opts[0], D_opts[0], color="blue", s=40, zorder=5)
    ax.scatter(N_opts[-1], D_opts[-1], color="blue", s=40, zorder=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Model Parameters  N")
    ax.set_ylabel("Dataset Size  D  (tokens)")
    ax.set_title("Chinchilla Loss Surface  L(N, D) = E + A/N^{a} + B/D^{b}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, which="both")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"[Saved] {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图 3: IsoFLOP 曲线
# ---------------------------------------------------------------------------


def plot_isflop_curves(save_path: str = "isflop_curves.png") -> None:
    """对于多个固定 compute budget，绘制 loss vs N 的曲线（IsoFLOP 曲线）。"""
    compute_budgets = np.logspace(15, 20, 6)  # 6 个 budget
    iso_data = generate_isflop_data(compute_budgets)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(compute_budgets)))

    p = GT_CHINCHILLA
    for C, color in zip(compute_budgets, colors):
        N_vals, L_vals = iso_data[float(C)]
        ax.plot(N_vals, L_vals, color=color, linewidth=1.8, label=f"C = {C:.1e}")

        # 标记最优值
        N_opt, _, L_opt = find_isflop_optimal(
            C,
            p["E"],
            p["A"],
            p["alpha"],
            p["B"],
            p["beta"],
        )
        ax.scatter(
            N_opt,
            L_opt,
            color=color,
            s=50,
            zorder=5,
            edgecolors="white",
            linewidths=0.8,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Model Parameters  N")
    ax.set_ylabel("Loss  L(N, D=C/(6N))")
    ax.set_title("IsoFLOP Curves: Loss vs Model Size for Fixed Compute Budgets")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"[Saved] {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图 4: Compute-optimal N/D 比率趋势
# ---------------------------------------------------------------------------


def plot_optimal_ratio(save_path: str = "optimal_ratio.png") -> None:
    """绘制 compute-optimal N/D 比率随 compute budget 变化的趋势。"""
    C_vals = np.logspace(14, 23, 150)

    N_opt_vals = np.empty_like(C_vals)
    D_opt_vals = np.empty_like(C_vals)
    ratios = np.empty_like(C_vals)

    for i, C in enumerate(C_vals):
        Na, Da, _ = compute_optimal_analytical(C)
        N_opt_vals[i] = Na
        D_opt_vals[i] = Da
        ratios[i] = Na / Da

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 左: 比率 vs compute
    ax = axes[0]
    ax.plot(C_vals, ratios, color="darkgreen", linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("Compute Budget  C  (FLOPs)")
    ax.set_ylabel("Optimal N / D ratio")
    ax.set_title("Compute-Optimal N/D Ratio vs Compute Budget")
    ax.grid(True, alpha=0.3, which="both")

    # 为 70B 量级标注 Chinchilla-optimal 比率
    C_70B = 6.0 * 70e9 * 1.4e12  # 粗略的 70B 模型，在 1.4T tokens 上训练
    ratio_70B = compute_optimal_ratio(C_70B)
    ax.axvline(C_70B, color="gray", linestyle="--", alpha=0.5)
    ax.annotate(
        f"C(70B) → ratio={ratio_70B:.3f}",
        xy=(C_70B, ratio_70B),
        fontsize=8,
        color="gray",
        ha="right",
    )

    # 右: N_opt 和 D_opt vs C
    ax = axes[1]
    ax.plot(
        C_vals, N_opt_vals, color="steelblue", linewidth=2, label=r"$N_{opt}$ (params)"
    )
    ax.plot(
        C_vals, D_opt_vals, color="crimson", linewidth=2, label=r"$D_{opt}$ (tokens)"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute Budget  C  (FLOPs)")
    ax.set_ylabel("Optimal Size (count)")
    ax.set_title("Optimal N and D vs Compute Budget")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")

    fig.suptitle("Compute-Optimal Allocation Analysis", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"[Saved] {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图 5: Chinchilla 拟合残差
# ---------------------------------------------------------------------------


def plot_chinchilla_fit(save_path: str = "chinchilla_fit.png") -> None:
    """展示数据 vs 拟合的 Chinchilla 模型，以及残差。"""
    Nch, Dch, Lch = generate_chinchilla_grid(noise_std=0.005)
    E_hat, A_hat, a_hat, B_hat, b_hat, _ = fit_chinchilla(Nch, Dch, Lch)
    L_pred = chinchilla_fn((Nch, Dch), E_hat, A_hat, a_hat, B_hat, b_hat)
    residuals = Lch - L_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # 左: 预测值 vs 实际值散点图
    ax = axes[0]
    ax.scatter(L_pred, Lch, s=10, alpha=0.5, color="steelblue")
    lims = [min(L_pred.min(), Lch.min()), max(L_pred.max(), Lch.max())]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Predicted Loss")
    ax.set_ylabel("Actual Loss")
    ax.set_title("Chinchilla Fit: Predicted vs Actual")
    ax.grid(True, alpha=0.3)

    # 右: 残差直方图
    ax = axes[1]
    ax.hist(residuals, bins=40, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual  (Actual - Predicted)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual Distribution (std = {residuals.std():.4f})")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Chinchilla Fit: E={E_hat:.3f}, A={A_hat:.1f}, "
        f"a={a_hat:.3f}, B={B_hat:.1f}, b={b_hat:.3f}",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"[Saved] {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------


def main() -> None:
    set_style()
    print("Generating scaling law visualizations ...\n")

    plot_kaplan_fits("kaplan_fits.png")
    plot_chinchilla_surface("chinchilla_surface.png")
    plot_isflop_curves("isflop_curves.png")
    plot_optimal_ratio("optimal_ratio.png")
    plot_chinchilla_fit("chinchilla_fit.png")

    print("\nAll plots saved.")


if __name__ == "__main__":
    main()
