"""
scaling_law_fit.py — 缩放定律拟合 (Kaplan 与 Chinchilla)

生成合成缩放定律数据，并使用 scipy.optimize.curve_fit 拟合
Kaplan（独立）和 Chinchilla（联合）幂律公式。

用法:
    python scaling_law_fit.py
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from typing import Tuple, Dict, Any


# ---------------------------------------------------------------------------
# 真实参数（大致参考文献）
# ---------------------------------------------------------------------------

# Kaplan 风格独立幂律:  L(x) = (x_c / x)^alpha + L_inf
GT_KAPLAN: Dict[str, Dict[str, float]] = {
    "N": {"N_c": 8.8e13, "alpha": 0.076, "L_inf": 1.69},
    "D": {"D_c": 5.4e13, "alpha": 0.095, "L_inf": 1.69},
    "C": {"C_c": 3.1e8, "alpha": 0.050, "L_inf": 1.69},
}

# Chinchilla 联合幂律:  L(N,D) = E + A/N^alpha + B/D^beta
GT_CHINCHILLA: Dict[str, float] = {
    "E": 1.69,
    "A": 406.4,
    "alpha": 0.34,
    "B": 410.7,
    "beta": 0.28,
}


# ---------------------------------------------------------------------------
# 合成数据生成
# ---------------------------------------------------------------------------


def generate_kaplan_data(
    seed: int = 42,
    noise_std: float = 0.005,
) -> Tuple[
    Tuple[NDArray[np.float64], NDArray[np.float64]],
    Tuple[NDArray[np.float64], NDArray[np.float64]],
    Tuple[NDArray[np.float64], NDArray[np.float64]],
]:
    """生成三个独立缩放定律的合成数据。

    返回三个 (x, L(x)) 元组，分别对应 N、D 和 C。
    """
    rng = np.random.default_rng(seed)

    def _make(
        prefix: str, xs: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        p = GT_KAPLAN[prefix]
        x_c = p[f"{prefix}_c"]
        alpha = p["alpha"]
        L_inf = p["L_inf"]
        L_true = (x_c / xs) ** alpha + L_inf
        L_noisy = L_true + rng.normal(0, noise_std, size=xs.shape)
        return xs, L_noisy

    N_vals = np.logspace(6, 10, 80)  # 1M to 10B parameters
    D_vals = np.logspace(6, 10, 80)  # 1M to 10B tokens
    C_vals = np.logspace(15, 21, 80)  # PF-days range

    return _make("N", N_vals), _make("D", D_vals), _make("C", C_vals)


def generate_chinchilla_grid(
    seed: int = 42,
    noise_std: float = 0.005,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """生成 (N, D) 对的二维网格及对应的带噪声 loss。

    返回 (N_flat, D_flat, L_noisy)，每个都是一维数组。
    """
    rng = np.random.default_rng(seed)
    N_vals = np.logspace(6, 9, 25)  # 1M to 1B parameters
    D_vals = np.logspace(6, 9, 25)  # 1M to 1B tokens
    NN, DD = np.meshgrid(N_vals, D_vals)
    N_flat = NN.ravel()
    D_flat = DD.ravel()

    p = GT_CHINCHILLA
    L_true = p["E"] + p["A"] / (N_flat ** p["alpha"]) + p["B"] / (D_flat ** p["beta"])
    L_noisy = L_true + rng.normal(0, noise_std, size=L_true.shape)

    return N_flat, D_flat, L_noisy


def generate_isflop_data(
    compute_budgets: NDArray[np.float64] | None = None,
) -> Dict[float, Tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """对于每个 compute budget C，沿 6ND = C 曲线生成 loss。

    返回字典，映射 C -> (N_vals, L_vals)。
    """
    if compute_budgets is None:
        compute_budgets = np.logspace(16, 20, 5)  # 1e16 to 1e20 FLOPs

    p = GT_CHINCHILLA
    result: Dict[float, Tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    for C in compute_budgets:
        N_vals = np.logspace(5, 9, 200)  # sweep N
        D_vals = C / (6.0 * N_vals)  # from C ≈ 6ND
        L_vals = (
            p["E"] + p["A"] / (N_vals ** p["alpha"]) + p["B"] / (D_vals ** p["beta"])
        )
        result[float(C)] = (N_vals.copy(), L_vals)
    return result


# ---------------------------------------------------------------------------
# Kaplan 风格拟合（独立幂律）
# ---------------------------------------------------------------------------


def kaplan_fn(
    x: NDArray[np.float64],
    x_c: float,
    alpha: float,
    L_inf: float,
) -> NDArray[np.float64]:
    """Kaplan 风格幂律: (x_c / x)^{alpha} + L_inf。"""
    return (x_c / x) ** alpha + L_inf


def fit_kaplan(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    label: str = "",
) -> Tuple[float, float, float, NDArray[np.float64]]:
    """拟合三参数 Kaplan 定律，返回 (x_c, alpha, L_inf, pcov)。"""
    p0 = (np.mean(x), 0.05, min(y))
    popt, pcov = curve_fit(kaplan_fn, x, y, p0=p0, maxfev=10000)
    x_c_hat, alpha_hat, L_inf_hat = popt
    return x_c_hat, alpha_hat, L_inf_hat, np.diag(pcov)


# ---------------------------------------------------------------------------
# Chinchilla 风格拟合（联合幂律）
# ---------------------------------------------------------------------------


def chinchilla_fn(
    ND: Tuple[NDArray[np.float64], NDArray[np.float64]],
    E: float,
    A: float,
    alpha: float,
    B: float,
    beta: float,
) -> NDArray[np.float64]:
    """Chinchilla 联合幂律: E + A/N^{alpha} + B/D^{beta}。"""
    N, D = ND
    return E + A / (N**alpha) + B / (D**beta)


def fit_chinchilla(
    N_flat: NDArray[np.float64],
    D_flat: NDArray[np.float64],
    L_flat: NDArray[np.float64],
) -> Tuple[float, float, float, float, float, NDArray[np.float64]]:
    """拟合五参数 Chinchilla 定律；返回 (E, A, alpha, B, beta, pcov)。"""
    p0 = (1.5, 300.0, 0.3, 300.0, 0.3)
    bounds = ([0, 0, 0.01, 0, 0.01], [10, 1e5, 1.0, 1e5, 1.0])
    popt, pcov = curve_fit(
        chinchilla_fn,
        (N_flat, D_flat),
        L_flat,
        p0=p0,
        bounds=bounds,
        maxfev=50000,
    )
    E_hat, A_hat, alpha_hat, B_hat, beta_hat = popt
    return E_hat, A_hat, alpha_hat, B_hat, beta_hat, np.diag(pcov)


# ---------------------------------------------------------------------------
# IsoFLOP 曲线分析
# ---------------------------------------------------------------------------


def find_isflop_optimal(
    C: float,
    E: float,
    A: float,
    alpha: float,
    B: float,
    beta: float,
    n_points: int = 500,
) -> Tuple[float, float, float]:
    """对于给定的 compute budget C，通过数值方法找到最优 (N_opt, D_opt, L_opt)。

    使用对 N 的细粒度网格搜索；D 由 C = 6ND 确定。
    """
    N_candidates = np.logspace(4, 10, n_points)
    D_candidates = C / (6.0 * N_candidates)
    L_candidates = E + A / (N_candidates**alpha) + B / (D_candidates**beta)
    best_idx = int(np.argmin(L_candidates))
    return (
        float(N_candidates[best_idx]),
        float(D_candidates[best_idx]),
        float(L_candidates[best_idx]),
    )


# ---------------------------------------------------------------------------
# 主演示
# ---------------------------------------------------------------------------


def main() -> None:
    # --- 生成数据 ---
    (N_data, L_N), (D_data, L_D), (C_data, L_C) = generate_kaplan_data()
    Nch, Dch, Lch = generate_chinchilla_grid()

    # --- 拟合 Kaplan ---
    print("=" * 70)
    print("Kaplan-style isolated power law fits")
    print("=" * 70)

    for prefix, (x, y) in zip(
        ["N", "D", "C"], [(N_data, L_N), (D_data, L_D), (C_data, L_C)]
    ):
        x_c_hat, a_hat, L_inf_hat, _ = fit_kaplan(x, y, prefix)
        gt = GT_KAPLAN[prefix]
        gt_xc = gt[f"{prefix}_c"]
        gt_a = gt["alpha"]
        gt_Linf = gt["L_inf"]
        print(f"\n  L({prefix}):")
        print(f"    {prefix}_c : fitted={x_c_hat:.3e}  ground-truth={gt_xc:.3e}")
        print(f"    alpha   : fitted={a_hat:.4f}     ground-truth={gt_a:.4f}")
        print(f"    L_inf   : fitted={L_inf_hat:.4f}     ground-truth={gt_Linf:.4f}")

    # --- Fit Chinchilla ---
    print("\n" + "=" * 70)
    print("Chinchilla joint power law fit  L(N,D) = E + A/N^a + B/D^b")
    print("=" * 70)

    f = fit_chinchilla(Nch, Dch, Lch)
    E_hat, A_hat, a_hat, B_hat, b_hat, _ = f
    g = GT_CHINCHILLA
    for name, fval, gval in [
        ("E", E_hat, g["E"]),
        ("A", A_hat, g["A"]),
        ("alpha", a_hat, g["alpha"]),
        ("B", B_hat, g["B"]),
        ("beta", b_hat, g["beta"]),
    ]:
        print(f"  {name:6s}: fitted={fval:.4f}  ground-truth={gval:.4f}")

    # --- IsoFLOP analysis ---
    print("\n" + "=" * 70)
    print("IsoFLOP optimal (N, D) for various compute budgets")
    print("=" * 70)
    print(
        f"  {'C (FLOPs)':>14s}  {'N_opt':>10s}  {'D_opt':>10s}  {'N/D ratio':>10s}  {'L_opt':>8s}"
    )
    print("  " + "-" * 60)

    for C in np.logspace(15, 21, 7):
        N_opt, D_opt, L_opt = find_isflop_optimal(
            C,
            *GT_CHINCHILLA.values(),
            n_points=1000,
        )
        print(
            f"  {C:14.2e}  {N_opt:10.2e}  {D_opt:10.2e}  "
            f"{N_opt / D_opt:10.4f}  {L_opt:8.4f}"
        )


if __name__ == "__main__":
    main()
