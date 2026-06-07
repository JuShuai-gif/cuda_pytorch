"""
compute_optimal.py — Compute-Optimal 模型大小与数据集大小

给定 Chinchilla 定律  L(N, D) = E + A/N^alpha + B/D^beta  以及
近似的 Transformer FLOPs 公式  C ≈ 6 N D，本模块推导
在固定 compute budget 下使 loss 最小化的最优 N 和 D。

解析解通过求解  dL/dN = 0  并在约束  D = C / (6N) 下得到:

    N_opt = [ (alpha * A * C^beta) / (beta * B * 6^beta) ] ^ (1 / (alpha + beta))
    D_opt = C / (6 * N_opt)

用法:
    python compute_optimal.py
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Tuple


# ---------------------------------------------------------------------------
# Chinchilla 真实参数（与 scaling_law_fit.py 保持一致）
# ---------------------------------------------------------------------------

CHINCHILLA_PARAMS = {
    "E": 1.69,
    "A": 406.4,
    "alpha": 0.34,
    "B": 410.7,
    "beta": 0.28,
}


# ---------------------------------------------------------------------------
# Chinchilla loss 函数
# ---------------------------------------------------------------------------


def chinchilla_loss(
    N: NDArray[np.float64],
    D: NDArray[np.float64],
    E: float,
    A: float,
    alpha: float,
    B: float,
    beta: float,
) -> NDArray[np.float64]:
    """计算 Chinchilla 联合幂律 loss。

    Args:
        N: 模型参数数量（数组或标量）。
        D: 以 tokens 计的数据集大小（数组或标量）。
        E, A, alpha, B, beta: Chinchilla 参数。

    Returns:
        Loss 值。
    """
    return E + A / (N**alpha) + B / (D**beta)


# ---------------------------------------------------------------------------
# 解析 compute-optimal 解
# ---------------------------------------------------------------------------


def compute_optimal_analytical(
    C: float,
    E: float = CHINCHILLA_PARAMS["E"],
    A: float = CHINCHILLA_PARAMS["A"],
    alpha: float = CHINCHILLA_PARAMS["alpha"],
    B: float = CHINCHILLA_PARAMS["B"],
    beta: float = CHINCHILLA_PARAMS["beta"],
) -> Tuple[float, float, float]:
    """对于给定的 compute budget C，通过解析方法计算最优 N, D。

    推导:
        L(N, D) = E + A/N^a + B/D^b,  约束 C = 6 N D。
        代入 D = C/(6N):
            L(N) = E + A/N^a + B*(6N)^b / C^b
        dL/dN = -aA/N^{a+1} + bB*6^b*N^{b-1}/C^b = 0
        => N_opt^{a+b} = (aA * C^b) / (bB * 6^b)
        => N_opt = [aA*C^b / (bB*6^b)] ^ (1/(a+b))
        D_opt = C / (6 * N_opt)

    Args:
        C: 以 FLOPs 计的 compute budget（对于 Transformers，C ≈ 6 N D）。
        E, A, alpha, B, beta: Chinchilla 定律参数。

    Returns:
        (N_opt, D_opt, L_opt) 元组。
    """
    numer = alpha * A * (C**beta)
    denom = beta * B * (6.0**beta)
    N_opt = (numer / denom) ** (1.0 / (alpha + beta))
    D_opt = C / (6.0 * N_opt)
    L_opt = chinchilla_loss(np.array(N_opt), np.array(D_opt), E, A, alpha, B, beta)
    return float(N_opt), float(D_opt), float(L_opt)


# ---------------------------------------------------------------------------
# 数值优化（网格搜索细化）
# ---------------------------------------------------------------------------


def compute_optimal_numerical(
    C: float,
    E: float = CHINCHILLA_PARAMS["E"],
    A: float = CHINCHILLA_PARAMS["A"],
    alpha: float = CHINCHILLA_PARAMS["alpha"],
    B: float = CHINCHILLA_PARAMS["B"],
    beta: float = CHINCHILLA_PARAMS["beta"],
    n_grid: int = 2000,
) -> Tuple[float, float, float]:
    """通过在细粒度对数间隔网格上扫描 N 来数值求解最优 (N, D)。"""
    N_vals = np.logspace(4, 11, n_grid)  # wide sweep
    D_vals = C / (6.0 * N_vals)
    L_vals = chinchilla_loss(N_vals, D_vals, E, A, alpha, B, beta)
    idx = int(np.argmin(L_vals))
    return float(N_vals[idx]), float(D_vals[idx]), float(L_vals[idx])


# ---------------------------------------------------------------------------
# Compute-optimal 比率分析
# ---------------------------------------------------------------------------


def compute_optimal_ratio(
    C: float,
    E: float = CHINCHILLA_PARAMS["E"],
    A: float = CHINCHILLA_PARAMS["A"],
    alpha: float = CHINCHILLA_PARAMS["alpha"],
    B: float = CHINCHILLA_PARAMS["B"],
    beta: float = CHINCHILLA_PARAMS["beta"],
) -> float:
    """对于给定的 budget C，返回 compute-optimal 的 N/D 比率。

    由解析推导可得:
        N/D = (alpha * A / (beta * B)) * (C / 6)^{beta - alpha}
              * (1/6)^{...} -- 但注意 N_opt 依赖于 C。
    实际上:
        N_opt = [alpha*A*C^beta / (beta*B*6^beta)] ^ (1/(alpha+beta))
        D_opt = C / (6*N_opt)
        => N_opt / D_opt = 6 * N_opt^2 / C

    使用解析表达式:
        N_opt^{alpha+beta} = alpha*A*C^beta / (beta*B*6^beta)
        => N_opt^2 = [alpha*A / (beta*B*6^beta)] ^ (2/(alpha+beta)) * C^(2beta/(alpha+beta))
        => N_opt/D_opt = 6 * N_opt^2 / C
                       = 6 * [alpha*A/(beta*B*6^beta)]^(2/(alpha+beta)) * C^((beta-alpha)/(alpha+beta))

    对于 Chinchilla (~alpha=0.34, beta=0.28): 指数 (beta-alpha)/(alpha+beta) < 0，
    因此比率随 C 减小，意味着 larger models need proportionally more data。
    """
    N_opt, D_opt, _ = compute_optimal_analytical(C, E, A, alpha, B, beta)
    return N_opt / D_opt


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 72)
    print("Compute-Optimal Model Size & Dataset Size (Chinchilla Law)")
    print("=" * 72)
    print(
        f"Parameters: E={CHINCHILLA_PARAMS['E']}, A={CHINCHILLA_PARAMS['A']}, "
        f"alpha={CHINCHILLA_PARAMS['alpha']}, B={CHINCHILLA_PARAMS['B']}, "
        f"beta={CHINCHILLA_PARAMS['beta']}"
    )
    print()

    compute_budgets = np.logspace(15, 22, 8)  # 1e15 to 1e22 FLOPs

    print(
        f"{'C (FLOPs)':>14s}  {'N_opt (anal.)':>14s}  {'D_opt (anal.)':>14s}  "
        f"{'N/D ratio':>10s}  {'Loss':>8s}  {'N_opt (num.)':>14s}"
    )
    print("-" * 84)

    for C in compute_budgets:
        Na, Da, La = compute_optimal_analytical(C)
        Nn, Dn, Ln = compute_optimal_numerical(C, n_grid=5000)
        ratio = Na / Da
        print(
            f"{C:14.2e}  {Na:14.2e}  {Da:14.2e}  {ratio:10.4f}  {La:8.5f}  {Nn:14.2e}"
        )

    # 合理性检查: 解析解与数值解应当一致
    print("\n" + "=" * 72)
    print("Analytical vs Numerical consistency check")
    print("=" * 72)
    max_diff_pct = 0.0
    for C in compute_budgets:
        Na, _, _ = compute_optimal_analytical(C)
        Nn, _, _ = compute_optimal_numerical(C, n_grid=5000)
        diff_pct = abs(Na - Nn) / Na * 100
        max_diff_pct = max(max_diff_pct, diff_pct)
        status = "OK" if diff_pct < 0.5 else "WARN"
        print(
            f"  C={C:.2e}: N_anal={Na:.4e}  N_num={Nn:.4e}  "
            f"diff={diff_pct:.3f}%  [{status}]"
        )
    print(f"\n  Maximum discrepancy: {max_diff_pct:.4f}%")

    # N/D 比率趋势
    print("\n" + "=" * 72)
    print("N/D ratio trend with compute budget")
    print("=" * 72)
    for C in compute_budgets:
        ratio = compute_optimal_ratio(C)
        print(f"  C={C:14.2e}  N/D ratio = {ratio:.4f}")


if __name__ == "__main__":
    main()
