"""
compute_optimal.py — Compute-Optimal Model & Dataset Size

Given the Chinchilla law  L(N, D) = E + A/N^alpha + B/D^beta  and the
approximate transformer FLOPs formula  C ≈ 6 N D, this module derives the
optimal N and D that minimize loss for a fixed compute budget.

The analytical solution comes from solving  dL/dN = 0  subject to D = C / (6N):

    N_opt = [ (alpha * A * C^beta) / (beta * B * 6^beta) ] ^ (1 / (alpha + beta))
    D_opt = C / (6 * N_opt)

Usage:
    python compute_optimal.py
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Tuple


# ---------------------------------------------------------------------------
# Chinchilla ground-truth (matched with scaling_law_fit.py)
# ---------------------------------------------------------------------------

CHINCHILLA_PARAMS = {
    "E": 1.69,
    "A": 406.4,
    "alpha": 0.34,
    "B": 410.7,
    "beta": 0.28,
}


# ---------------------------------------------------------------------------
# Chinchilla loss function
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
    """Evaluate the Chinchilla joint power-law loss.

    Args:
        N: Model parameter count (array or scalar).
        D: Dataset size in tokens (array or scalar).
        E, A, alpha, B, beta: Chinchilla parameters.

    Returns:
        Loss value(s).
    """
    return E + A / (N**alpha) + B / (D**beta)


# ---------------------------------------------------------------------------
# Analytical compute-optimal solution
# ---------------------------------------------------------------------------


def compute_optimal_analytical(
    C: float,
    E: float = CHINCHILLA_PARAMS["E"],
    A: float = CHINCHILLA_PARAMS["A"],
    alpha: float = CHINCHILLA_PARAMS["alpha"],
    B: float = CHINCHILLA_PARAMS["B"],
    beta: float = CHINCHILLA_PARAMS["beta"],
) -> Tuple[float, float, float]:
    """Compute optimal N, D analytically for a given compute budget C.

    Derivation:
        L(N, D) = E + A/N^a + B/D^b,  constraint C = 6 N D.
        Substitute D = C/(6N):
            L(N) = E + A/N^a + B*(6N)^b / C^b
        dL/dN = -aA/N^{a+1} + bB*6^b*N^{b-1}/C^b = 0
        => N_opt^{a+b} = (aA * C^b) / (bB * 6^b)
        => N_opt = [aA*C^b / (bB*6^b)] ^ (1/(a+b))
        D_opt = C / (6 * N_opt)

    Args:
        C: Compute budget in FLOPs (C ≈ 6 N D for transformers).
        E, A, alpha, B, beta: Chinchilla law parameters.

    Returns:
        Tuple of (N_opt, D_opt, L_opt).
    """
    numer = alpha * A * (C**beta)
    denom = beta * B * (6.0**beta)
    N_opt = (numer / denom) ** (1.0 / (alpha + beta))
    D_opt = C / (6.0 * N_opt)
    L_opt = chinchilla_loss(np.array(N_opt), np.array(D_opt), E, A, alpha, B, beta)
    return float(N_opt), float(D_opt), float(L_opt)


# ---------------------------------------------------------------------------
# Numerical optimization (grid-search refinement)
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
    """Find optimal (N, D) numerically by sweeping N on a fine log-spaced grid."""
    N_vals = np.logspace(4, 11, n_grid)  # wide sweep
    D_vals = C / (6.0 * N_vals)
    L_vals = chinchilla_loss(N_vals, D_vals, E, A, alpha, B, beta)
    idx = int(np.argmin(L_vals))
    return float(N_vals[idx]), float(D_vals[idx]), float(L_vals[idx])


# ---------------------------------------------------------------------------
# Compute-optimal ratio analysis
# ---------------------------------------------------------------------------


def compute_optimal_ratio(
    C: float,
    E: float = CHINCHILLA_PARAMS["E"],
    A: float = CHINCHILLA_PARAMS["A"],
    alpha: float = CHINCHILLA_PARAMS["alpha"],
    B: float = CHINCHILLA_PARAMS["B"],
    beta: float = CHINCHILLA_PARAMS["beta"],
) -> float:
    """Return the compute-optimal N/D ratio for a given budget C.

    From the analytical derivation:
        N/D = (alpha * A / (beta * B)) * (C / 6)^{beta - alpha}
              * (1/6)^{...} -- but note N_opt depends on C.
    Actually:
        N_opt = [alpha*A*C^beta / (beta*B*6^beta)] ^ (1/(alpha+beta))
        D_opt = C / (6*N_opt)
        => N_opt / D_opt = 6 * N_opt^2 / C

    Using the analytical expression:
        N_opt^{alpha+beta} = alpha*A*C^beta / (beta*B*6^beta)
        => N_opt^2 = [alpha*A / (beta*B*6^beta)] ^ (2/(alpha+beta)) * C^(2beta/(alpha+beta))
        => N_opt/D_opt = 6 * N_opt^2 / C
                       = 6 * [alpha*A/(beta*B*6^beta)]^(2/(alpha+beta)) * C^((beta-alpha)/(alpha+beta))

    For Chinchilla (~alpha=0.34, beta=0.28): the exponent (beta-alpha)/(alpha+beta) < 0,
    so the ratio decreases with C, meaning larger models need proportionally more data.
    """
    N_opt, D_opt, _ = compute_optimal_analytical(C, E, A, alpha, B, beta)
    return N_opt / D_opt


# ---------------------------------------------------------------------------
# Demonstration
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

    # Sanity check: analytical vs numerical should agree
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

    # N/D ratio trend
    print("\n" + "=" * 72)
    print("N/D ratio trend with compute budget")
    print("=" * 72)
    for C in compute_budgets:
        ratio = compute_optimal_ratio(C)
        print(f"  C={C:14.2e}  N/D ratio = {ratio:.4f}")


if __name__ == "__main__":
    main()
