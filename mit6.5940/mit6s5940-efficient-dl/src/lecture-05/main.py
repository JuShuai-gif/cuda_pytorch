"""
Linear Quantization int8/int4/int2 & K-Means Quantization (Lecture 05)
======================================================================
Implements linear (affine) quantization with configurable bit widths,
K-means-based non-linear quantization, and visual comparison of
quantization levels against the original weight distribution.

Key concepts:
  - linear_quantize: asymmetric affine quantisation to b bits
  - dequantize: reconstruct approximate float values
  - kmeans_quantize: cluster weights via K-means, store codebook as quantized values
  - compare errors (MSE, MAE, cosine similarity) across int8, int4, int2
  - plot weight histogram overlaid with quantisation grid lines

All computations run on CPU; no GPU required.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

# Use a non-interactive backend so plots can be saved without a display
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BITS_LIST: List[int] = [8, 4, 2]
SEED: int = 42
NUM_WEIGHTS: int = 5000  # number of synthetic weights to generate

# ---------------------------------------------------------------------------
# Linear (Affine) Quantization
# ---------------------------------------------------------------------------


def linear_quantize(
    tensor: torch.Tensor, bits: int
) -> Tuple[torch.Tensor, float, int, float, float]:
    """Quantize a float tensor to `bits`-bit integers using asymmetric quantisation.

    The affine mapping is:

        scale  = (x_max - x_min) / (2^bits - 1)
        zp     = round(-x_min / scale)     [clamped to [0, 2^bits - 1]]
        q      = round(x / scale + zp)     [clamped to [0, 2^bits - 1]]

    Args:
        tensor: Float32 tensor of any shape.
        bits:   Bit width (e.g. 8, 4, 2).

    Returns:
        Tuple of (quantized_tensor_int, scale, zero_point, x_min, x_max).
    """
    if bits <= 0:
        raise ValueError(f"bits must be positive; got {bits}")

    qmin: int = 0
    qmax: int = int(2**bits - 1)

    x_min = tensor.min().item()
    x_max = tensor.max().item()

    # Avoid division by zero when all values are identical
    if x_max == x_min:
        scale = 1.0
        zp = 0
        q = torch.zeros_like(tensor, dtype=torch.float32).round().long()
        return q, scale, zp, x_min, x_max

    scale = (x_max - x_min) / (qmax - qmin)

    # Compute zero point: the quantised value corresponding to float 0.0
    zp_f = -x_min / scale
    zp = int(round(zp_f))
    zp = max(qmin, min(qmax, zp))  # clamp

    # Quantize: x -> q
    q = torch.round(tensor / scale + zp)
    q = torch.clamp(q, qmin, qmax).long()

    return q, scale, zp, x_min, x_max


def dequantize(q: torch.Tensor, scale: float, zp: int) -> torch.Tensor:
    """Reconstruct approximate float values from quantized integers.

    Args:
        q:     Quantized integer tensor (int32/int64).
        scale: Quantisation scale factor (float).
        zp:    Zero point (integer).

    Returns:
        Reconstructed float32 tensor.
    """
    return (q.float() - zp) * scale


def compute_quantization_error(
    original: torch.Tensor, reconstructed: torch.Tensor
) -> Dict[str, float]:
    """Compute error metrics between the original and reconstructed tensors.

    Args:
        original:      The original float tensor.
        reconstructed: The dequantized float tensor.

    Returns:
        Dictionary with keys: 'mse', 'mae', 'cosine_sim', 'max_abs_err'.
    """
    orig_flat = original.view(-1).float()
    recon_flat = reconstructed.view(-1).float()

    mse = torch.mean((orig_flat - recon_flat) ** 2).item()
    mae = torch.mean(torch.abs(orig_flat - recon_flat)).item()
    max_abs_err = torch.max(torch.abs(orig_flat - recon_flat)).item()

    # Cosine similarity: dot(a, b) / (||a|| * ||b||)
    dot = torch.dot(orig_flat, recon_flat).item()
    norm_orig = orig_flat.norm(p=2).item()
    norm_recon = recon_flat.norm(p=2).item()
    if norm_orig > 1e-12 and norm_recon > 1e-12:
        cosine_sim = dot / (norm_orig * norm_recon)
    else:
        cosine_sim = 1.0

    return {
        "mse": mse,
        "mae": mae,
        "max_abs_err": max_abs_err,
        "cosine_sim": cosine_sim,
    }


# ---------------------------------------------------------------------------
# K-Means Quantization
# ---------------------------------------------------------------------------


def kmeans_quantize(
    tensor: torch.Tensor,
    bits: int,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize weights using K-means clustering.

    Each weight is assigned to one of k = 2^bits clusters.  The cluster
    centroids form the codebook; the assignment indices are stored as the
    quantized representation.  This is a non-uniform quantisation scheme.

    Uses scipy.cluster.vq.kmeans2 if available; otherwise falls back to a
    pure-PyTorch Lloyd implementation.

    Args:
        tensor:   Float32 tensor of any shape.
        bits:     Bit width (number of clusters = 2^bits).
        max_iter: Maximum K-means iterations.
        tol:      Convergence tolerance.

    Returns:
        Tuple of (assignments, centroids, reconstructed).
          assignments:  int32 tensor with original shape, values in [0, 2^bits - 1].
          centroids:    float32 tensor of shape (2^bits,) -- the codebook.
          reconstructed: dequantised float32 tensor with original shape.
    """
    num_clusters = int(2**bits)
    data = tensor.view(-1, 1).float()  # (N, 1)

    try:
        from scipy.cluster.vq import kmeans2

        data_np = data.numpy()
        centroids_np, assignments_np = kmeans2(
            data_np[:, 0], num_clusters, iter=max_iter, thresh=tol, minit="points"
        )
        # kmeans2 returns centroids as 1-D; convert to torch
        centroids = torch.from_numpy(centroids_np).float()
        assignments = torch.from_numpy(assignments_np.astype(np.int32)).long()
    except ImportError:
        # Fallback: pure-PyTorch K-means (Lloyd's algorithm)
        centroids = _kmeans_fallback(data.squeeze(-1), num_clusters, max_iter, tol)
        diffs = (data.squeeze(-1).unsqueeze(1) - centroids.unsqueeze(0)).abs()
        assignments = diffs.argmin(dim=1).long()

    # Reconstruct
    reconstructed = centroids[assignments].view(tensor.shape)
    assignments = assignments.view(tensor.shape)

    return assignments, centroids, reconstructed


def _kmeans_fallback(
    data: torch.Tensor,
    num_clusters: int,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> torch.Tensor:
    """Pure-PyTorch K-means (Lloyd's algorithm) for 1-D clustering.

    Args:
        data:         1-D float tensor (N,).
        num_clusters: Number of clusters.
        max_iter:     Maximum iterations.
        tol:          Convergence tolerance for centroid movement.

    Returns:
        Centroids tensor of shape (num_clusters,), sorted ascending.
    """
    n = data.numel()
    # Initialise centroids by sampling evenly-spaced percentiles
    sorted_data = torch.sort(data).values
    indices = torch.linspace(0, n - 1, num_clusters).long()
    centroids = sorted_data[indices].clone()

    for _iter in range(max_iter):
        # Assign each point to the nearest centroid
        diffs = (data.unsqueeze(1) - centroids.unsqueeze(0)).abs()
        assignments = diffs.argmin(dim=1)

        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        for k in range(num_clusters):
            mask = assignments == k
            if mask.sum() > 0:
                new_centroids[k] = data[mask].mean()
            else:
                new_centroids[k] = centroids[k]  # keep old if empty

        shift = (new_centroids - centroids).abs().max().item()
        centroids = new_centroids

        if shift < tol:
            break

    return torch.sort(centroids).values


# ---------------------------------------------------------------------------
# Weight Generation
# ---------------------------------------------------------------------------


def generate_synthetic_weights(
    num_weights: int = NUM_WEIGHTS,
    seed: int = SEED,
) -> torch.Tensor:
    """Generate synthetic weights mimicking a real weight distribution.

    The distribution is a mixture of:
      - Gaussian N(0, 0.5) for the bulk of weights
      - A few outliers drawn from N(0, 2.0) to simulate heavy tails
      - Negative-skewed component to make it asymmetric

    Args:
        num_weights: Number of scalar weights to generate.
        seed:        Random seed for reproducibility.

    Returns:
        1-D float tensor of length `num_weights`.
    """
    torch.manual_seed(seed)

    # Bulk: normal distribution centered at 0
    bulk = torch.randn(int(num_weights * 0.85)) * 0.5

    # Outliers / heavy tails
    tails = torch.randn(int(num_weights * 0.10)) * 2.0

    # Slight positive skew
    skewed = torch.randn(int(num_weights * 0.05)) * 0.8 + 1.5

    weights = torch.cat([bulk, tails, skewed])

    # Trim to exact count
    if weights.numel() > num_weights:
        weights = weights[:num_weights]
    elif weights.numel() < num_weights:
        extra = torch.randn(num_weights - weights.numel()) * 0.5
        weights = torch.cat([weights, extra])

    return weights


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def plot_weight_histogram(
    weights: torch.Tensor,
    quant_info: Dict[int, Dict[str, object]],
    save_path: str = "quantization_histogram.png",
) -> None:
    """Plot the original weight histogram with overlaid quantization grid lines.

    For each bit width, the quantization levels (dequantized values) are drawn
    as vertical dashed lines on top of the histogram.

    Args:
        weights:     Original 1-D float tensor.
        quant_info:  Dictionary keyed by bits:
                     {bits: {"levels": List[float], "scale": float, "zp": int}}
        save_path:   File path to save the figure (PNG).
    """
    w_np = weights.numpy()

    fig, axes = plt.subplots(len(BITS_LIST), 1, figsize=(10, 4 * len(BITS_LIST)))

    for idx, bits in enumerate(BITS_LIST):
        ax = axes[idx]
        ax.hist(w_np, bins=80, color="steelblue", alpha=0.7, edgecolor="white")
        ax.set_title(f"Weight Histogram + int{bits} Quantization Levels", fontsize=13)
        ax.set_xlabel("Weight Value", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)

        # Draw quantization levels as vertical lines
        if bits in quant_info:
            levels = quant_info[bits].get("levels", [])
            if levels:
                for lv in levels:
                    ax.axvline(
                        x=lv,
                        color="red",
                        linestyle="--",
                        linewidth=0.8,
                        alpha=0.6,
                    )
                # Add a proxy artist for the legend
                ax.axvline(
                    x=levels[0],
                    color="red",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.6,
                    label=f"Quant levels ({len(levels)})",
                )
                ax.legend(loc="upper right", fontsize=9)

        # Add annotation
        if bits in quant_info:
            scale = quant_info[bits].get("scale", 0.0)
            text = f"int{bits}: {len(quant_info[bits].get('levels', []))} levels, scale={scale:.4f}"
            ax.text(
                0.98,
                0.95,
                text,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\nWeight histogram saved to: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full quantisation demonstration pipeline."""
    torch.manual_seed(SEED)

    print("=" * 70)
    print("  LECTURE 05: Linear Quantization int8 / int4 / int2")
    print("=" * 70)

    # ---- 1. Generate synthetic weights --------------------------------------
    print(f"\n[1] Generating {NUM_WEIGHTS} synthetic weights ...")
    weights = generate_synthetic_weights(NUM_WEIGHTS, SEED)
    print(f"  Shape: {tuple(weights.shape)}")
    print(f"  Min: {weights.min().item():.4f},  Max: {weights.max().item():.4f}")
    print(f"  Mean: {weights.mean().item():.4f},  Std: {weights.std().item():.4f}")

    # ---- 2. Linear (affine) quantisation ------------------------------------
    print("\n[2] Linear (affine) quantisation across bit widths ...")
    linear_results: Dict[int, Dict[str, object]] = {}
    quant_level_map: Dict[int, Dict[str, object]] = {}

    for bits in BITS_LIST:
        q, scale, zp, x_min, x_max = linear_quantize(weights, bits)
        reconstructed = dequantize(q, scale, zp)
        errors = compute_quantization_error(weights, reconstructed)

        # Compute unique quantisation levels (dequantised values)
        unique_q = torch.unique(q).long()
        levels = ((unique_q.float() - zp) * scale).tolist()

        linear_results[bits] = {
            "q": q,
            "scale": scale,
            "zp": zp,
            "reconstructed": reconstructed,
            "errors": errors,
        }
        quant_level_map[bits] = {"levels": levels, "scale": scale, "zp": zp}

        print(
            f"  int{bits:>2d}:  "
            f"range=[{x_min:.4f}, {x_max:.4f}], "
            f"scale={scale:.6f}, "
            f"zp={zp}, "
            f"levels={len(levels)}, "
            f"MSE={errors['mse']:.6f}, "
            f"MAE={errors['mae']:.6f}, "
            f"max_err={errors['max_abs_err']:.6f}, "
            f"cos_sim={errors['cosine_sim']:.6f}"
        )

    # ---- 3. Error comparison summary ----------------------------------------
    print("\n[3] Quantisation error comparison:")
    print(
        f"  {'Bit Width':<12} {'MSE':>12} {'MAE':>12} {'Max Abs Err':>14} {'Cos Sim':>10}"
    )
    print(f"  {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 14} {'-' * 10}")
    for bits in BITS_LIST:
        e = linear_results[bits]["errors"]
        print(
            f"  int{bits:<9d} {e['mse']:>12.6f} {e['mae']:>12.6f} "
            f"{e['max_abs_err']:>14.6f} {e['cosine_sim']:>10.6f}"
        )

    # ---- 4. K-Means quantisation --------------------------------------------
    print("\n[4] K-Means quantisation (non-uniform) ...")
    kmeans_rec: Dict[int, torch.Tensor] = {}

    for bits in BITS_LIST:
        assignments, centroids, reconstructed = kmeans_quantize(weights, bits)
        errors = compute_quantization_error(weights, reconstructed)
        kmeans_rec[bits] = reconstructed

        print(
            f"  int{bits:>2d}:  "
            f"clusters={centroids.numel()}, "
            f"MSE={errors['mse']:.6f}, "
            f"MAE={errors['mae']:.6f}, "
            f"max_err={errors['max_abs_err']:.6f}, "
            f"cos_sim={errors['cosine_sim']:.6f}"
        )

    # ---- 5. Linear vs K-Means comparison ------------------------------------
    print("\n[5] Linear vs K-Means quantisation comparison:")
    print(
        f"  {'Bit Width':<12} {'Linear MSE':>12} {'K-Means MSE':>12} {'Improvement':>14}"
    )
    print(f"  {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 14}")
    for bits in BITS_LIST:
        linear_mse = linear_results[bits]["errors"]["mse"]
        kmeans_mse = compute_quantization_error(weights, kmeans_rec[bits])["mse"]
        improvement = (
            (1.0 - kmeans_mse / linear_mse) * 100 if linear_mse > 1e-12 else 0.0
        )
        print(
            f"  int{bits:<9d} {linear_mse:>12.6f} {kmeans_mse:>12.6f} "
            f"{improvement:>13.2f}%"
        )

    # ---- 6. Visualise weight histogram --------------------------------------
    print("\n[6] Plotting weight histogram with quantisation levels ...")
    plot_weight_histogram(
        weights, quant_level_map, save_path="quantization_histogram.png"
    )

    # ---- 7. Summary ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Synthetic weights: {NUM_WEIGHTS}")
    print(f"  Bit widths tested: {BITS_LIST}")
    print(f"  Quantisation: asymmetric affine (linear) + K-means (non-uniform)")
    print(f"  Plot saved to: quantization_histogram.png")
    print("=" * 70)

    print("\nLecture 05 complete.")


if __name__ == "__main__":
    main()
