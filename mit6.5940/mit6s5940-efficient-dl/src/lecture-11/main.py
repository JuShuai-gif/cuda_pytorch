"""
Lecture 11: TinyEngine Optimization Simulation
==============================================
Benchmark different convolution implementations (naive, im2col, Winograd),
demonstrate operator fusion (Conv+BN+ReLU), and analyze memory layout
trade-offs (NCHW vs NHWC).  All benchmarks run on CPU only.

Key concepts:
  - im2col: transform convolution into matrix multiplication
  - Winograd F(2,3): minimal filtering algorithm for 3x3 conv, stride 1
  - Operator fusion: fold BatchNorm parameters into Conv weights at inference
  - Memory layout: NCHW (channels-first) vs NHWC (channels-last) access patterns
"""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _timeit(fn, warmup: int = 3, repeat: int = 10) -> Tuple[float, float]:
    """Measure mean and std of execution time in milliseconds.

    Parameters
    ----------
    fn : callable
        Function to benchmark (zero-argument).
    warmup : int
        Number of warmup calls before timing.
    repeat : int
        Number of timed repetitions.

    Returns
    -------
    (mean_ms, std_ms) : (float, float)
    """
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    return float(arr.mean()), float(arr.std())


def _gflops(ms: float, ops: float) -> float:
    """Convert milliseconds to GFLOPS."""
    return ops / (ms * 1e6)  # ops per ms → ops per sec → GFLOPS


# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

# Use a representative image tensor: N=8, C=64, H=56, W=56 (e.g. ResNet C3)
BATCH, IN_C, OUT_C, H, W = 8, 64, 64, 56, 56
KERNEL = 3
STRIDE = 1
PADDING = 1

DEVICE = torch.device("cpu")
DTYPE = torch.float32


def _make_tensors():
    """Create consistent input/weight tensors for all benchmarks."""
    rng = torch.Generator(device=DEVICE).manual_seed(42)
    x = torch.randn(BATCH, IN_C, H, W, device=DEVICE, dtype=DTYPE, generator=rng)
    w = torch.randn(
        OUT_C, IN_C, KERNEL, KERNEL, device=DEVICE, dtype=DTYPE, generator=rng
    )
    return x, w


# ===========================================================================
# 1. NAIVE CONVOLUTION (PyTorch baseline via F.conv2d)
# ===========================================================================


def naive_conv2d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Baseline convolution using PyTorch's highly-optimised CPU backend.

    This serves as the "naive" reference — not truly a nested-loop
    implementation, but the fastest CPU version available through
    PyTorch's oneDNN integration.
    """
    return F.conv2d(x, w, stride=STRIDE, padding=PADDING)


# ===========================================================================
# 2. IM2COL CONVOLUTION
# ===========================================================================


def _im2col(x: torch.Tensor, k_h: int, k_w: int, stride: int, pad: int) -> torch.Tensor:
    """Transform an NCHW image tensor into a column matrix for GEMM.

    Each column holds a flattened k_h*k_w*C_in receptive field.  The
    output has shape (C_in*k_h*k_w, N*H_out*W_out).

    Parameters
    ----------
    x : (N, C, H, W)
    k_h, k_w : kernel height / width
    stride : int
    pad : int

    Returns
    -------
    cols : (C * k_h * k_w, N * H_out * W_out)
    """
    N, C, H_in, W_in = x.shape
    H_out = (H_in + 2 * pad - k_h) // stride + 1
    W_out = (W_in + 2 * pad - k_w) // stride + 1

    # Pad input
    x_pad = F.pad(x, (pad, pad, pad, pad))  # (N, C, H_pad, W_pad)

    # Use unfold to extract sliding windows — each (k_h, k_w) patch
    # becomes a column of length C * k_h * k_w.  The patch elements
    # are ordered channel-first, then height, then width.
    patches = x_pad.unfold(2, k_h, stride).unfold(3, k_w, stride)
    # patches: (N, C, H_out, W_out, k_h, k_w)
    # Reorder so that spatial position varies first, then channel & kernel
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    # patches: (N, H_out, W_out, C, k_h, k_w)
    cols = patches.view(N * H_out * W_out, C * k_h * k_w).t()
    # cols: (C * k_h * k_w, N * H_out * W_out)
    return cols


def im2col_conv2d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """im2col + GEMM convolution.

    Algorithm:
      1.  Unfold input into column matrix          (C_in*K*K, N*H_out*W_out)
      2.  Reshape weights into row matrix           (C_out, C_in*K*K)
      3.  Matrix multiply: out = W @ cols
      4.  Reshape back to NCHW
    """
    N, C_in, H_in, W_in = x.shape
    C_out, _, k_h, k_w = w.shape
    H_out = (H_in + 2 * PADDING - k_h) // STRIDE + 1
    W_out = (W_in + 2 * PADDING - k_w) // STRIDE + 1

    # 1. im2col
    cols = _im2col(x, k_h, k_w, STRIDE, PADDING)  # (C_in*K*K, N*H_out*W_out)

    # 2. Reshape filter
    w_mat = w.view(C_out, -1)  # (C_out, C_in*K*K)

    # 3. GEMM
    out_mat = w_mat @ cols  # (C_out, N*H_out*W_out)

    # 4. Reshape → NCHW
    out = out_mat.view(C_out, N, H_out, W_out).permute(1, 0, 2, 3).contiguous()
    return out


# ===========================================================================
# 3. WINOGRAD CONVOLUTION – F(2, 3)
# ===========================================================================


def _winograd_transform_matrices():
    """Return Aᵀ, G, Bᵀ for Winograd F(2, 3).

    F(m, r)  = F(2, 3) produces m=2 outputs from an r=3 filter.
    The theoretical arithmetic reduction is  m²·r² / (m+r-1)²
    = (4·9) / 16 = 2.25×  for 3×3 filters with stride 1.
    """
    # Transformation matrices from the Winograd minimal filtering algorithm
    # (Lavin & Gray, 2016)
    B_T = torch.tensor(
        [
            [1.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, -1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, -1.0],
        ],
        device=DEVICE,
        dtype=DTYPE,
    )
    G = torch.tensor(
        [[1.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.0, 0.0, 1.0]],
        device=DEVICE,
        dtype=DTYPE,
    )
    A_T = torch.tensor(
        [[1.0, 1.0, 1.0, 0.0], [0.0, 1.0, -1.0, -1.0]],
        device=DEVICE,
        dtype=DTYPE,
    )
    return A_T, G, B_T


def winograd_f23_conv2d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Winograd F(2,3) convolution for 3×3 filters, stride 1.

    The algorithm works in the *Winograd domain*:
      1.  Transform input tiles   : V = Bᵀ · tile · B
      2.  Transform filter        : U = G · filter · Gᵀ
      3.  Element-wise multiply   : M = U ⊙ V
      4.  Inverse transform       : out_tile = Aᵀ · M · A

    For a 3×3 filter (r=3) producing 2×2 output (m=2) per tile,
    the internal tile size is α = m + r − 1 = 4.

    References
    ----------
    Lavin & Gray, "Fast Algorithms for Convolutional Neural Networks", CVPR 2016
    """
    N, C_in, H_in, W_in = x.shape
    C_out, _, k_h, k_w = w.shape
    assert k_h == 3 and k_w == 3, "F(2,3) requires 3×3 kernels"
    assert STRIDE == 1, "F(2,3) requires stride 1"

    H_out = H_in - k_h + 2 * PADDING + 1  # =H_in for pad=1
    W_out = W_in - k_w + 2 * PADDING + 1
    assert H_out % 2 == 0 and W_out % 2 == 0, (
        f"Output spatial dims must be even for tiling; got {H_out}×{W_out}"
    )

    tile_H = H_out // 2
    tile_W = W_out // 2
    alpha = 4  # m + r - 1 = 2 + 3 - 1

    A_T, G, B_T = _winograd_transform_matrices()

    # Pad input to facilitate tiling
    x_pad = F.pad(x, (PADDING, PADDING, PADDING, PADDING))

    # ---- Step 1: Transform filter into Winograd domain ----
    # U = G @ w @ Gᵀ, shape: (C_out, C_in, alpha, alpha)
    # G: (alpha=4, k=3).  w: (C_out, C_in, k, k).
    # Apply G along the k_h (height) dimension first.
    U = torch.einsum("ax,oixy->oiay", G, w)  # (C_out, C_in, 4, 3)
    # Apply G along the k_w (width) dimension via right-multiply:
    #   (G @ w @ Gᵀ)[o,i,a,b] = Σ_y (G @ w)[o,i,a,y] · G[b,y]
    U = torch.einsum("by,oiay->oiab", G, U)  # (C_out, C_in, 4, 4)

    # ---- Step 2: Transform input tiles into Winograd domain ----
    # V = Bᵀ @ tile @ B
    # Each tile covers an α×α patch with stride 2.
    tiles = x_pad.unfold(2, alpha, 2).unfold(3, alpha, 2)
    # tiles: (N, C_in, tile_H, tile_W, alpha, alpha)
    # Reorder so that spatial tile index varies before channel index,
    # ensuring correct contiguous layout before flattening.
    tiles = tiles.permute(0, 2, 3, 1, 4, 5).contiguous()
    tiles = tiles.view(N * tile_H * tile_W, C_in, alpha, alpha)

    # Bᵀ @ tile  (left multiply)
    V = torch.einsum("ax,nixy->niay", B_T, tiles)  # (N_tiles, C_in, 4, 4)
    # (Bᵀ @ tile) @ B  (right multiply: Σ_y temp[.,a,y] · B[b,y])
    V = torch.einsum("by,niay->niab", B_T, V)  # (N_tiles, C_in, 4, 4)

    # ---- Step 3: Element-wise multiply + sum over input channels ----
    # M[n, o, p, q] = Σ_i U[o, i, p, q] · V[n, i, p, q]
    M = torch.einsum("oipq,nipq->nopq", U, V)  # (N_tiles, C_out, 4, 4)

    # ---- Step 4: Inverse transform – out_tile = Aᵀ @ M @ A ----
    # Aᵀ: (m=2, alpha=4).  Labels: "cx" where c=2, x=4.
    # left  multiply: temp[n,o,c,y] = Σ_x Aᵀ[c,x] · M[n,o,x,y]
    out = torch.einsum("cx,noxy->nocy", A_T, M)  # (N_tiles, C_out, 2, 4)
    # right multiply: out_tile[n,o,c,d] = Σ_y temp[n,o,c,y] · A[d,y]
    out = torch.einsum("dy,nocy->nocd", A_T, out)  # (N_tiles, C_out, 2, 2)

    # Reshape back to NCHW
    out = out.view(N, tile_H, tile_W, C_out, 2, 2)
    out = out.permute(0, 3, 1, 4, 2, 5).contiguous()
    out = out.view(N, C_out, H_out, W_out)
    return out


def winograd_theoretical_speedup() -> float:
    """Compute the theoretical FLOP reduction of F(2,3) vs naive.

    Naive: m² · r² = 4 · 9 = 36 multiplies per tile
    Winograd: (m+r-1)² = 16 multiplies in transform domain
    Speedup = 36 / 16 = 2.25
    """
    m, r = 2, 3
    naive_mults = m * m * r * r
    wino_mults = (m + r - 1) * (m + r - 1)
    return naive_mults / wino_mults  # 2.25


# ===========================================================================
# 4. OPERATOR FUSION: Conv + BatchNorm + ReLU
# ===========================================================================


def fuse_conv_bn_relu(
    conv_w: torch.Tensor,
    conv_b: torch.Tensor | None,
    bn_weight: torch.Tensor,
    bn_bias: torch.Tensor,
    bn_running_mean: torch.Tensor,
    bn_running_var: torch.Tensor,
    bn_eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fold BatchNorm parameters into Conv weights and bias.

    After training, batch-normalisation is a linear transform at inference:
        y = γ * (x - μ) / √(σ² + ε) + β

    This can be absorbed into the preceding convolution:
        W_fused = W * γ / √(σ² + ε)
        b_fused = β + γ * (b - μ) / √(σ² + ε)

    Combined with ReLU (max(0, x)), we get a single fused operation.

    Parameters
    ----------
    conv_w : (C_out, C_in, kH, kW)
    conv_b : (C_out,) or None
    bn_weight, bn_bias : (C_out,)
    bn_running_mean, bn_running_var : (C_out,)
    bn_eps : float

    Returns
    -------
    fused_w : (C_out, C_in, kH, kW)
    fused_b : (C_out,)
    """
    gamma = bn_weight
    beta = bn_bias
    mu = bn_running_mean
    sigma = torch.sqrt(bn_running_var + bn_eps)

    scale = gamma / sigma
    bias_correction = beta - mu * scale

    # Broadcast scale over all but the output-channel dim
    scale_4d = scale.view(-1, 1, 1, 1)
    fused_w = conv_w * scale_4d

    if conv_b is not None:
        fused_b = conv_b * scale + bias_correction
    else:
        fused_b = bias_correction

    return fused_w, fused_b


def fused_conv_bn_relu_forward(
    x: torch.Tensor,
    fused_w: torch.Tensor,
    fused_b: torch.Tensor,
) -> torch.Tensor:
    """Single-pass Conv+BN+ReLU using pre-fused parameters.

    This avoids the separate BN normalisation and ReLU activation calls,
    reducing memory bandwidth and kernel-launch overhead.
    """
    y = F.conv2d(x, fused_w, fused_b, stride=STRIDE, padding=PADDING)
    return F.relu(y)


# ===========================================================================
# 5. MEMORY LAYOUT: NCHW vs NHWC
# ===========================================================================


def benchmark_layout_access():
    """Compare channel-first (NCHW) vs channel-last (NHWC) access efficiency.

    NHWC (channels-last) often yields better cache utilisation on CPUs
    because consecutive elements belong to the same spatial location
    across channels, making vectorised operations more efficient.

    We benchmark two patterns:
      1. Channel reduction (mean over C dim) – favours NHWC
      2. Spatial reduction (mean over H, W) – favours NCHW
    """
    size = BATCH * IN_C * H * W
    x_nchw = torch.randn(BATCH, IN_C, H, W, device=DEVICE, dtype=DTYPE)
    x_nhwc = x_nchw.contiguous(memory_format=torch.channels_last)

    # Operation 1: reduce over channel dimension (per-pixel mean)
    # NHWC has channels contiguous, so this should be faster
    def _chan_reduce_nchw():
        return x_nchw.mean(dim=1)  # reduce over dim 1 (channels)

    def _chan_reduce_nhwc():
        return x_nhwc.mean(dim=3)  # reduce over dim 3 (channels-last)

    # Operation 2: reduce over spatial dimensions (per-channel mean)
    # NCHW has spatial dims contiguous, so this should be faster
    def _spatial_reduce_nchw():
        return x_nchw.mean(dim=(2, 3))

    def _spatial_reduce_nhwc():
        return x_nhwc.mean(dim=(1, 2))

    t_cr_nchw, _ = _timeit(_chan_reduce_nchw)
    t_cr_nhwc, _ = _timeit(_chan_reduce_nhwc)
    t_sr_nchw, _ = _timeit(_spatial_reduce_nchw)
    t_sr_nhwc, _ = _timeit(_spatial_reduce_nhwc)

    return {
        "NCHW  (chan-reduce)": (t_cr_nchw, 0.0),
        "NHWC  (chan-reduce)": (t_cr_nhwc, 0.0),
        "NCHW  (spatial-reduce)": (t_sr_nchw, 0.0),
        "NHWC  (spatial-reduce)": (t_sr_nhwc, 0.0),
    }, size


# ===========================================================================
# 6. MAIN BENCHMARK & COMPARISON TABLE
# ===========================================================================


def _compute_flops(
    n: int, c_in: int, c_out: int, h_out: int, w_out: int, k: int
) -> float:
    """Total multiply-add operations for a single conv2d layer."""
    return float(2 * n * c_out * c_in * h_out * w_out * k * k)


def main() -> None:
    print("=" * 72)
    print("  TinyEngine Optimization Simulation – Lecture 11")
    print("=" * 72)
    print(
        f"\nConfiguration: N={BATCH} C={IN_C}→{OUT_C} H×W={H}×{W}  K={KERNEL} S={STRIDE} P={PADDING}"
    )
    print(f"Device: {DEVICE}    dtype: {DTYPE}")
    print()

    x, w = _make_tensors()
    H_out = W_out = H  # pad=1, stride=1 keeps spatial dims
    ops = _compute_flops(BATCH, IN_C, OUT_C, H_out, W_out, KERNEL)

    # ---- Verify correctness ----
    print("Verifying correctness against baseline...")
    ref = naive_conv2d(x, w)
    im2col_out = im2col_conv2d(x, w)
    winograd_out = winograd_f23_conv2d(x, w)

    # im2col should match closely
    im2col_diff = (ref - im2col_out).abs().max().item()
    print(f"  im2col   max diff: {im2col_diff:.2e}")

    # Winograd may differ due to floating-point order; check relative error
    wino_diff = (ref - winograd_out).abs().max().item()
    wino_rel = wino_diff / ref.abs().max().item() if ref.abs().max().item() > 0 else 0.0
    print(f"  Winograd max diff: {wino_diff:.2e}  (relative: {wino_rel:.2e})")

    im2col_ok = im2col_diff < 1e-3
    winograd_ok = wino_rel < 0.02  # Winograd introduces more numerical error
    print(f"  im2col   correctness: {'PASS' if im2col_ok else 'FAIL'}")
    print(f"  Winograd correctness: {'PASS' if winograd_ok else 'FAIL'}")
    print()

    # ---- Benchmark convolution methods ----
    print("Benchmarking convolution implementations...")
    results: dict[str, Tuple[float, float, float]] = {}

    t_naive, s_naive = _timeit(lambda: naive_conv2d(x, w))
    results["Naive (F.conv2d)"] = (t_naive, s_naive, _gflops(t_naive, ops))

    t_im2col, s_im2col = _timeit(lambda: im2col_conv2d(x, w))
    results["im2col + GEMM"] = (t_im2col, s_im2col, _gflops(t_im2col, ops))

    t_wino, s_wino = _timeit(lambda: winograd_f23_conv2d(x, w))
    results["Winograd F(2,3)"] = (t_wino, s_wino, _gflops(t_wino, ops))

    # ---- Compute speedups ----
    speedup_im2col = t_naive / t_im2col
    speedup_wino = t_naive / t_wino
    speedup_theory = winograd_theoretical_speedup()

    # ---- Operator fusion benchmark ----
    print("Benchmarking operator fusion...")
    # Create mock BN parameters
    rng = torch.Generator(device=DEVICE).manual_seed(123)
    bn_w = torch.randn(OUT_C, device=DEVICE, dtype=DTYPE, generator=rng)
    bn_b = torch.randn(OUT_C, device=DEVICE, dtype=DTYPE, generator=rng)
    bn_mean = torch.randn(OUT_C, device=DEVICE, dtype=DTYPE, generator=rng) * 0.1
    bn_var = (
        torch.abs(torch.randn(OUT_C, device=DEVICE, dtype=DTYPE, generator=rng)) * 0.5
        + 0.5
    )
    conv_bias = torch.randn(OUT_C, device=DEVICE, dtype=DTYPE, generator=rng)

    # Fuse parameters
    fused_w, fused_b = fuse_conv_bn_relu(w, conv_bias, bn_w, bn_b, bn_mean, bn_var)

    def _unfused_forward():
        y = F.conv2d(x, w, conv_bias, stride=STRIDE, padding=PADDING)
        # BN in eval mode
        y = (y - bn_mean.view(1, -1, 1, 1)) / torch.sqrt(
            bn_var.view(1, -1, 1, 1) + 1e-5
        )
        y = y * bn_w.view(1, -1, 1, 1) + bn_b.view(1, -1, 1, 1)
        return F.relu(y)

    def _fused_forward():
        return fused_conv_bn_relu_forward(x, fused_w, fused_b)

    t_unfused, s_unfused = _timeit(_unfused_forward)
    t_fused, s_fused = _timeit(_fused_forward)
    speedup_fusion = t_unfused / t_fused

    # Verify fusion correctness numerically
    unfused_out = _unfused_forward()
    fused_out = _fused_forward()
    fusion_diff = (unfused_out - fused_out).abs().max().item()
    print(
        f"  Fusion correctness: {'PASS' if fusion_diff < 1e-3 else 'FAIL'}  "
        f"(max diff: {fusion_diff:.2e})"
    )

    # ---- Memory layout benchmark ----
    print("Benchmarking memory layouts...")
    layout_results, layout_size = benchmark_layout_access()
    print()

    # =========================================================================
    # Print comparison tables
    # =========================================================================

    # --- Table 1: Conv implementations ---
    print("=" * 72)
    print("  TABLE 1: Convolution Implementation Comparison")
    print("=" * 72)
    print(
        f"  {'Method':<22s} {'Time (ms)':>12s} {'Std (ms)':>10s} {'GFLOPS':>10s} {'vs Naive':>10s}"
    )
    print("  " + "-" * 66)
    for name, (mean, std, gflops) in results.items():
        ratio = t_naive / mean
        print(
            f"  {name:<22s} {mean:>10.3f}  {std:>8.3f}  {gflops:>8.1f}  {ratio:>8.2f}×"
        )
    print()

    # --- Table 2: Winograd theoretical ---
    print("=" * 72)
    print("  TABLE 2: Winograd F(2,3) – Theoretical Analysis")
    print("=" * 72)
    print(f"  Theoretical arithmetic reduction:  {speedup_theory:.2f}×")
    print(f"  Measured speedup vs naive:         {speedup_wino:.2f}×")
    print(f"  Note: Winograd trades fewer multiplies for more additions")
    print(f"        and higher memory pressure from intermediate tiles.")
    print()

    # --- Table 3: Operator fusion ---
    print("=" * 72)
    print("  TABLE 3: Operator Fusion – Conv + BN + ReLU")
    print("=" * 72)
    print(f"  {'Variant':<22s} {'Time (ms)':>12s} {'Std (ms)':>10s} {'Speedup':>10s}")
    print("  " + "-" * 56)
    print(
        f"  {'Unfused (3 ops)':<22s} {t_unfused:>10.3f}  {s_unfused:>8.3f}  {'1.00× (baseline)':>14s}"
    )
    print(
        f"  {'Fused (1 op)':<22s} {t_fused:>10.3f}  {s_fused:>8.3f}  {speedup_fusion:>8.2f}×"
    )
    print()
    print("  Fusion transforms:")
    print("    W_fused = W_conv * γ / √(σ² + ε)")
    print("    b_fused = β + γ * (b_conv - μ) / √(σ² + ε)")
    print("  This eliminates two intermediate tensor reads/writes and")
    print("  two kernel launches at inference time.")
    print()

    # --- Table 4: Memory layout ---
    print("=" * 72)
    print("  TABLE 4: Memory Layout – NCHW vs NHWC")
    print("=" * 72)
    print(f"  Tensor size: {layout_size:,} elements ({layout_size * 4 / 1024:.0f} KiB)")
    print()
    print(f"  {'Layout / Operation':<30s} {'Time (ms)':>12s} {'Rel. Winner':>12s}")
    print("  " + "-" * 56)

    t_cr_nchw = layout_results["NCHW  (chan-reduce)"][0]
    t_cr_nhwc = layout_results["NHWC  (chan-reduce)"][0]
    t_sr_nchw = layout_results["NCHW  (spatial-reduce)"][0]
    t_sr_nhwc = layout_results["NHWC  (spatial-reduce)"][0]

    cr_speedup = t_cr_nchw / max(t_cr_nhwc, 1e-9)
    sr_speedup = t_sr_nhwc / max(t_sr_nchw, 1e-9)
    print(f"  {'NCHW  (channel-mean)':<30s} {t_cr_nchw:>10.3f}  {'baseline':>12s}")
    winner_cr = (
        "NHWC wins"
        if cr_speedup > 1.01
        else "NCHW wins"
        if cr_speedup < 0.99
        else "tie"
    )
    print(
        f"  {'NHWC  (channel-mean)':<30s} {t_cr_nhwc:>10.3f}  {winner_cr:>12s}"
        f"  ({cr_speedup:.2f}x)"
    )
    print(f"  {'NCHW  (spatial-mean)':<30s} {t_sr_nchw:>10.3f}  {'baseline':>12s}")
    winner_sr = (
        "NCHW wins"
        if sr_speedup > 1.01
        else "NHWC wins"
        if sr_speedup < 0.99
        else "tie"
    )
    print(
        f"  {'NHWC  (spatial-mean)':<30s} {t_sr_nhwc:>10.3f}  {winner_sr:>12s}"
        f"  ({sr_speedup:.2f}x)"
    )
    print()
    print("  Insight:")
    print("    - On this PyTorch CPU backend, NCHW wins for simple reductions")
    print("      because oneDNN kernels are heavily optimised for NCHW.")
    print("    - The theory predicts NHWC should favour channel-heavy access")
    print("      (channels contiguous in memory → better cache line usage).")
    print("    - In practice, NHWC benefits appear most clearly with custom")
    print("      kernels (e.g. depthwise conv) and on mobile/embedded CPUs")
    print("      where SIMD width matches the channel dimension.")
    print("    - Inference engines (TFLite, MNN, ncnn) adopt NHWC because")
    print("      they implement their own optimised kernels for this layout.")
    print()

    # --- Summary ---
    print("=" * 72)
    print("  Optimization Summary (higher is better)")
    print("=" * 72)
    print(f"  Winograd theoretical speedup (vs naive):   {speedup_theory:.2f}×")
    print(f"  Winograd measured speedup (vs naive):      {speedup_wino:.2f}×")
    print(f"  Operator fusion speedup (vs unfused):      {speedup_fusion:.2f}×")
    chan_ratio = t_cr_nchw / max(t_cr_nhwc, 1e-9)
    print(f"  NHWC channel-reduce speedup vs NCHW:       {chan_ratio:.2f}×")
    print()
    print("  Key takeaways:")
    print("  1. im2col enables GEMM-based conv but may increase memory footprint.")
    print("  2. Winograd reduces arithmetic at the cost of numerical precision")
    print("     and increased memory for transform matrices.")
    print("  3. Operator fusion eliminates intermediate buffers and kernel")
    print("     launches — crucial for tiny devices with limited bandwidth.")
    print("  4. Channels-last (NHWC) layout improves cache locality on CPUs,")
    print("     especially for depthwise and pointwise convolutions.")
    print()


if __name__ == "__main__":
    main()
