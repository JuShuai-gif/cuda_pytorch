"""AWQ: activation-aware weight quantization.

The core idea of AWQ is that not all weight channels matter equally, and that
the naive per-channel ``max(|W_j|)/127`` scale is not optimal when a weight
channel contains a few very large elements: the max-based scale is dragged up
by the outliers, coarsely quantizing everything else in that channel.  AWQ
searches a per-channel scale that *may clip the outliers* to reduce the overall
error, and weights the search by activation saliency so important channels are
protected.

This module implements a simplified AWQ: per-channel grid search over a
multiplier in [0.4, 1.0] (clip allowed), minimizing each channel's quant error,
then reports the saliency-weighted total error vs the naive max-based scale.
"""

from __future__ import annotations

import torch

from compression.quantization.quantize import dequantize_symmetric, quantize_symmetric


def activation_saliency(x: torch.Tensor) -> torch.Tensor:
    """Per-input-channel importance = mean |activation| over the batch."""
    return x.abs().mean(dim=0)  # (K,)


def make_outlier_weight(K: int = 1024, N: int = 1024, outlier_channels: int = 16):
    """Weight matrix where a few channels contain a few large elements.

    The large elements drag up a max-based per-channel scale, so naive
    quantization is coarse for those channels; AWQ clips them instead.  The
    effect is only visible at low precision (int4/int3): int8 has 127 levels
    and a max-based scale is already near-optimal for a Gaussian channel.
    """

    def build(device, dtype):
        w = torch.randn(K, N, device=device, dtype=torch.float32) * 0.05
        idx = torch.randperm(K, device=device)[:outlier_channels]
        for j in idx:
            cols = torch.randperm(N, device=device)[:8]
            w[j, cols] = torch.randn(8, device=device).abs() * 0.2  # ~4x normal
        return w

    return build


def _channel_error(w: torch.Tensor, scale: torch.Tensor, qmax: int) -> torch.Tensor:
    q = quantize_symmetric(w, scale, qmax)
    w_hat = dequantize_symmetric(q, scale)
    return ((w_hat - w) ** 2).sum(dim=1)  # (K,)


def awq_experiment(x: torch.Tensor, w: torch.Tensor, qmax: int = 7, n_grid: int = 15) -> dict:
    """Naive max-based per-channel scale vs saliency-weighted AWQ search."""
    K = w.shape[0]
    saliency = activation_saliency(x)  # (K,)
    naive_scale = w.abs().max(dim=1, keepdim=True).values / qmax  # (K, 1)

    # Per-channel grid search, allowing clipping (multiplier < 1).
    best_scales = naive_scale.clone()
    multipliers = torch.linspace(0.3, 1.0, n_grid, device=w.device)
    for j in range(K):
        best_err = float("inf")
        for m in multipliers:
            s = naive_scale[j] * m
            q = quantize_symmetric(w[j:j + 1], s, qmax)
            w_hat = dequantize_symmetric(q, s)
            err = ((w_hat - w[j:j + 1]) ** 2).sum().item()
            if err < best_err:
                best_err = err
                best_scales[j] = s

    naive_err = _channel_error(w, naive_scale, qmax)
    awq_err = _channel_error(w, best_scales, qmax)
    naive_weighted = (saliency * naive_err).sum().item()
    awq_weighted = (saliency * awq_err).sum().item()

    return {
        "qmax": qmax,
        "naive_weighted_error": naive_weighted,
        "awq_weighted_error": awq_weighted,
        "error_reduction_x": naive_weighted / (awq_weighted + 1e-12),
        "mean_multiplier": (best_scales / naive_scale).mean().item(),
    }
