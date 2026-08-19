"""SmoothQuant: migrate activation outliers to the weight.

LLM activations have "outlier channels" - a few input channels whose magnitude
is ~100x the rest.  A per-tensor int8 scale is driven by those outliers, which
destroys precision on every other channel.  SmoothQuant's insight: the outlier
lives in the *product* X @ W, so we can push a per-channel smoothing factor
from the activation into the weight without changing the result:

    Y = X @ W = (X @ diag(s)) @ (diag(s)^-1 @ W) = X_hat @ W_hat

with s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha).  After the migration both
X_hat and W_hat are smooth, so per-tensor int8 works again.  This module
measures the output error of int8 quantization with and without the migration.
"""

from __future__ import annotations

import torch

from compression.quantization.quantize import dequantize_symmetric, quantize_symmetric


def make_outlier_activation(M: int, K: int, outlier_channels: int = 8):
    """Activation matrix where a few channels (columns) are ~100x larger."""

    def build(device, dtype):
        x = torch.randn(M, K, device=device, dtype=torch.float32) * 0.01
        idx = torch.randperm(K, device=device)[:outlier_channels]
        x[:, idx] = torch.randn(M, outlier_channels, device=device) * 1.0
        return x

    return build


def smooth_scale(x: torch.Tensor, w: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """Per-input-channel smoothing factor s in R^K.

    s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha).  The migration divides X by
    s and multiplies W by s, so an outlier channel (large max|X_j|) shrinks in
    X_hat and grows in W_hat - pushing the range out of the activation.
    """
    max_x = x.abs().amax(dim=0)  # (K,) max over batch/seq
    max_w = w.abs().amax(dim=1)  # (K,) max over output channels
    return (max_x ** alpha) / (max_w ** (1 - alpha) + 1e-8)


def quantize_int8_tensor(x: torch.Tensor) -> torch.Tensor:
    """Per-tensor symmetric int8 quantization (single scalar scale)."""
    scale = (x.abs().max() / 127.0).detach()
    return quantize_symmetric(x, scale), scale


def smoothquant_experiment(x: torch.Tensor, w: torch.Tensor, alpha: float = 0.5) -> dict:
    """Compare int8 output error with and without SmoothQuant migration."""
    y_ref = x @ w  # fp32 reference

    # Baseline: per-tensor int8 on both x and w.
    q_x, sx = quantize_int8_tensor(x)
    q_w, sw = quantize_int8_tensor(w)
    y_direct = dequantize_symmetric(q_x, sx) @ dequantize_symmetric(q_w, sw)
    err_direct = (y_direct - y_ref).abs().max().item()

    # SmoothQuant: divide X by s (outlier shrinks) and multiply W by s.
    s = smooth_scale(x, w, alpha)
    x_hat = x / s[None, :]
    w_hat = w * s[:, None]
    q_xh, sxh = quantize_int8_tensor(x_hat)
    q_wh, swh = quantize_int8_tensor(w_hat)
    y_smooth = dequantize_symmetric(q_xh, sxh) @ dequantize_symmetric(q_wh, swh)
    err_smooth = (y_smooth - y_ref).abs().max().item()

    # How much did the activation range shrink after the migration?
    range_ratio = (x_hat.abs().max() / x.abs().max()).item()

    return {
        "direct_max_abs_err": err_direct,
        "smooth_max_abs_err": err_smooth,
        "error_reduction_x": err_direct / (err_smooth + 1e-12),
        "activation_range_ratio_after": range_ratio,
    }
