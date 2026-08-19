"""Quantization fundamentals: quant/dequant and granularity.

Implements the symmetric integer quantization formula and the four common
granularities, then measures the round-trip error of each on a weight matrix
with per-channel amplitude differences (the "outlier channel" regime that
motivates per-channel and group-wise quantization).

    quant:  x_q = clamp(round(x / scale), -127, 127)
    dequant: x ~= scale * x_q

The scales are computed from the data (calibration-free, symmetric min/max):
per-tensor uses one scale for the whole tensor, per-channel one scale per
column, per-token one per row, per-group one per chunk of ``group_size`` along
the column axis.  The error metric is both max-abs and MSE, so the tradeoff
between coarse scales (cheap) and fine scales (accurate) is visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import torch

TensorFactory = Callable[[torch.device, torch.dtype], torch.Tensor]


def quantize_symmetric(x: torch.Tensor, scale: torch.Tensor, qmax: int = 127) -> torch.Tensor:
    """Quantize x with a per-element scale broadcastable to x's shape.

    ``qmax`` selects the precision: 127 for int8, 7 for int4, 3 for int3 (the
    values are still stored in int8).  Lower qmax means fewer quantization
    levels and therefore a coarser step, which is exactly where scale choice
    (and clipping) starts to matter.
    """
    q = torch.clamp(torch.round(x / scale), -qmax, qmax).to(torch.int8)
    return q


def dequantize_symmetric(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale


@dataclass
class QuantResult:
    granularity: str
    scale_shape: str
    max_abs_err: float
    mse: float
    # extra bytes to store the scales per element (0 for per-tensor)
    scale_bytes_per_element: float


def _measure(x: torch.Tensor, scale: torch.Tensor, granularity: str) -> QuantResult:
    q = quantize_symmetric(x, scale)
    x_hat = dequantize_symmetric(q, scale)
    err = (x_hat - x).float()
    max_abs = err.abs().max().item()
    mse = (err ** 2).mean().item()
    scale_bytes = scale.numel() * 4 / x.numel()  # fp32 scales, amortized per element
    return QuantResult(
        granularity=granularity,
        scale_shape=tuple(scale.shape),
        max_abs_err=max_abs,
        mse=mse,
        scale_bytes_per_element=scale_bytes,
    )


def granularity_error(x: torch.Tensor, group_size: int = 128) -> List[QuantResult]:
    """Compute quant/dequant error for per-tensor / channel / token / group."""
    results: List[QuantResult] = []

    # per-tensor: one scalar scale.
    scale = (x.abs().max() / 127.0).detach()
    results.append(_measure(x, scale, "per-tensor"))

    # per-channel: one scale per column.
    scale = x.abs().max(dim=0, keepdim=True).values / 127.0  # (1, N)
    results.append(_measure(x, scale, "per-channel"))

    # per-token: one scale per row.
    scale = x.abs().max(dim=1, keepdim=True).values / 127.0  # (K, 1)
    results.append(_measure(x, scale, "per-token"))

    # per-group: one scale per group_size chunk along columns.
    K, N = x.shape
    grouped = x.view(K, N // group_size, group_size)
    scale = grouped.abs().max(dim=2, keepdim=True).values / 127.0  # (K, N/group, 1)
    scale = scale.expand(K, N // group_size, group_size).reshape(K, N)
    results.append(_measure(x, scale, f"per-group({group_size})"))

    return results


def make_outlier_weight(K: int = 1024, N: int = 1024, outlier_cols: int = 8):
    """Weight matrix where a few columns have much larger amplitude.

    This models the real LLM "activation outlier channel" problem: a handful of
    channels dominate the range, so a per-tensor scale is driven by them and
    loses precision on the rest.
    """

    def build(device, dtype):
        w = torch.randn(K, N, device=device, dtype=torch.float32) * 0.05
        # A few columns are ~100x larger.
        idx = torch.randperm(N, device=device)[:outlier_cols]
        w[:, idx] = torch.randn(K, outlier_cols, device=device) * 5.0
        return w

    return build
