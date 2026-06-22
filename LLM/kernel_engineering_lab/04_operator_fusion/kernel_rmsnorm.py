"""
RMSNorm Triton kernel (used in LLaMA, Mistral, Gemma).

RMSNorm (Root Mean Square Normalization) has replaced LayerNorm in most
modern LLMs because it's simpler (no mean computation, no bias) and
thus faster while performing similarly.

RMSNorm(x) = x * weight / sqrt(mean(x^2) + eps)

This kernel implements the full RMSNorm including learnable weight.
Handles the reduction (sum of squares) carefully using warp-level reduction.

Reference: https://arxiv.org/abs/1910.07467
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

EPS = 1e-6


@triton.jit
def _rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    n_cols: int,
    BLOCK_SIZE: tl.constexpr,
):
    """RMSNorm kernel: out = x * weight * rsqrt(mean(x^2) + eps).

    Each program handles one row. Uses tl.sum for reduction within the block.
    """
    row_idx = tl.program_id(axis=0)
    row_start = row_idx * n_cols
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < n_cols

    # Load x row
    x = tl.load(x_ptr + row_start + col_offsets, mask=col_mask, other=0.0)

    # Compute mean of squares
    x_sq = x * x
    mean_sq = tl.sum(x_sq, axis=0) / n_cols

    # Compute rsqrt(mean_sq + eps)
    rstd = 1.0 / tl.sqrt(mean_sq + EPS)

    # Normalize
    normalized = x * rstd

    # Apply learnable weight (elementwise)
    weight = tl.load(weight_ptr + col_offsets, mask=col_mask, other=0.0)
    out = normalized * weight

    tl.store(out_ptr + row_start + col_offsets, out, mask=col_mask)


def triton_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    block_size: int = 1024,
) -> torch.Tensor:
    """Full RMSNorm implementation in Triton.

    Args:
        x: Input tensor (CUDA, 2D [rows, cols]).
        weight: Learnable weight (CUDA, 1D [cols]).
        block_size: Elements per program (should be >= n_cols).

    Returns:
        RMSNorm(x) with learnable weight, same shape as x.
    """
    assert x.is_cuda and weight.is_cuda, "Tensors must be on CUDA"
    assert x.dim() == 2, f"Expected 2D input, got {x.dim()}D"
    assert weight.dim() == 1, f"Expected 1D weight, got {weight.dim()}D"
    assert x.shape[-1] == weight.shape[0], (
        f"Last dim of x ({x.shape[-1]}) must match weight dim ({weight.shape[0]})"
    )

    out = torch.empty_like(x)
    n_rows, n_cols = x.shape

    assert block_size >= n_cols, f"block_size ({block_size}) must be >= n_cols ({n_cols})"

    grid = (n_rows,)
    _rmsnorm_kernel[grid](x, weight, out, n_cols, BLOCK_SIZE=block_size)
    return out


def torch_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """RMSNorm implemented with pure PyTorch ops for comparison."""
    x_float = x.float()
    rms = torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + EPS)
    return (x_float * rms).to(x.dtype) * weight


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping demo.")
    else:
        B, D = 4, 1024
        x = torch.randn(B, D, device="cuda")
        weight = torch.ones(D, device="cuda")

        y_triton = triton_rmsnorm(x, weight)
        y_torch = torch_rmsnorm(x, weight)

        err = (y_triton - y_torch).abs().max().item()
        print(f"Triton RMSNorm - max error vs PyTorch: {err:.2e}")
