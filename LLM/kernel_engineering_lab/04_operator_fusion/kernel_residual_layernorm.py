"""
Fused Residual + LayerNorm simplified Triton kernel.

Industrial: Appears in every transformer block.
  residual = x + f(x)  (where f is attention or FFN)
  output = LayerNorm(residual)

Without fusion: 2 intermediate tensors (residual, normalized output) written to
global memory. With fusion: only the final output is written.

This is a simplified version without learnable affine parameters (gamma, beta)
for clarity. The fusion pattern is the key concept.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

EPS = 1e-5


@triton.jit
def _residual_layernorm_fused_kernel(
    x_ptr,
    residual_ptr,
    out_ptr,
    n_cols: int,
    n_elements: int,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused residual + LayerNorm.

    For each row:
      1. residual[i] = x[i] + residual_ptr[i] (elementwise)
      2. Compute mean and variance of residual[i]
      3. Normalize: out[i] = (residual[i] - mean) / sqrt(var + eps)
    """
    row_idx = tl.program_id(axis=0)
    row_start = row_idx * n_cols
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # Load x and residual, compute sum
    x_part = tl.load(x_ptr + row_start + col_offsets, mask=col_offsets < n_cols, other=0.0)
    res_part = tl.load(residual_ptr + row_start + col_offsets, mask=col_offsets < n_cols, other=0.0)

    combined = x_part + res_part

    # Compute mean: sum all elements in this row / n_cols
    total = tl.sum(combined, axis=0)
    mean = total / n_cols

    # Compute variance: sum((x - mean)^2) / n_cols
    centered = combined - mean
    variance = tl.sum(centered * centered, axis=0) / n_cols

    # Normalize
    rstd = 1.0 / tl.sqrt(variance + EPS)
    normalized = centered * rstd

    # Store result
    tl.store(out_ptr + row_start + col_offsets, normalized, mask=col_offsets < n_cols)


def fused_residual_layernorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    block_size: int = 1024,
) -> torch.Tensor:
    """Fused residual + LayerNorm.

    Args:
        x: Input tensor (CUDA, 2D [rows, cols]).
        residual: Residual tensor (CUDA, same shape as x).
        block_size: Elements per program (should be >= cols).

    Returns:
        LayerNorm(x + residual) without learnable parameters.
    """
    assert x.is_cuda and residual.is_cuda, "Tensors must be on CUDA"
    assert x.shape == residual.shape, f"Shape mismatch: {x.shape} vs {residual.shape}"
    assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"

    out = torch.empty_like(x)
    n_rows, n_cols = x.shape
    n_elements = x.numel()

    assert block_size >= n_cols, f"block_size ({block_size}) must be >= n_cols ({n_cols})"

    grid = (n_rows,)
    _residual_layernorm_fused_kernel[grid](
        x, residual, out, n_cols, n_elements, BLOCK_SIZE=block_size
    )
    return out


def sequential_residual_layernorm(
    x: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    """Sequential residual + LayerNorm: materializes intermediate tensors."""
    combined = x + residual
    return torch.nn.functional.layer_norm(
        combined, normalized_shape=[combined.shape[-1]], weight=None, bias=None
    )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping demo.")
    else:
        B, D = 4, 1024
        x = torch.randn(B, D, device="cuda")
        residual = torch.randn(B, D, device="cuda")

        y_fused = fused_residual_layernorm(x, residual)
        y_seq = sequential_residual_layernorm(x, residual)

        err = (y_fused - y_seq).abs().max().item()
        print(f"Fused Residual+LayerNorm - max error vs sequential: {err:.2e}")
