"""
Triton kernel for element-wise vector addition.

Demonstrates the fundamental Triton pattern:
  1. Compute program_id to determine which data slice this program handles
  2. Calculate offsets using block size and program_id
  3. Load data with a mask for edge elements
  4. Compute
  5. Store result with mask

Run as script to see a quick demonstration:
    python 02_triton_basics/triton_vector_add.py
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _vector_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements: int,
    BLOCK_SIZE: tl.constexpr,
):
    """Element-wise vector addition: out = x + y."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_vector_add(
    x: torch.Tensor,
    y: torch.Tensor,
    block_size: int = 1024,
) -> torch.Tensor:
    """
    Element-wise vector addition using Triton kernel.

    Args:
        x: First input tensor (CUDA, any shape).
        y: Second input tensor (CUDA, same shape as x).
        block_size: Threads per block (must be a power of 2, <= 1024).

    Returns:
        Tensor of same shape as x, containing x + y.
    """
    assert x.shape == y.shape, f"Shape mismatch: {x.shape} vs {y.shape}"
    assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA device"

    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _vector_add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=block_size)
    return out


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping demo.")
    else:
        n = 1024
        a = torch.randn(n, device="cuda")
        b = torch.randn(n, device="cuda")
        c_triton = triton_vector_add(a, b)
        c_torch = a + b
        print(f"Input shape: {a.shape}")
        print(f"Max error vs torch.add: {(c_triton - c_torch).abs().max().item():.2e}")
        print("Vector add demo passed.")
