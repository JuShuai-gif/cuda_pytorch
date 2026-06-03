"""
Triton memory copy kernels demonstrating bandwidth optimization.

Kernels:
  - copy_kernel: Simple memory copy (1 element per thread)
  - copy_vectorized: Vectorized copy (4 elements per thread)
  - copy_non_contiguous: Strided copy showing bandwidth degradation

These demonstrate that Triton can match peak bandwidth for simple copies.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Simple copy kernel
# ---------------------------------------------------------------------------


@triton.jit
def _copy_kernel(
    src_ptr,
    dst_ptr,
    n_elements: int,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple memory copy: 1 element per thread load/store."""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    val = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + offsets, val, mask=mask)


def copy_kernel(
    src: torch.Tensor,
    block_size: int = 1024,
) -> torch.Tensor:
    """Copy a tensor using a Triton kernel (1 element per thread).

    Args:
        src: Source tensor on CUDA.
        block_size: Number of elements per program.

    Returns:
        A new tensor with the same data as src.
    """
    assert src.is_cuda, "Source tensor must be on CUDA"
    dst = torch.empty_like(src)
    n_elements = src.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _copy_kernel[grid](src, dst, n_elements, BLOCK_SIZE=block_size)
    return dst


# ---------------------------------------------------------------------------
# Vectorized copy kernel
# ---------------------------------------------------------------------------


@triton.jit
def _copy_vectorized_kernel(
    src_ptr,
    dst_ptr,
    n_elements: int,
    VEC_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Vectorized copy: VEC_SIZE elements per thread."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * VEC_SIZE

    for elem_idx in tl.static_range(VEC_SIZE):
        offsets = block_start + elem_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        val = tl.load(src_ptr + offsets, mask=mask)
        tl.store(dst_ptr + offsets, val, mask=mask)


def copy_vectorized(
    src: torch.Tensor,
    vec_size: int = 4,
    block_size: int = 256,
) -> torch.Tensor:
    """Copy a tensor using a vectorized Triton kernel.

    Each program launches VEC_SIZE blocks of BLOCK_SIZE elements,
    allowing more elements per dispatch and better memory throughput.

    Args:
        src: Source tensor on CUDA.
        vec_size: Vector width (elements per dispatch).
        block_size: Elements per inner block.

    Returns:
        Copy of src.
    """
    assert src.is_cuda, "Source tensor must be on CUDA"
    dst = torch.empty_like(src)
    n_elements = src.numel()
    elements_per_program = block_size * vec_size
    grid = lambda meta: (triton.cdiv(n_elements, elements_per_program),)
    _copy_vectorized_kernel[grid](src, dst, n_elements, VEC_SIZE=vec_size, BLOCK_SIZE=block_size)
    return dst


# ---------------------------------------------------------------------------
# Non-contiguous (strided) copy kernel
# ---------------------------------------------------------------------------


@triton.jit
def _copy_non_contiguous_kernel(
    src_ptr,
    dst_ptr,
    n_elements: int,
    STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Strided copy: elements are separated by STRIDE in source."""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Source access pattern is strided
    src_offsets = offsets * STRIDE
    mask = offsets < n_elements

    val = tl.load(src_ptr + src_offsets, mask=mask)
    tl.store(dst_ptr + offsets, val, mask=mask)


def copy_non_contiguous(
    src: torch.Tensor,
    stride: int,
    block_size: int = 1024,
) -> torch.Tensor:
    """Copy with strided access pattern from source.

    This simulates copying from a non-contiguous view (e.g., transposed,
    sliced) where each consecutive element in source is stride elements
    apart.

    Args:
        src: Source tensor on CUDA (must be large enough for stride * n_copy).
        stride: Number of elements between consecutive reads.
        block_size: Elements per program.

    Returns:
        A tensor containing strided copies from src.
    """
    assert src.is_cuda, "Source tensor must be on CUDA"
    n_copy = src.numel() // stride
    if n_copy < 1:
        n_copy = src.numel()
    dst = torch.empty(n_copy, device=src.device, dtype=src.dtype)
    grid = lambda meta: (triton.cdiv(n_copy, meta["BLOCK_SIZE"]),)
    _copy_non_contiguous_kernel[grid](src, dst, n_copy, STRIDE=stride, BLOCK_SIZE=block_size)
    return dst


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping demo.")
    else:
        n = 2**20
        x = torch.randn(n, device="cuda", dtype=torch.float32)

        c1 = copy_kernel(x)
        c2 = copy_vectorized(x, vec_size=4)
        assert torch.equal(c1, x), "Simple copy failed"
        assert torch.equal(c2, x), "Vectorized copy failed"

        # Strided copy
        stride = 8
        x_large = torch.randn(n * stride, device="cuda", dtype=torch.float32)
        c3 = copy_non_contiguous(x_large, stride=stride)
        assert torch.equal(c3, x_large[::stride]), "Strided copy failed"

        print(f"All copy kernels produce correct output (n={n:,}).")
