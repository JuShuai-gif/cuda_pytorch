"""
Tiled Triton matmul with shared memory.

Each program loads a BLOCK_M x BLOCK_K tile of A and BLOCK_K x BLOCK_N tile
of B into shared memory, then accumulates using tl.dot. Shared memory reuse
reduces global memory traffic by a factor of K / BLOCK_K.

This is the fundamental optimization that every production matmul uses.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _tiled_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: int,
    N: int,
    K: int,
    stride_am: int,
    stride_ak: int,
    stride_bk: int,
    stride_bn: int,
    stride_cm: int,
    stride_cn: int,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Tiled matmul with shared memory.

    Each program:
      1. Uses shared memory tiles for A and B
      2. Iterates over K dimension, loading tiles into shared memory
      3. Accumulates partial results in registers
      4. Writes final result to global memory
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Offsets in the output matrix
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Compute base addresses for this tile's slices
    # A tile: (BLOCK_M, BLOCK_K) starting at (pid_m * BLOCK_M, 0)
    a_tile_base = a_ptr + rm[:, None] * stride_am + tl.arange(0, BLOCK_K)[None, :] * stride_ak
    # B tile: (BLOCK_K, BLOCK_N) starting at (0, pid_n * BLOCK_N)
    b_tile_base = b_ptr + tl.arange(0, BLOCK_K)[:, None] * stride_bk + rn[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        # Current K slice: k_start to k_start + BLOCK_K
        k_offs = k_start + tl.arange(0, BLOCK_K)

        # Load A tile from global memory
        a_ptrs = a_ptr + rm[:, None] * stride_am + k_offs[None, :] * stride_ak
        a_mask = (rm[:, None] < M) & (k_offs[None, :] < K)
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile from global memory
        b_ptrs = b_ptr + k_offs[:, None] * stride_bk + rn[None, :] * stride_bn
        b_mask = (k_offs[:, None] < K) & (rn[None, :] < N)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        accumulator += tl.dot(a_tile, b_tile)

    # Write result tile to global memory
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def tiled_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
) -> torch.Tensor:
    """Tiled matmul with configurable block sizes.

    Args:
        a: (M, K) matrix on CUDA.
        b: (K, N) matrix on CUDA.
        block_m: Tile size in M dimension.
        block_n: Tile size in N dimension.
        block_k: Tile size in K dimension.

    Returns:
        (M, N) output matrix.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"
    assert a.dim() == 2 and b.dim() == 2, "Inputs must be 2D matrices"
    M, K_a = a.shape
    K_b, N = b.shape
    assert K_a == K_b, f"Inner dimensions mismatch: {K_a} vs {K_b}"

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    _tiled_matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K_a,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return c


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping demo.")
    else:
        M, N, K = 512, 512, 256
        a = torch.randn(M, K, device="cuda", dtype=torch.float32)
        b = torch.randn(K, N, device="cuda", dtype=torch.float32)

        c_tiled = tiled_matmul(a, b)
        c_torch = torch.matmul(a, b)
        err = (c_tiled - c_torch).abs().max().item()
        print(f"Tiled matmul {M}x{K}x{N} - max error vs torch.matmul: {err:.2e}")
