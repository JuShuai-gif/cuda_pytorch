"""
Simple Triton GEMM (General Matrix Multiply) as a warmup for later tiling work.

Computes C = A @ B where A is (M, K) and B is (K, N).

This implementation uses a basic tiled approach over the K dimension:
each program computes a BLOCK_M x BLOCK_N tile of the output by iterating
over K in chunks of BLOCK_K. This is the fundamental pattern that later
modules will extend with advanced tiling, swizzling, and pipelining.

This is NOT an optimized GEMM -- it is deliberately simple to demonstrate
the core Triton pattern: block pointers, index calculation, and accumulation.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_kernel(
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
    """Naive tiled GEMM: C = A @ B."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Offsets for this tile
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # Base pointers to the A and B tiles for this program
    a_tile_ptr = a_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    b_tile_ptr = b_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        # Mask for K dimension to avoid out-of-bounds
        k_offs = k + rk
        k_mask_a = k_offs[None, :] < K
        k_mask_b = k_offs[:, None] < K

        a_tile = tl.load(a_tile_ptr, mask=k_mask_a, other=0.0)
        b_tile = tl.load(b_tile_ptr, mask=k_mask_b, other=0.0)

        accumulator += tl.dot(a_tile, b_tile)

        # Advance pointers by BLOCK_K in the K dimension
        a_tile_ptr += BLOCK_K * stride_ak
        b_tile_ptr += BLOCK_K * stride_bk

    # Mask for M and N dimensions
    mask_m = rm[:, None] < M
    mask_n = rn[None, :] < N
    c_mask = mask_m & mask_n

    c_tile_ptr = c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(c_tile_ptr, accumulator, mask=c_mask)


def triton_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
) -> torch.Tensor:
    """
    Matrix multiplication C = A @ B using a Triton kernel.

    Args:
        a: (M, K) input matrix on CUDA.
        b: (K, N) input matrix on CUDA.
        block_m: Block size for M dimension.
        block_n: Block size for N dimension.
        block_k: Block size for K dimension.

    Returns:
        (M, N) output matrix on CUDA.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"
    assert a.dim() == 2 and b.dim() == 2, "Inputs must be 2D"
    M, K_a = a.shape
    K_b, N = b.shape
    assert K_a == K_b, f"Inner dimensions must match: {K_a} vs {K_b}"

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    _gemm_kernel[grid](
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

        c_triton = triton_gemm(a, b)
        c_torch = torch.matmul(a, b)

        err = (c_triton - c_torch).abs().max().item()
        print(f"GEMM {M}x{K}x{N} -- max error vs torch.matmul: {err:.2e}")
        print("GEMM demo passed.")
