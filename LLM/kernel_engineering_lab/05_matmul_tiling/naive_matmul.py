"""
Naive Triton matmul: no shared memory tiling.

Each program loads one row from A and one column from B from global memory,
computes the dot product, and writes one output element to C. This demonstrates
the performance problem: no data reuse, excessive global memory traffic.

C = A @ B  where A is (M, K), B is (K, N)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _naive_matmul_kernel(
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
    """Naive GEMM: loads elements directly from global memory each iteration.

    No shared memory tiling - every read hits global memory.
    This is NOT how you should write matmul, but it demonstrates the
    performance gap that tiling solves.
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Range of rows this program handles
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # Pointers to the output tile
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_offs = k + rk

        # A[batch_m, batch_k]: (BLOCK_M, BLOCK_K) tile
        a_ptrs = a_ptr + rm[:, None] * stride_am + k_offs[None, :] * stride_ak
        a_mask = (rm[:, None] < M) & (k_offs[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # B[batch_k, batch_n]: (BLOCK_K, BLOCK_N) tile
        b_ptrs = b_ptr + k_offs[:, None] * stride_bk + rn[None, :] * stride_bn
        b_mask = (k_offs[:, None] < K) & (rn[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        accumulator += tl.dot(a, b)

    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def naive_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
) -> torch.Tensor:
    """Naive matmul: loads directly from global memory, no shared memory reuse.

    Args:
        a: (M, K) matrix on CUDA.
        b: (K, N) matrix on CUDA.
        block_m: Tile size in M dimension.
        block_n: Tile size in N dimension.
        block_k: Tile size in K dimension (number of elements to load at once).

    Returns:
        (M, N) output matrix on CUDA.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"
    assert a.dim() == 2 and b.dim() == 2, "Inputs must be 2D matrices"
    M, K_a = a.shape
    K_b, N = b.shape
    assert K_a == K_b, f"Inner dimensions must match: {K_a} vs {K_b}"

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    _naive_matmul_kernel[grid](
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
        M, N, K = 256, 256, 128
        a = torch.randn(M, K, device="cuda", dtype=torch.float32)
        b = torch.randn(K, N, device="cuda", dtype=torch.float32)

        c_naive = naive_matmul(a, b)
        c_torch = torch.matmul(a, b)
        err = (c_naive - c_torch).abs().max().item()
        print(f"Naive matmul {M}x{K}x{N} - max error vs torch.matmul: {err:.2e}")
