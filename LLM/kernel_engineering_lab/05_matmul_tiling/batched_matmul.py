"""
Batched matmul in Triton.

Handles 3D tensors: C[b] = A[b] @ B[b] for b in 0..batch-1.

Shows how batching interacts with tiling and compares vs
torch.bmm and torch.matmul.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _batched_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: int,
    N: int,
    K: int,
    stride_ab: int,
    stride_am: int,
    stride_ak: int,
    stride_bb: int,
    stride_bk: int,
    stride_bn: int,
    stride_cb: int,
    stride_cm: int,
    stride_cn: int,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Batched tiled matmul: each program handles one batch element's tile."""
    pid_b = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # Pointers to output tile for this batch
    c_base = c_ptr + pid_b * stride_cb
    c_ptrs = c_base + rm[:, None] * stride_cm + rn[None, :] * stride_cn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    a_base = a_ptr + pid_b * stride_ab
    b_base = b_ptr + pid_b * stride_bb

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + rk

        # Load A tile
        a_ptrs = a_base + rm[:, None] * stride_am + k_offs[None, :] * stride_ak
        a_mask = (rm[:, None] < M) & (k_offs[None, :] < K)
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile
        b_ptrs = b_base + k_offs[:, None] * stride_bk + rn[None, :] * stride_bn
        b_mask = (k_offs[:, None] < K) & (rn[None, :] < N)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

        accumulator += tl.dot(a_tile, b_tile)

    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def batched_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
) -> torch.Tensor:
    """Batched matrix multiplication: C[b] = A[b] @ B[b].

    Args:
        a: (batch, M, K) tensor on CUDA.
        b: (batch, K, N) tensor on CUDA.
        block_m: Tile size in M dimension.
        block_n: Tile size in N dimension.
        block_k: Tile size in K dimension.

    Returns:
        (batch, M, N) tensor.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"
    assert a.dim() == 3 and b.dim() == 3, "Inputs must be 3D (batched)"
    batch_a, M, K_a = a.shape
    batch_b, K_b, N = b.shape
    assert batch_a == batch_b, f"Batch sizes must match: {batch_a} vs {batch_b}"
    assert K_a == K_b, f"Inner dimensions mismatch: {K_a} vs {K_b}"

    batch = batch_a
    c = torch.empty((batch, M, N), device=a.device, dtype=a.dtype)

    grid = lambda meta: (
        batch,
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )

    _batched_matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K_a,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return c


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping demo.")
    else:
        B, M, N, K = 4, 256, 256, 128
        a = torch.randn(B, M, K, device="cuda", dtype=torch.float32)
        b = torch.randn(B, K, N, device="cuda", dtype=torch.float32)

        c_triton = batched_matmul(a, b)
        c_torch_bmm = torch.bmm(a, b)
        c_torch_matmul = torch.matmul(a, b)

        err_bmm = (c_triton - c_torch_bmm).abs().max().item()
        err_mm = (c_triton - c_torch_matmul).abs().max().item()
        print(f"Batched matmul {B}x{M}x{K}x{N} - max error vs bmm: {err_bmm:.2e}")
        print(f"Batched matmul {B}x{M}x{K}x{N} - max error vs matmul: {err_mm:.2e}")
