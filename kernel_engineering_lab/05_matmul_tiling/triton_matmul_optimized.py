"""
Optimized Triton matmul with configurable warp count and tile sizes.

Extends the basic tiled matmul with:
  - Configurable num_warps showing impact on occupancy
  - Configurable tile sizes showing tradeoff (bigger tiles = more reuse but fewer blocks)
  - Sub-tiling for register-level reuse (optional)

Performance is tuned through autotuning-like configuration at runtime.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _optimized_matmul_kernel(
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
    GROUP_M: tl.constexpr,
):
    """Optimized matmul kernel.

    GROUP_M: Number of programs in M dimension to group together before
    moving to the next group in N. This improves L2 cache locality by
    processing nearby rows together.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Output tile offsets
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # Pointers to output
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + rk

        # Load A tile
        a_ptrs = a_ptr + rm[:, None] * stride_am + k_offs[None, :] * stride_ak
        a_mask = (rm[:, None] < M) & (k_offs[None, :] < K)
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile
        b_ptrs = b_ptr + k_offs[:, None] * stride_bk + rn[None, :] * stride_bn
        b_mask = (k_offs[:, None] < K) & (rn[None, :] < N)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate with tensor core operation
        accumulator += tl.dot(a_tile, b_tile)

    # Write back
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def optimized_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 32,
    num_warps: int = 4,
    num_stages: int = 3,
    group_m: int = 8,
) -> torch.Tensor:
    """Optimized matmul with configurable parameters.

    Args:
        a: (M, K) matrix on CUDA.
        b: (K, N) matrix on CUDA.
        block_m: Tile size in M dimension. Larger = more reuse, fewer blocks.
        block_n: Tile size in N dimension.
        block_k: Tile size in K dimension. Larger = more reuse per iteration.
        num_warps: Number of warps per kernel (2, 4, 8, 16).
                   More warps = higher occupancy but fewer registers per thread.
        num_stages: Number of pipeline stages (for future pipelining).
        group_m: Group size for M dimension scheduling (L2 cache locality).

    Returns:
        (M, N) output matrix.
    """
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"
    assert a.dim() == 2 and b.dim() == 2, "Inputs must be 2D matrices"
    M, K_a = a.shape
    K_b, N = b.shape
    assert K_a == K_b, f"Inner dimensions mismatch: {K_a} vs {K_b}"

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    num_pid_m = triton.cdiv(M, block_m)
    num_pid_n = triton.cdiv(N, block_n)
    grid = (num_pid_m * num_pid_n, 1, 1)

    _optimized_matmul_kernel[grid](
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
        GROUP_M=group_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c


# Pre-defined configurations for common scenarios
PRESET_CONFIGS = {
    "small": {
        "block_m": 64,
        "block_n": 64,
        "block_k": 32,
        "num_warps": 4,
        "group_m": 8,
    },
    "medium": {
        "block_m": 128,
        "block_n": 128,
        "block_k": 32,
        "num_warps": 4,
        "group_m": 8,
    },
    "large": {
        "block_m": 128,
        "block_n": 128,
        "block_k": 64,
        "num_warps": 8,
        "group_m": 8,
    },
}


def optimized_matmul_preset(
    a: torch.Tensor,
    b: torch.Tensor,
    preset: str = "medium",
) -> torch.Tensor:
    """Run optimized matmul with a named preset configuration.

    Args:
        a: (M, K) matrix on CUDA.
        b: (K, N) matrix on CUDA.
        preset: One of 'small', 'medium', 'large'.

    Returns:
        (M, N) output matrix.
    """
    config = PRESET_CONFIGS[preset]
    return optimized_matmul(a, b, **config)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping demo.")
    else:
        M, N, K = 1024, 1024, 512
        a = torch.randn(M, K, device="cuda", dtype=torch.float32)
        b = torch.randn(K, N, device="cuda", dtype=torch.float32)

        for preset in ("small", "medium", "large"):
            c_opt = optimized_matmul_preset(a, b, preset)
            c_torch = torch.matmul(a, b)
            err = (c_opt - c_torch).abs().max().item()
            print(f"Optimized matmul [{preset}] {M}x{K}x{N} - max error: {err:.2e}")
