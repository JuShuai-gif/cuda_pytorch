"""
Triton autotune demo: matmul kernel with automatic parameter tuning.

Industrial context: GPU kernel performance depends on hyperparameters
(block size, num_warps, num_stages) that vary with GPU architecture and
problem size. Triton's @autotune searches the config space and caches
the best configuration per problem shape.

FlashAttention, xFormers, torch.inductor all use autotuning extensively.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Autotuned matmul kernel
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": m, "BLOCK_N": n, "BLOCK_K": k, "GROUP_M": 8},
            num_warps=w,
            num_stages=s,
        )
        for m in [32, 64, 128]
        for n in [32, 64, 128]
        for k in [32, 64]
        for w in [4, 8]
        for s in [2, 3, 4]
    ],
    key=["M", "N", "K"],
)
@triton.jit
def autotuned_matmul_kernel(
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
    """Tiled matmul kernel with autotuned tile sizes, warps, and stages.

    The GROUP_M parameter groups programs in the M dimension for L2 cache locality.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = num_pid_m - first_pid_m
    group_size_m = tl.minimum(group_size_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + rk
        a_mask = (rm[:, None] < M) & (k_offs[None, :] < K)
        b_mask = (k_offs[:, None] < K) & (rn[None, :] < N)
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a_tile, b_tile)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=c_mask)


# ---------------------------------------------------------------------------
# Autotuned matmul wrapper
# ---------------------------------------------------------------------------


def autotuned_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Fully autotuned matmul.

    On first call for a given (M, N, K), Triton runs all configs
    in the autotune space and caches the best one. Subsequent calls
    with the same shape use the cached config directly.

    Args:
        a: (M, K) matrix on CUDA.
        b: (K, N) matrix on CUDA.

    Returns:
        (M, N) output matrix.
    """
    assert a.is_cuda and b.is_cuda
    assert a.dim() == 2 and b.dim() == 2

    M, K_a = a.shape
    K_b, N = b.shape
    assert K_a == K_b, f"Inner dimension mismatch: {K_a} vs {K_b}"

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
        1,
        1,
    )

    autotuned_matmul_kernel[grid](
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
    )
    return c


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def matmul_autotune_demo() -> None:
    """Run autotuned matmul for typical shapes and print best configs."""
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping demo.")
        return

    print("=" * 70)
    print("  AUTOTUNE: Matmul Demo")
    print("=" * 70)

    shapes = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 512),
        (2048, 2048, 1024),
        (4096, 4096, 1024),
    ]

    for M, N, K in shapes:
        a = torch.randn(M, K, device="cuda", dtype=torch.float32)
        b = torch.randn(K, N, device="cuda", dtype=torch.float32)

        c_auto = autotuned_matmul(a, b)
        c_ref = torch.matmul(a, b)

        err = (c_auto - c_ref).abs().max().item()

        # Extract the best config from the autotuner
        best_config = autotuned_matmul_kernel.best_config
        if best_config is not None:
            cfg = best_config.kwargs
        else:
            cfg = {}

        print(f"\n  Shape: {M}x{N}x{K}")
        print(
            f"    Best config: BLOCK_M={cfg.get('BLOCK_M', '?')}, "
            f"BLOCK_N={cfg.get('BLOCK_N', '?')}, "
            f"BLOCK_K={cfg.get('BLOCK_K', '?')}, "
            f"num_warps={cfg.get('num_warps', '?')}, "
            f"num_stages={cfg.get('num_stages', '?')}"
        )
        print(f"    Max error:   {err:.2e}")

    print(f"\n  Total configs tested per shape: {len(autotuned_matmul_kernel.configs)}")


if __name__ == "__main__":
    matmul_autotune_demo()
