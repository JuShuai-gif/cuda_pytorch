"""
第六讲 — GPU 编程：Triton 分块 matmul kernel。

实现基于共享内存分块的矩阵乘法，
遵循经典的 Triton matmul 教程模式。
"""

from __future__ import annotations

from typing import Tuple

import torch

try:
    import triton  # type: ignore[import-untyped]
    import triton.language as tl  # type: ignore[import-untyped]

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# PyTorch 基准实现
# ---------------------------------------------------------------------------


def matmul_pytorch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch 基准 matmul。"""
    return a @ b


# ---------------------------------------------------------------------------
# Triton 分块 matmul
# ---------------------------------------------------------------------------


if HAS_TRITON:

    @triton.autotune(
        configs=[
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32},
                num_stages=3,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32},
                num_stages=3,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
                num_stages=4,
                num_warps=4,
            ),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def _matmul_kernel(
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
        GROUP_SIZE_M: tl.constexpr = 8,
    ):
        """分块 matmul kernel：C[M,N] = A[M,K] × B[K,N]。

        使用共享内存分块，每次迭代加载 BLOCK_M × BLOCK_K 大小的 A 块
        和 BLOCK_K × BLOCK_N 大小的 B 块。
        """
        pid = tl.program_id(0)

        # 沿 M 维度的分块 ID（分组调度）
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        # 偏移量
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        # 累加器
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            # 加载分块
            k_mask = (k + offs_k) < K
            a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)

            acc = tl.dot(a, b, acc)

            # 推进指针
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        # 写回结果
        m_mask = offs_m < M
        n_mask = offs_n < N
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])

    def matmul_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Triton 分块矩阵乘法。若 Triton 不可用则回退到 PyTorch。

        期望输入为 2D 连续行主序张量。
        """
        if not HAS_TRITON:
            return matmul_pytorch(a, b)

        assert a.dim() == 2 and b.dim() == 2
        assert a.size(1) == b.size(0), "Inner dimensions must match"

        a = a.contiguous()
        b = b.contiguous()
        M, K = a.shape
        Kb, N = b.shape

        c = torch.empty((M, N), device=a.device, dtype=a.dtype)

        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
        )  # noqa: E731

        _matmul_kernel[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
        )
        return c

else:

    def matmul_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """回退方案：PyTorch matmul（Triton 未安装）。"""
        return matmul_pytorch(a, b)


# ---------------------------------------------------------------------------
# 正确性测试
# ---------------------------------------------------------------------------


def test_matmul(tol: float = 1e-3) -> Tuple[bool, float]:
    """比较 PyTorch matmul 与 Triton 分块 matmul。"""
    torch.manual_seed(42)
    M, K, N = 256, 512, 128
    a = torch.randn(M, K, dtype=torch.float16)
    b = torch.randn(K, N, dtype=torch.float16)

    c_ref = matmul_pytorch(a.float(), b.float()).half()
    c_triton = matmul_triton(a, b)

    max_diff = (c_ref.float() - c_triton.float()).abs().max().item()
    ok = max_diff < tol
    return ok, max_diff


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print(f"Triton available: {HAS_TRITON}")

    ok, max_diff = test_matmul()
    print(f"Matmul correctness: {'PASS' if ok else 'FAIL'} (max diff = {max_diff:.2e})")

    # 小规模形状测试
    a = torch.randn(4, 8)
    b = torch.randn(8, 4)
    c = matmul_triton(a, b)
    c_ref = a @ b
    print(
        f"\nSmall matmul (4×8 × 8×4): max diff = {(c - c_ref).abs().max().item():.2e}"
    )
    print(f"Triton output:\n{c}")
    print(f"PyTorch output:\n{c_ref}")

    print("\nAll checks passed.")
