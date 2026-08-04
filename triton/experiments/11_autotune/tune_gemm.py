"""Autotune a GEMM over block sizes / warps / stages.

Prereq: triton installed (see 12_编译与安装指南.md).
"""

import torch

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BM": 64, "BN": 64, "BK": 32}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 128, "BK": 32}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 128, "BK": 64}, num_warps=8, num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def gemm(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B + offs_n[None, :] * K + offs_k[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptrs + k)
        b = tl.load(b_ptrs + k)
        acc += tl.dot(a, b)
    offs = C + offs_m[:, None] * N + offs_n[None, :]
    tl.store(offs, acc)


def main():
    M = N = K = 1024
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.empty(M, N, device="cuda", dtype=torch.float32)
    gemm[(M // 128, N // 128)](a, b, c, M, N, K)
    torch.cuda.synchronize()
    print("best config:", gemm.best_config)
    ref = a.float() @ b.float()
    print("max mismatch:", (c - ref).abs().max().item())


if __name__ == "__main__":
    main()
