"""Save generated PTX/CUBIN for a GEMM and compare stages 2 vs 4.

Prereq: triton installed (see 12_编译与安装指南.md).
"""

import os

import torch

import triton
import triton.language as tl


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
        acc += tl.dot(tl.load(a_ptrs + k), tl.load(b_ptrs + k))
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc)


def main():
    os.environ.setdefault("TRITON_DUMP_IR", "1")
    os.environ.setdefault("TRITON_DUMP_DIR", "/tmp/triton_dump")
    M = N = K = 1024
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.empty(M, N, device="cuda", dtype=torch.float32)
    grid = (M // 128, N // 128)
    for stages in (2, 4):
        os.environ["TRITON_DUMP_DIR"] = f"/tmp/triton_dump_stages{stages}"
        gemm[grid](
            a, b, c, M, N, K, BM=128, BN=128, BK=64, num_warps=8, num_stages=stages
        )
    torch.cuda.synchronize()
    print("done. dumps in /tmp/triton_dump_stages{2,4}")
    print("compare with: diff -r /tmp/triton_dump_stages2 /tmp/triton_dump_stages4")


if __name__ == "__main__":
    main()
