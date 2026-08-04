"""Autotune a GEMM over block sizes / threads.

Prereq: tilelang installed (see 12_编译与安装指南.md).
"""

import torch

import tilelang
import tilelang.language as T


@tilelang.autotune(
    configs=[
        {"block_M": 128, "block_N": 128, "block_K": 32, "threads": 128},
        {"block_M": 128, "block_N": 128, "block_K": 64, "threads": 256},
        {"block_M": 64, "block_N": 64, "block_K": 64, "threads": 128},
    ]
)
@tilelang.jit(out_idx=[2])
def matmul(M, N, K, block_M, block_N, block_K, threads, stages):
    dtype = "float16"
    A = T.Tensor((M, K), dtype)
    B = T.Tensor((N, K), dtype)
    C = T.Tensor((M, N), "float32")
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (
        bx,
        by,
    ):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_N, block_K), dtype)
        C_local = T.alloc_fragment((block_M, block_N), "float32")
        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=stages):
            T.copy(A[by * block_M, ko * block_K], A_shared)
            T.copy(B[bx * block_N, ko * block_K], B_shared)
            T.gemm(A_shared, B_shared, C_local)
        T.copy(C_local, C[by * block_M, bx * block_N])
    return C


def main():
    M = N = K = 4096
    k = matmul(M=M, N=N, K=K, stages=3)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(N, K, device="cuda", dtype=torch.float16)
    c = k(a, b)
    print("output shape:", c.shape)
    print("result sum:", c.float().sum().item())


if __name__ == "__main__":
    main()
