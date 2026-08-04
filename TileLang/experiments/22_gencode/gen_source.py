"""Generated code analysis: save CUDA source for elementwise and matmul.

Prereq: tilelang installed (see 12_编译与安装指南.md).
"""

import torch

import tilelang
import tilelang.language as T


@tilelang.jit
def add(A, B):
    n = T.const("n")
    A: T.Tensor((n,), T.float32)
    B: T.Tensor((n,), T.float32)
    C = T.empty((n,), T.float32)
    with T.Kernel(T.ceildiv(n, 256), threads=256) as bx:
        for i in T.Parallel(256):
            C[bx * 256 + i] = A[bx * 256 + i] + B[bx * 256 + i]
    return C


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
    k_add = add.compile(n=1024)
    open("add.cu", "w").write(k_add.get_kernel_source())
    print("wrote add.cu")

    for stages in (1, 3):
        k = matmul.compile(
            M=1024,
            N=1024,
            K=1024,
            block_M=128,
            block_N=128,
            block_K=64,
            threads=256,
            stages=stages,
        )
        name = f"matmul_stages{stages}.cu"
        open(name, "w").write(k.get_kernel_source())
        print(f"wrote {name}")

    print("\ncompare stages 1 vs 3:")
    import difflib

    a = open("matmul_stages1.cu").readlines()
    b = open("matmul_stages3.cu").readlines()
    diff = list(
        difflib.unified_diff(a, b, fromfile="stages1", tofile="stages3", lineterm="")
    )
    for line in diff[:40]:
        print(line, end="")


if __name__ == "__main__":
    main()
