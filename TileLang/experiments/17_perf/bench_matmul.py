"""Perf experiment: benchmark GEMM configs with do_bench.

Prereq: tilelang installed (see 12_编译与安装指南.md).
"""

import torch

import tilelang
import tilelang.language as T


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
    configs = [
        dict(block_M=128, block_N=128, block_K=64, threads=256, stages=2),
        dict(block_M=128, block_N=256, block_K=64, threads=256, stages=3),
        dict(block_M=64, block_N=128, block_K=64, threads=128, stages=4),
    ]
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(N, K, device="cuda", dtype=torch.float16)
    for cfg in configs:
        k = matmul.compile(M=M, N=N, K=K, **cfg)
        c = k(a, b)
        prof = k.get_profiler()
        lat = prof.do_bench(n_warmup=25, n_repeat=100)
        tflops = 2 * M * N * K / (lat * 1e-3) / 1e12
        print(f"{cfg} -> {lat * 1e6:.1f} us, {tflops:.1f} TFLOPS")

    # cuBLAS baseline
    a32 = a.float()
    b32 = b.float()
    ref = a32 @ b32.t()
    lat_ref = None
    try:
        from tilelang.profiler import do_bench

        lat_ref = do_bench(lambda: a32 @ b32.t(), warmup=25, rep=100)
        print(
            f"cuBLAS(torch) -> {lat_ref * 1e6:.1f} us, {2 * M * N * K / (lat_ref * 1e-3) / 1e12:.1f} TFLOPS"
        )
    except Exception as e:
        print("cuBLAS baseline failed:", e)


if __name__ == "__main__":
    main()
