"""Benchmark GEMM configs and compare with cuBLAS.

Prereq: triton installed (see 12_编译与安装指南.md).
"""

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
    from triton.testing import do_bench

    M = N = K = 4096
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.empty(M, N, device="cuda", dtype=torch.float32)

    configs = [
        dict(BM=128, BN=128, BK=32, num_warps=4, num_stages=3),
        dict(BM=128, BN=128, BK=64, num_warps=8, num_stages=4),
        dict(BM=64, BN=64, BK=64, num_warps=4, num_stages=4),
    ]

    def report(name, lat):
        tflops = 2 * M * N * K / (lat / 1e3) / 1e12
        print(f"{name:14s}: {lat * 1e3:8.1f} us  {tflops:7.1f} TFLOPS")

    for cfg in configs:
        grid = (M // cfg["BM"], N // cfg["BN"])
        lat = do_bench(lambda: gemm[grid](a, b, c, M, N, K, **cfg))
        report(str(cfg), lat)

    # cuBLAS baseline
    a32, b32 = a.float(), b.float()
    lat_ref = do_bench(lambda: a32 @ b32)
    report("cublas(torch)", lat_ref)


if __name__ == "__main__":
    main()
