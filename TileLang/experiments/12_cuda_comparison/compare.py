"""12_cuda_comparison: same GEMM in torch / tilelang / (triton if available).

Prereq: tilelang installed (see 12_编译与安装指南.md); triton optional.
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
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(N, K, device="cuda", dtype=torch.float16)
    a32, b32 = a.float(), b.float()

    from tilelang.profiler import do_bench

    # torch / cuBLAS
    lat_torch = do_bench(lambda: a32 @ b32.t(), warmup=25, rep=100)

    # tilelang
    k = matmul.compile(
        M=M, N=N, K=K, block_M=128, block_N=128, block_K=64, threads=256, stages=3
    )
    c = k(a, b)
    torch.cuda.synchronize()
    lat_tl = do_bench(lambda: k(a, b), warmup=25, rep=100)

    def report(name, lat):
        tflops = 2 * M * N * K / (lat * 1e-3) / 1e12
        print(f"{name:12s}: {lat * 1e6:7.1f} us  {tflops:7.1f} TFLOPS")

    report("torch", lat_torch)
    report("tilelang", lat_tl)

    # triton (if installed)
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def triton_gemm(
            a_ptr,
            b_ptr,
            c_ptr,
            M,
            N,
            K,
            BM: tl.constexpr,
            BN: tl.constexpr,
            BK: tl.constexpr,
            TM: tl.constexpr,
            TN: tl.constexpr,
        ):
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)
            offs_m = pid_m * BM + tl.arange(0, TM)
            offs_n = pid_n * BN + tl.arange(0, TN)
            acc = tl.zeros((TM, TN), dtype=tl.float32)
            for k in range(0, K, BK):
                offs_k = k + tl.arange(0, BK)
                a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
                b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
                av = tl.load(a_ptrs).to(tl.float32)
                bv = tl.load(b_ptrs).to(tl.float32)
                acc += tl.sum(av[:, :, None] * bv[None, :, :], axis=1)
            offs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
            tl.store(offs, acc)

        c_triton = torch.empty(M, N, device="cuda", dtype=torch.float32)
        grid = (M // 128, N // 128)
        lat_tr = do_bench(
            lambda: triton_gemm[grid](
                a, b, c_triton, M, N, K, 128, 128, 64, 128, 128, num_warps=4
            )
        )
        report("triton", lat_tr)
        print("triton ok:", torch.allclose(c_triton, c, rtol=1e-2, atol=1e-2))
    except ImportError:
        print("triton not installed; skip")


if __name__ == "__main__":
    main()
