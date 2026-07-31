"""Bonus（选做）：Triton tiled matmul。

任务：调 bench() 的 BLOCK_M / BLOCK_N / BLOCK_K，记录实测数据。
    哪组参数最快？差多少？为什么 tile 大小会影响这么大

    python -c "from kernels.matmul_triton import bench; bench()"

Triton tiled matmul 的写法要点（对比 _05 TileLang 版）：

1. 二维 grid：pid_m（行 block） + pid_n（列 block）
   grid = (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))
   每个 block 负责输出矩阵 C 的一个 (BLOCK_M × BLOCK_N) 子块

2. tl.arange(0, BLOCK_M) + [:, None] 做 2D 广播
   offs_m[:, None] 变成列向量 (BLOCK_M, 1)
   offs_n[None, :] 变成行向量 (1, BLOCK_N)
   相加得到 (BLOCK_M, BLOCK_N) 的 2D 索引矩阵
   这就是 Triton 的"隐式向量化"：用广播语法写 loop-free 的 2D 操作

3. stride 参数：处理非连续 tensor
   不是用固定列数 N 来算偏移，而是用 tensor 的 stride
   A[off_m, off_k] = A_ptr + off_m * stride_am + off_k * stride_ak
   这样 kernel 可以处理任意 stride 的 tensor（如转置、切片后的 view）

4. K 维循环 tiling：
   for k0 in range(0, K, BLOCK_K):
       每次加载 A 的一个 (BLOCK_M, BLOCK_K) 块和 B 的一个 (BLOCK_K, BLOCK_N) 块
       做 tl.dot(a, b) 累加到 acc
   与 _05_tilelang 的 T.Pipelined 循环逻辑相同，但这里没有显式的 shared memory
   (Triton >= 2.0 默认开启 MMA，编译器自动插入 shared 搬运)

5. tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32) 累加器
   对应 TileLang 的 T.alloc_fragment + T.clear
   用 float32 累加（即使输入是 fp16），防止精度丢失

6. tl.dot(a, b) tile 级矩阵乘法
   对应 TileLang 的 T.gemm(A_shared, B_shared, C_local)
   Triton 隐式调用 Tensor Core；TileLang 也是同样底层

7. tl.store 写回结果时用 mask 处理边缘越界

和 _05 TileLang 版对比：
┌──────────────┬──────────────────────────┬─────────────────────────┐
│              │ Triton                   │ TileLang                │
├──────────────┼──────────────────────────┼─────────────────────────┤
│ grid 声明     │ kernel[grid](...)        │ with T.Kernel(...) as   │
│ 数据搬运      │ stl.load（显式）          │ T.copy（隐式）           │
│ shared memory │ 隐式（编译器自动）        │ T.alloc_shared（显式）   │
│ 累加器        │ tl.zeros(...)            │ T.alloc_fragment + clear│
│ tile 乘累加   │ tl.dot(a, b)             │ T.gemm(A, B, C)         │
│ K 维流水      │ for k0 in range(...)     │ T.Pipelined(...)        │
│ mask / 越界   │ 手动 & (offs < M)        │ T.if_then_else（显式）   │
└──────────────┴──────────────────────────┴─────────────────────────┘

性能调优要点（tile 大小影响）：
- BLOCK_M/BLOCK_N 大 → 每个 block 算得多 → occupancy 可能降低
- BLOCK_K 大 → 内层循环次数少 → 算术强度高 → 可能到 compute bound
- 小 tile → 更多 block → 更高 occupancy → 但每个 block 算术强度低
- 最佳配置取决于 GPU 的 SM 数 / register 数 / shared memory 大小
"""

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        a_ptrs = (
            a_ptr + offs_m[:, None] * stride_am + (k0 + offs_k[None, :]) * stride_ak
        )
        b_ptrs = (
            b_ptr + (k0 + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn
        )
        a = tl.load(
            a_ptrs, mask=(offs_m[:, None] < M) & (k0 + offs_k[None, :] < K), other=0.0
        )
        b = tl.load(
            b_ptrs, mask=(k0 + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0
        )
        acc += tl.dot(a, b)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def matmul(
    a: torch.Tensor, b: torch.Tensor, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](
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
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return c


def bench(M=2048, N=2048, K=2048):
    assert torch.cuda.is_available(), "benchmark 需要 GPU"
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    configs = [
        (32, 32, 32),
        (64, 64, 32),
        (128, 64, 32),
        (128, 128, 32),
        (128, 128, 64),
        (64, 64, 64),
    ]
    for bm, bn, bk in configs:
        ms = triton.testing.do_bench(lambda: matmul(a, b, bm, bn, bk))
        tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12
        print(
            f"BLOCK_M={bm:4d} BLOCK_N={bn:4d} BLOCK_K={bk:3d}  "
            f"{ms:8.3f} ms  {tflops:6.1f} TFLOPS"
        )
    ms = triton.testing.do_bench(lambda: a @ b)
    tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12
    print(f"torch (cuBLAS)                      {ms:8.3f} ms  {tflops:6.1f} TFLOPS")
