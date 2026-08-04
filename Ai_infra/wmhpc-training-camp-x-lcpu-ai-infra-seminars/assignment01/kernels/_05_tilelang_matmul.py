"""问题 7.6：TileLang tiled matmul（填空）。

C = A @ B，A 形状 (M, K)，B 形状 (K, N)，fp16 输入、fp32 累加。
共五个空：两块 shared tile、一个 fragment 累加器、沿 K 维的流水
循环、T.copy 搬运与 T.gemm 计算。
需要 GPU 和 tilelang（uv sync --extra tilelang），在集群上运行：
    pytest tests/test_tilelang.py -k matmul

Bonus 用的也是这个文件：填完后调 bench() 里的配置，记录实测数据。

    python -c "from kernels.tilelang_matmul import bench; bench()"

演示的 TileLang 用法（tiled matmul 全流程）：

1. T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
   分配 shared memory tile：A_shared 为 A 的分块缓存，B_shared 为 B 的分块缓存
   后续 T.gemm 从 shared 读取，避免重复访问全局内存
2. T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
   分配寄存器级累加器（fragment），存 C 的子矩阵部分和
   fragment 在寄存器中，速度最快，算完最后一并写回全局内存
3. T.clear(C_local)
   将累加器清零（相当于 C_local = 0），为 K 维累加做准备
4. for k in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=3):
   沿 K 维流水推进：每一轮搬 A/B 的一个 tile 进 shared，做一次 tile 级 GEMM 累加到 C_local
   T.Pipelined 实现软件流水线（software pipeline），num_stages 控制预取深度，
   让数据搬运和计算重叠执行，隐藏访存延迟
5. T.copy(源, 目标)
   搬运数据：T.copy(A[行切片, 列切片], A_shared) 把 A 的子块拷进 shared
   T.copy(C_local, C[行切片, 列切片]) 算完把累加器写回全局内存
6. T.gemm(A_shared, B_shared, C_local)
   tile 级矩阵乘累加：C_local += A_shared @ B_shared
   TileLang 自动生成高效的 warp tile MMA 指令

总结：这一套（shared + fragment + Pipelined + T.copy + T.gemm）就是
tiled matmul 的标准范式。手写 Triton 需要自己管理 TMA / cp_async / warp MMA，
TileLang 把这三层抽象合并成上面几个 API，大幅降低编写成本。
"""

import tilelang
import tilelang.language as T


def make_matmul(
    M,
    N,
    K,
    BLOCK_M=128,
    BLOCK_N=128,
    BLOCK_K=32,
    threads=128,
    num_stages=3,
    dtype="float16",
    accum_dtype="float32",
):
    @T.prim_func
    def main(
        # 这是需要操作的三个矩阵的指针
        # 编译成 CUDA kernel后，这三个 T.Buffer 分别变成一个 float* (或 half*等)指针，指向 GPU 全局内存
        A: T.Buffer((M, K), dtype),
        B: T.Buffer((K, N), dtype),
        C: T.Buffer((M, N), accum_dtype),
    ):
        with T.Kernel(
            # x 方向 ceil(N/BLOCK_N) 个 block
            T.ceildiv(N, BLOCK_N),
            # y 方向 ceil(M/BLOCK_M) 个 block
            T.ceildiv(M, BLOCK_M),
            # threads 是每个 block 内的线程数
            threads=threads,
        ) as (bx, by):  # bx / by 是当前block的grid坐标(作为循环变量)，用于计算该 block 处理的是 A B C 矩阵中的那一块数据
            # 可以理解为将 M*N 大小的输出拆成 BLOCK_M * BLOCK_N 的小块
            
            
            # K 维度会被分成多个 step，每个 step 把 A 的一列 tile 和 B 的一行 tile 搬进 共享内存，
            # 二者在共享内存上做乘加，结果累加到 C 的一个 BLOCK_M * BLOCK_N tile 中
            # 共享内存让 K 维度上的数据只需要加载一次就能被 block 内所有线程复用，大幅减少全局内存访问
            A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), dtype)
            B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), dtype)

            # 在寄存器上分配一个累加器 tile
            # BLOCK_M * BLOCK_N ---- 输出 C 矩阵的一个小块
            # accum_dtype ---- 累加精度()
            C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)

            T.clear(C_local)

            # ====== 空 3：沿 K 维流水推进 ======
            for k in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=num_stages):
                # ====== 空 4：把 A、B 的当前 tile 搬进 shared ======
                T.copy(A[by * BLOCK_M, k * BLOCK_K], A_shared)
                T.copy(B[k * BLOCK_K, bx * BLOCK_N], B_shared)
                # ====== 空 5：tile 级乘累加 ======
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * BLOCK_M, bx * BLOCK_N])

    return main


def bench(M=2048, N=2048, K=2048):
    import torch
    import triton

    assert torch.cuda.is_available(), "benchmark 需要 GPU"
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)

    # =====就在这里修改，可以加入多条config
    # (block_M, block_N, block_K, threads, num_stages)
    configs = [
        (64, 64, 32, 128, 1),
        (64, 64, 32, 128, 3),
        (128, 128, 32, 128, 3),
        (128, 128, 64, 256, 3),
        (128, 256, 64, 256, 3),
    ]

    checked = False
    for bm, bn, bk, threads, stages in configs:
        kernel = tilelang.compile(
            make_matmul(M, N, K, bm, bn, bk, threads=threads, num_stages=stages),
            out_idx=[2],
        )
        if not checked:  # 第一个配置和 torch 对拍一次
            ref = a.float() @ b.float()
            torch.testing.assert_close(kernel(a, b), ref, rtol=1e-2, atol=1e-1)
            checked = True
        ms = triton.testing.do_bench(lambda: kernel(a, b))
        tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12
        print(
            f"block_M={bm:4d} block_N={bn:4d} block_K={bk:3d} "
            f"threads={threads:3d} stages={stages}  {ms:8.3f} ms  {tflops:6.1f} TFLOPS"
        )
