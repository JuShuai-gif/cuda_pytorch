"""问题 7.3：TileLang 版 scale-add（填空）。

Y = 2 * X + 1，X 形状 (M, N)。两个空对应 TileLang 的两个 basic operation。
需要 GPU 和 tilelang（uv sync --extra tilelang），在集群上运行：
    pytest tests/test_tilelang.py -k scale_add

TileLang 核心用法（对比 Triton）：

1. @T.prim_func — 声明可编译为 GPU 代码的原语函数（≈ @triton.jit）
2. T.Buffer((M, N), dtype) — 带形状和类型的 buffer 注解输入/输出
   （Triton 用裸指针，自己管边界）
3. with T.Kernel(grid_M, grid_N, threads=128) as (bx, by):
   启动 2D grid 并解包 block 索引 bx/by
   （≈ kernel[grid](...) + tl.program_id，集成在 context manager 里）
4. T.ceildiv — 向上取整算 grid 大小（≈ triton.cdiv）
5. for i, j in T.Parallel(block_M, block_N):
   声明 tile 内并行，编译器自动将 (i,j) 映射到线程
   （≈ tl.arange + 隐式向量化）
6. Y[gi, gj] = X[gi, gj] * 2.0 + 1.0
   直接下标读写，无需手动 tl.load / tl.store，TileLang 自动处理访存

总结：比 Triton 更高一层，kernel 写成嵌套循环，load/store/grid 映射全部隐式，
读起来像普通 Python 但生成的是 GPU 代码。
"""

import tilelang
import tilelang.language as T


def make_scale_add(M, N, block_M=32, block_N=32, dtype="float32"):
    @T.prim_func
    def scale_add(
        X: T.Buffer((M, N), dtype),
        Y: T.Buffer((M, N), dtype),
    ):
        # ====== 空 1：二维 CTA grid——x 方向要多少个 block（管 N 列），
        #         y 方向要多少个（管 M 行）？提示：T.ceildiv ======
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
            bx,
            by,
        ):
            # ====== 空 2：block 内并行遍历 tile 的每个元素，
            #         提示：T.Parallel(维度1, 维度2) ======
            for i, j in T.Parallel(block_M, block_N):
                gi = by * block_M + i
                gj = bx * block_N + j
                if gi < M and gj < N:
                    Y[gi, gj] = X[gi, gj] * 2.0 + 1.0

    return scale_add
