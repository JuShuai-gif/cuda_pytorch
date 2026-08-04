"""问题 7.4：TileLang 版二维缩放拷贝（填空）。

Y = 2 * X，X 形状 (M, N)，M、N 都不保证整除 tile 边长（类似 prob 2.6）。
这次把 tile 先搬进 shared memory，算完再写回：数据搬运交给 T.copy。
填完对照 2.6 想一想：行列号、边界保护、grid 尺寸这几个空，
哪些在这里还有对应，哪些被 T.copy 吃掉了。
需要 GPU 和 tilelang（uv sync --extra tilelang），在集群上运行：
    pytest tests/test_tilelang.py -k copy2d

演示的 TileLang 用法：

1. T.alloc_shared((block_M, block_N), dtype)
   在 shared memory 上分配 tile 缓冲区，加速 block 内数据复用
   （≈ Triton 的 tl.static_shared + 手动管理索引）
2. T.copy(源, 目标)
   自动搬数据：T.copy(X[切片], shared) 把 X 的子矩阵拷进 shared，
   T.copy(shared, Y[切片]) 算完写回全局内存
   边界保护、行列偏移全部由 T.copy 隐式处理，不需要手动写 mask 和越界检查
3. 对比 _03_tilelang_scale_add.py：
   那里直接 Y[gi,gj]=... 操作全局内存（每次读都走 global memory），
   这里先搬进 shared 再算（同一 tile 内数据被多次复用），
   这就是 shared memory + T.copy 组合的价值：减少全局访存，提升带宽利用率。
"""

import tilelang
import tilelang.language as T


def make_scale2d(M, N, block_M=32, block_N=32, dtype="float32"):
    @T.prim_func  # 等同于 @triton.jit: 声明此函数可编译为 GPU kernel
    def scale2d(
        X: T.Buffer((M, N), dtype),  # 输入 buffer: 形状 (M, N)，float32
        Y: T.Buffer((M, N), dtype),  # 输出 buffer: 同形状同类型
    ):
        # Grid: (ceil(N/block_N), ceil(M/block_M)) 个 block，每个 block 128 线程
        # bx, by: 当前 block 在 grid 中的 x 方向(列)和 y 方向(行)索引
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
            bx,
            by,
        ):
            # 在片上共享内存中分配一个 tile（比全局内存快约 100 倍）
            X_shared = T.alloc_shared((block_M, block_N), dtype)

            # 将 X 中当前 block 对应的 tile 从全局内存搬进共享内存
            # T.copy 自动处理边界 mask / 越界检查，无需手动写
            T.copy(X[by * block_M, bx * block_N], X_shared)

            # 并行计算: block 内每个线程处理 tile 中的一个元素
            # T.Parallel 将 block_M * block_N 个线程分配到 i, j 维度上
            for i, j in T.Parallel(block_M, block_N):
                X_shared[i, j] = X_shared[i, j] * 2.0  # Y = 2 * X，在共享内存中完成计算

            # 将算完的 tile 从共享内存写回全局内存 Y 的对应位置
            T.copy(X_shared, Y[by * block_M, bx * block_N])

    return scale2d
