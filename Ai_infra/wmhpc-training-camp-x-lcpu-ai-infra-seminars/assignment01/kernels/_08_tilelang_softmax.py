"""问题 7.7（压轴）：softmax in TileLang（FROM-SCRATCH）。

contract：
- softmax(x) 接收形状 (M, N) 的 float32 CUDA tensor，返回同形状结果，
  对每一行独立做 softmax；
- kernel 用 TileLang 自己写，一个 block 处理一行（或一小批行）；
- 为了确保数值稳定，要求行内先减最大值，再做 exp 与求和。测试里有一行
  数值巨大的输入，不稳定的实现会得到 inf/nan；
- 行宽 N 任意，可以假设 N <= 4096。TileLang 的 kernel 按形状编译，
  用 make_xxx(M, N) 针对形状生成、在 wrapper 里按形状缓存编译结果
  是常见做法（结构可以参考 7.3、7.4）；
- 归约用 T.reduce_max / T.reduce_sum，逐元素部分用 T.Parallel 加 T.exp；
- fragment 的宽度建议取不小于 N 的 2 的幂（类比 Triton 的
  next_power_of_2），不足的位置补 -inf（T.if_then_else 加 T.infinity），
  否则布局推断可能报 no available layout；
- 通过 pytest tests/test_tilelang_softmax.py 即为完成。

(Optional) 将你的实现和 torch.softmax 比较一下性能（行宽取 256/1024/4096），
Tip: elementwise + 行内归约的 kernel 大概率是带宽瓶颈，可以想想理论上限是多少。

TileLang softmax 和 Triton 版（_07）的对照：

相同思路（reduce_max → exp → reduce_sum → divide），API 不同：

┌──────────────────┬─────────────────────────────┬──────────────────────────────┐
│ 步骤              │ Triton                      │ TileLang                     │
├──────────────────┼─────────────────────────────┼──────────────────────────────┤
│ block 索引        │ pid = tl.program_id(0)      │ with T.Kernel(1, M) as(bx,by)│
│ 行定位            │ row_start = pid * N          │ by 直接索引行，bx=0（1D grid）│
│ 分配寄存器缓冲     │ （隐式，用 tl.arange 向量）  │ T.alloc_fragment((BLOCK_N,))  │
│ 加载 + 边界填充    │ tl.load(..., other=-inf)    │ T.Parallel + T.if_then_else   │
│ 求行最大值         │ tl.max(x, axis=0)           │ T.reduce_max(x_local, axis=0) │
│ exp               │ tl.exp(x - x_max)           │ T.Parallel + T.exp            │
│ 求行和            │ tl.sum(x, axis=0)           │ T.reduce_sum(x_local, axis=0) │
│ 除法 + 写回       │ y = x / x_sum; tl.store     │ T.Parallel + Y[by,j] = ...    │
│ JIT 编译          │ @triton.jit（隐式）          │ tilelang.compile(...)          │
└──────────────────┴─────────────────────────────┴──────────────────────────────┘

TileLang 额外要点：
- T.if_then_else(cond, true_val, false_val)  条件赋值（类似 Python 的三元表达式）
- T.infinity("float32")  浮点无穷大常量
- T.float32(0.0)  显式类型转换，不然 T.if_then_else 可能类型推断失败
- BLOCK_N = 2 的幂：TileLang 的布局推导要求 fragment 尺寸对齐，
  不足位置用 -inf 填充
- _cache 按 (M, N) 缓存编译结果：TileLang 的 kernel 是 shape-specialized 的，
  换形状要重新编译，缓存避免重复 JIT

对比感受：Triton 更"指令级"（显式 load/store/arange/mask），
TileLang 更"语义级"（Parallel/if_then_else/reduce），
后者读起来更像数学公式但需要理解其隐式映射。
"""

import torch
import tilelang
import tilelang.language as T

_cache = {}


def make_softmax(M, N):
    BLOCK_N = 1
    while BLOCK_N < N:
        BLOCK_N *= 2

    @T.prim_func
    def kernel(
        X: T.Buffer((M, N), "float32"),
        Y: T.Buffer((M, N), "float32"),
    ):
        with T.Kernel(1, M, threads=128) as (bx, by):
            x_local = T.alloc_fragment((BLOCK_N,), "float32")

            # Load row with -inf padding for positions >= N
            for j in T.Parallel(BLOCK_N):
                x_local[j] = T.if_then_else(
                    j < N,
                    X[by, j],
                    -T.infinity("float32"),
                )

            # Numerically stable: subtract max before exp
            x_max = T.reduce_max(x_local, axis=0)
            for j in T.Parallel(BLOCK_N):
                x_local[j] = T.exp(x_local[j] - x_max)

            x_sum = T.reduce_sum(x_local, axis=0)
            for j in T.Parallel(BLOCK_N):
                Y[by, j] = T.if_then_else(
                    j < N,
                    x_local[j] / x_sum,
                    T.float32(0.0),
                )

    return kernel


def softmax(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    key = (M, N)
    if key not in _cache:
        _cache[key] = tilelang.compile(make_softmax(M, N), out_idx=[1])
    kernel = _cache[key]
    y = torch.empty_like(x)
    kernel(x, y)
    return y
