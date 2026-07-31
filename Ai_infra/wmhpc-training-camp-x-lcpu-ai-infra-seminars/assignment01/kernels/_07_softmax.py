"""问题 7.8（选做）：softmax in Triton（FROM-SCRATCH）。

注：此题可以不用GPU (conftest.py 会自动切到 interpreter 模式)。

contract：
- softmax(x) 接收形状 (M, N) 的 2D tensor，返回同形状结果，
  对每一行独立做 softmax；
- kernel 自己写，一个 program 处理一行；
- 为了确保数值稳定，要求行内先减最大值，再做 exp 与求和。测试里有一行
  数值巨大的输入，不稳定的实现会得到 inf/nan；
- 行宽 N 任意（用 mask 处理），可以假设 N <= 4096，BLOCK_SIZE 用
  triton.next_power_of_2(N) 是常见做法；
- 通过 pytest tests/test_softmax.py 即为完成。

Triton softmax 的写法要点：

1. 一维 grid => 一个 block 处理一行
   grid = (M,)  即 M 个 block，每个 block 负责一行
   row_start = pid * N  定位当前行在 row-major 排布中的起始偏移

2. tl.arange(0, BLOCK_SIZE) 生成 [0, 1, ..., BLOCK_SIZE-1]，
   加上 row_start 得到行内全局偏移；mask = arange < N 屏蔽尾部越界

3. 归约（reduction）：Triton 内置的 tl.max / tl.sum
   axis=0 对整行归约得到一个标量，block 内所有 thread 自动参与，
   无需手动写 warp shuffle / shared memory

4. 数值稳定性：先减最大值再 exp，避免 exp(large_number) = inf

5. other=float("-inf") 确保 mask 外的位置不影响 max（-inf 比任何实数都小）

6. triton.next_power_of_2(N) 把 N 上取整到 2 的幂，编译器好做 padding/tiling

对比 03 的 elementwise kernel，softmax 多了归约操作（tl.max / tl.sum），
这是 Triton 相比手写 CUDA 的另一个优势：内置归约自动处理跨 thread 通信。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_start = pid * N
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < N

    x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf"))
    x_max = tl.max(x, axis=0)  # 行内最大值，数值稳定
    x = tl.exp(x - x_max)
    x_sum = tl.sum(x, axis=0)
    y = x / x_sum
    tl.store(y_ptr + offsets, y, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    softmax_kernel[grid](x, y, M, N, BLOCK_SIZE=BLOCK_SIZE)
    return y
