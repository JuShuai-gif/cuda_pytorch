"""问题 7.1：Triton 向量加法（填空）。

四个空对应 Triton kernel 的四个 basic operation。填完运行：
    pytest tests/test_vector_add.py
没有 GPU 也能跑，conftest.py 会自动切到 interpreter 模式。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, z_ptr, n, BLOCK_SIZE: tl.constexpr):
    # ====== 空 1：当前 program 在一维 grid 里的编号 ======
    # 相当于 CUDA 的 blockIdx.x。
    # 每个 block 通过 pid 计算出自己负责的数据区间（如 pid=0 处理 [0, BLOCK_SIZE)），
    # 不同 block 各管一段，实现数据并行。没有 pid，所有 block 会做完全相同的计算。
    pid = tl.program_id(0)
    # ====== 空 2：这个 program 负责的一段全局下标（长度 BLOCK_SIZE） ======
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # ====== 空 3：屏蔽越界位置的 mask ======
    mask = offsets < n

    # 每个 block 同时并发加载自己负责的数据段，
    # 如 pid=0 取 [0:BLOCK_SIZE]，pid=1 取 [BLOCK_SIZE:2*BLOCK_SIZE]，以此类推。
    # mask 用于尾块防越界（如 n=2500, BLOCK_SIZE=1024 时 pid=2 只有 452 个有效元素），
    # 越界位置用 other=0.0 填充。
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # ====== 空 4：把 x + y 写回 z（别忘了 mask） ======
    # 每个 block 将计算好的 x+y 写回 z 的对应偏移位置，
    # mask 屏蔽尾块越界写入（和 load 时逻辑一致，超出 n 的位置不写）。
    tl.store(z_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    add_kernel[grid](x, y, z, n, BLOCK_SIZE=BLOCK_SIZE)
    return z
