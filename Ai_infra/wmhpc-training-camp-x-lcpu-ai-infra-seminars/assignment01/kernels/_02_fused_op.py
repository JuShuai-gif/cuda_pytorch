"""问题 7.2：fused elementwise（改造题）。

scale_kernel 目前功能完整，相应代码不要变动。
fused_kernel 目前和 scale_kernel 完全一致，是你需要修改的 kernel。
任务：改成 z = relu(a * x + b)，其中 a、b 是标量。
TIP: 只需要动计算那一行，再把 a、b 传进 kernel——主体不变，
这正是 Tile 视角的好处:-)。改完运行：
    pytest tests/test_fused_op.py
"""

import torch
import triton
import triton.language as tl


@triton.jit
def scale_kernel(x_ptr, z_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    z = x * 2.0
    tl.store(z_ptr + offsets, z, mask=mask)


def scale(x: torch.Tensor) -> torch.Tensor:
    z = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    scale_kernel[grid](x, z, n, BLOCK_SIZE=BLOCK_SIZE)
    return z


# ====== 从这里开始改 ======


# kernel fusion: 将 a*x、+b、ReLU 三次运算合并到一个 kernel 里完成。
# 用 PyTorch 写 z = relu(a*x + b) 会拆成 3 个 kernel（mul→add→relu），
# 中间结果 a*x 和 a*x+b 各需一次显存 round-trip（写回再读出）。
# 这里只读一次 x、只写一次 z，中间值全在寄存器里，节省带宽。
@triton.jit
def fused_kernel(x_ptr, z_ptr, n, a, b, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    # pid * BLOCK_SIZE：表示每个块起始位置
    # tl.arange(0,BLOCK_SIZE): 表示每个块内的索引号
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # x_ptr + offsets: 计算实际要读取的 GPU 内存地址(基地址 + 偏移量)
    # mask = mask：布尔掩码，False的位置不会真正去读内存（避免越界访问）
    # other=0.0 — 被 mask 掉的位置填充 0.0
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    z = tl.maximum(a * x + b, 0.0)  # relu(a * x + b)
    tl.store(z_ptr + offsets, z, mask=mask)


def fused(x: torch.Tensor, a: float, b: float) -> torch.Tensor:
    z = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    fused_kernel[grid](x, z, n, a, b, BLOCK_SIZE=BLOCK_SIZE)
    return z
