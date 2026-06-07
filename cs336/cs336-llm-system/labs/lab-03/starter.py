"""
Lab 03: Systems — Kernels & Parallelism — 起始代码

完成以下内容:
  - Triton 融合 RMSNorm 前向 kernel
  - DDP 分析问题
  - Benchmarking 脚本
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Triton 可能并非在所有系统上可用; 包装导入
try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None  # type: ignore
    tl = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────
# 任务 2: 融合 RMSNorm Kernel (Triton)
# ──────────────────────────────────────────────────────────────────────

if HAS_TRITON:

    @triton.jit
    def rmsnorm_fwd_kernel(
        x_ptr,  # (B*L, D) 输入
        w_ptr,  # (D,) 权重
        y_ptr,  # (B*L, D) 输出
        rms_ptr,  # (B*L,) 输出 rms（供反向梯度使用）
        D: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        融合 RMSNorm 前向 kernel。

        每个 program 处理一行（索引为 B*L）。

        步骤:
          1. 将行以 BLOCK_SIZE 大小的 tile 加载到 SRAM
          2. 计算所有 tile 上的 sum(x^2)（以 FP32 累加）
          3. 计算 rms = sqrt(acc / D + eps)
          4. 对每个 tile: y = x * (1/rms) * w，写入输出
        """
        # TODO: 实现融合 RMSNorm 前向
        pass

    @triton.jit
    def rmsnorm_bwd_kernel(
        dy_ptr,  # (B*L, D) 梯度输出
        x_ptr,  # (B*L, D) 输入
        w_ptr,  # (D,) 权重
        rms_ptr,  # (B*L,) 前向时存储的 rms
        dx_ptr,  # (B*L, D) 梯度输入
        dw_ptr,  # (D,) 梯度权重（跨所有行累加）
        N: tl.constexpr,  # 总行数 (B*L)
        D: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        融合 RMSNorm 反向 kernel。

        每个 program 处理一行。

        步骤:
          1. 加载该行的 x, dy, w, rms
          2. 计算 dx = (1/rms) * w * (dy - x * (x^T dy) / (D * rms^2))
          3. 计算 dw contribution = x * dy / rms
          4. 写入 dx，累加 dw（如需要则使用 atomics）
        """
        # TODO: 实现 RMSNorm 反向
        pass


# ──────────────────────────────────────────────────────────────────────
# Triton RMSNorm 的 PyTorch 包装器
# ──────────────────────────────────────────────────────────────────────


class TritonRMSNorm(nn.Module):
    """使用 Triton kernel 实现的 RMSNorm。"""

    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) 或 (N, D)"""
        # TODO: 调用 rmsnorm_fwd_kernel
        # 1. 展平为 (N, D), 其中 N = B*L
        # 2. 以适当的 grid 启动 kernel
        # 3. 还原 shape
        raise NotImplementedError("TritonRMSNorm.forward() not implemented")


# ──────────────────────────────────────────────────────────────────────
# PyTorch 参考实现（用于正确性检查）
# ──────────────────────────────────────────────────────────────────────


def rmsnorm_pytorch(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """PyTorch 参考 RMSNorm 实现。"""
    rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
    return (x / rms).to(x.dtype) * weight


# ──────────────────────────────────────────────────────────────────────
# 任务 3: DDP 知识问答
# ──────────────────────────────────────────────────────────────────────


def answer_ddp_questions() -> str:
    """将你的答案写在此处，作为多行字符串。"""
    return """
Q1: DDP 中 AllReduce 梯度发生在哪个时机？为什么在 backward 而不是 forward？

YOUR ANSWER HERE

Q2: Gradient bucketing 是什么？为什么能提升性能？

YOUR ANSWER HERE

Q3: 如果使用 find_unused_parameters=True 会有什么性能影响？

YOUR ANSWER HERE
"""


# ──────────────────────────────────────────────────────────────────────
# 任务 4: Benchmarking
# ──────────────────────────────────────────────────────────────────────


def benchmark_rmsnorm(
    dims: Tuple[int, ...] = (1024, 4096, 8192),
    batch_size: int = 4096,
    num_iters: int = 100,
) -> None:
    """在不同 hidden dimension 下基准测试 PyTorch vs Triton RMSNorm。

    使用 CUDA events 进行精确的 GPU 计时。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA 不可用，跳过 benchmark。")
        return

    print(f"{'Dim':>8s}  {'PyTorch (μs)':>14s}  {'Triton (μs)':>14s}  {'Speedup':>8s}")
    print("-" * 52)

    for D in dims:
        x = torch.randn(batch_size, D, device=device, dtype=torch.float32)
        w = torch.ones(D, device=device, dtype=torch.float32)

        # 基准测试 PyTorch
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_iters):
            _ = rmsnorm_pytorch(x, w)
        end.record()
        torch.cuda.synchronize()
        pytorch_time = start.elapsed_time(end) / num_iters * 1000  # μs

        # 基准测试 Triton (TODO: 在 kernel 就绪后实现)
        # triton_time = ...  # TODO
        triton_time = float("nan")

        speedup = pytorch_time / triton_time if triton_time > 0 else float("nan")
        print(f"{D:>8d}  {pytorch_time:>14.1f}  {triton_time:>14.1f}  {speedup:>7.2f}x")


if __name__ == "__main__":
    print("Lab 03 starter — 实现 Triton kernel 并运行 benchmarks。")
    print(f"Triton 可用: {HAS_TRITON}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        benchmark_rmsnorm()
    else:
        print("未检测到 GPU。你仍然可以完成 Task 1 和 Task 3。")
