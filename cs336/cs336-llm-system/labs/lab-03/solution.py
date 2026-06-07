"""
Lab 03 解答: Systems — Kernels & Parallelism

完整的 Triton RMSNorm kernel + DDP 答案 + benchmarks。
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None


# ══════════════════════════════════════════════════════════════════════
# 任务 2: 融合 RMSNorm Kernel
# ══════════════════════════════════════════════════════════════════════

if HAS_TRITON:

    @triton.jit
    def rmsnorm_fwd_kernel(
        x_ptr,
        w_ptr,
        y_ptr,
        rms_ptr,
        N: tl.constexpr,  # 总行数 = B * L  (用于 grid 大小)
        D: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """融合 RMSNorm 前向: 每个 program 处理一行。"""
        pid = tl.program_id(0)  # 行索引
        if pid >= N:
            return

        # 该行的偏移量
        row_start = pid * D
        offsets = row_start + tl.arange(0, BLOCK_SIZE)
        mask = tl.arange(0, BLOCK_SIZE) < D

        # 以 FP32 累加 sum(x^2)
        acc = tl.zeros([1], dtype=tl.float32)
        for block_start in range(0, D, BLOCK_SIZE):
            off = row_start + block_start + tl.arange(0, BLOCK_SIZE)
            m = (block_start + tl.arange(0, BLOCK_SIZE)) < D
            x = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32)
            acc += tl.sum(x * x)

        # rms = sqrt(mean(x^2) + eps)
        rms_val = tl.sqrt(acc / D + eps)
        tl.store(rms_ptr + pid, rms_val)

        # 归一化并应用权重
        for block_start in range(0, D, BLOCK_SIZE):
            off = row_start + block_start + tl.arange(0, BLOCK_SIZE)
            m = (block_start + tl.arange(0, BLOCK_SIZE)) < D
            x = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32)
            w = tl.load(
                w_ptr + (block_start + tl.arange(0, BLOCK_SIZE)), mask=m, other=0.0
            )
            y = x * w / rms_val
            tl.store(y_ptr + off, y.to(x.dtype.element_ty), mask=m)

    @triton.jit
    def rmsnorm_bwd_kernel(
        dy_ptr,
        x_ptr,
        w_ptr,
        rms_ptr,
        dx_ptr,
        dw_ptr,
        N: tl.constexpr,
        D: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """融合 RMSNorm 反向: 每个 program 处理一行。"""
        pid = tl.program_id(0)
        if pid >= N:
            return

        row_start = pid * D
        rms_val = tl.load(rms_ptr + pid)

        # 对每个 block: 计算 dx 并累加 dw
        for block_start in range(0, D, BLOCK_SIZE):
            off = row_start + block_start + tl.arange(0, BLOCK_SIZE)
            w_off = block_start + tl.arange(0, BLOCK_SIZE)
            m = (block_start + tl.arange(0, BLOCK_SIZE)) < D

            x = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32)
            dy = tl.load(dy_ptr + off, mask=m, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + w_off, mask=m, other=0.0)

            # dx = (1/rms) * w * dy
            dx = w * dy / rms_val

            # dw contribution = x * (dy / rms) ... 简化版:
            # 完整的 RMSNorm 反向:
            #   dx = w/rms * dy  (在此演示中忽略均值修正以简化)
            #   dw = x * dy / rms
            # 对于生产级 kernel，需要包含完整的梯度路径，
            # 包括 rms 梯度路径。此简化版仅供教学使用。

            dw_contrib = x * dy / rms_val

            tl.store(dx_ptr + off, dx.to(x.dtype.element_ty), mask=m)
            tl.atomic_add(dw_ptr + w_off, dw_contrib, mask=m)


# ──────────────────────────────────────────────────────────────────────
# PyTorch 包装器
# ──────────────────────────────────────────────────────────────────────


class TritonRMSNorm(nn.Module):
    """基于 Triton kernel 的 RMSNorm。"""

    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not HAS_TRITON:
            return rmsnorm_pytorch(x, self.weight, self.eps)

        orig_shape = x.shape
        x_flat = x.reshape(-1, self.hidden_dim)
        N, D = x_flat.shape

        y = torch.empty_like(x_flat)
        rms = torch.empty(N, device=x.device, dtype=torch.float32)

        BLOCK_SIZE = min(1024, triton.next_power_of_2(D))
        grid = (N,)

        rmsnorm_fwd_kernel[grid](
            x_flat,
            self.weight,
            y,
            rms,
            N=N,
            D=D,
            eps=self.eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return y.reshape(orig_shape)


# ──────────────────────────────────────────────────────────────────────
# PyTorch 参考实现
# ──────────────────────────────────────────────────────────────────────


def rmsnorm_pytorch(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
    return (x / rms).to(x.dtype) * weight


# ══════════════════════════════════════════════════════════════════════
# 任务 3: DDP 答案
# ══════════════════════════════════════════════════════════════════════


def answer_ddp_questions() -> str:
    return """
Q1: DDP 中 AllReduce 梯度发生在哪个时机？为什么在 backward 而不是 forward？
────────────────────────────────────────────────────────────────────────
Answer:
DDP 的 gradient AllReduce 发生在 backward pass 期间，而不是 forward 之后。
具体机制如下：

- DDP 在构建时为每个参数注册了 backward hook (autograd hook)
- 当某个参数的 .grad 被计算出来时（在 backward 过程中），
  hook 被触发，该参数的梯度被放入相应的 gradient bucket
- 当一个 bucket 满了（默认 25MB），DDP 立即启动 AllReduce
- 这个 AllReduce 是异步的，可以与后续层的 backward 计算 overlap

为什么不在 forward 之后？
- 在 forward 时，梯度还不存在（梯度是在 backward 中计算的）
- 在 forward 中 AllReduce 只能做 activation/weight 同步，
  但 DDP 的设计目标是同步梯度而非 activation

Q2: Gradient bucketing 是什么？为什么能提升性能？
──────────────────────────────────────────────────
Answer:
Gradient bucketing 是将多个参数的梯度打包成一个 bucket，
对 bucket（而非单个参数）执行一次 AllReduce。

好处：
1. 减少通信次数：每个 bucket 一次 AllReduce vs 每个参数一次
2. 更大的 message size → 更好的网络带宽利用率
   （小 packet 的 latency 占主导，大 packet 的 bandwidth 占主导）
3. Overlap with compute：在 backward 计算后续层时，
   前面 bucket 的 AllReduce 可以异步进行
4. 减少 kernel launch overhead

实现细节：
- bucket 按模型反向顺序构建（从最后一层到第一层）
- 默认 bucket_size = 25MB，可通过 bucket_cap_mb 调整
- 当 bucket 满或 backward 完成时，触发 AllReduce

Q3: 如果使用 find_unused_parameters=True 会有什么性能影响？
─────────────────────────────────────────────────────────────
Answer:
find_unused_parameters=True 会带来以下性能影响：

1. 额外的前向 pass：DDP 需要做一次额外的 forward 来检测
   哪些参数在 forward 中未被使用（没有参与计算图的构建）

2. 无法使用 gradient bucketing 的 overlap 优化：
   DDP 必须等待 backward 完全结束，确认所有 unused parameters
   之后才能启动 AllReduce。这意味着：
   - 通信与计算无法 overlap
   - 所有 AllReduce 都在 backward 之后串行执行

3. 额外的内存开销：需要存储参数使用状态的标记

4. 对于有大比例 unused parameters 的模型（如某些多任务模型），
   性能可能会下降 10-30%。

建议：
- 如果确定所有参数都会在 forward 中使用，设置 find_unused_parameters=False
- 如果确实有 unused parameters，考虑重构模型使其始终使用全部参数
- 或用 torch.compile 的 dynamic=False 来消除 unused parameters
"""


# ══════════════════════════════════════════════════════════════════════
# 任务 4: Benchmarks
# ══════════════════════════════════════════════════════════════════════


def benchmark_rmsnorm(
    dims: Tuple[int, ...] = (1024, 4096, 8192),
    batch_size: int = 4096,
    num_iters: int = 100,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA 不可用，跳过 benchmark。")
        return

    print(f"Benchmark: RMSNorm (batch={batch_size}, iters={num_iters})")
    print(
        f"{'Dim':>8s}  {'PyTorch (μs)':>14s}  {'Triton (μs)':>14s}  {'Speedup':>8s}  {'Correct':>8s}"
    )
    print("-" * 72)

    for D in dims:
        x = torch.randn(batch_size, D, device=device, dtype=torch.float32)
        w = torch.ones(D, device=device, dtype=torch.float32)

        # 预热
        for _ in range(10):
            _ = rmsnorm_pytorch(x, w)

        # 基准测试 PyTorch
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_iters):
            _ = rmsnorm_pytorch(x, w)
        end.record()
        torch.cuda.synchronize()
        pytorch_us = start.elapsed_time(end) / num_iters * 1000

        # 基准测试 Triton
        if HAS_TRITON:
            triton_norm = TritonRMSNorm(D).to(device)
            for _ in range(10):
                _ = triton_norm(x)
            torch.cuda.synchronize()
            start.record()
            for _ in range(num_iters):
                _ = triton_norm(x)
            end.record()
            torch.cuda.synchronize()
            triton_us = start.elapsed_time(end) / num_iters * 1000

            # 正确性检查
            ref = rmsnorm_pytorch(x, w)
            out = triton_norm(x)
            is_correct = torch.allclose(ref, out, atol=1e-4)
        else:
            triton_us = float("nan")
            is_correct = False

        speedup = pytorch_us / triton_us if triton_us > 0 else float("nan")
        print(
            f"{D:>8d}  {pytorch_us:>14.1f}  {triton_us:>14.1f}  "
            f"{speedup:>7.2f}x  {'OK' if is_correct else 'FAIL'}"
        )


# ══════════════════════════════════════════════════════════════════════
# 验证
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Lab 03 解答验证 ===\n")

    # 检查 Triton 可用性
    print(f"Triton 可用: {HAS_TRITON}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    # 正确性测试
    if HAS_TRITON and torch.cuda.is_available():
        device = torch.device("cuda")
        D = 4096
        norm = TritonRMSNorm(D).to(device)
        x = torch.randn(2, 128, D, device=device, dtype=torch.float32)

        ref = rmsnorm_pytorch(x, norm.weight)
        out = norm(x)

        max_diff = (ref - out).abs().max().item()
        print(f"Triton vs PyTorch RMSNorm 最大差异: {max_diff:.2e}")
        print(f"通过: {torch.allclose(ref, out, atol=1e-4)}")

        # Benchmark
        print()
        benchmark_rmsnorm()

    # DDP 答案
    print("\n" + "=" * 60)
    print("DDP 知识问答")
    print("=" * 60)
    print(answer_ddp_questions())
