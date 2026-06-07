"""
CUDA event 计时，用于精确测量 GPU 操作。

CUDA events 提供 GPU 端的时间戳，能够准确测量 kernel 执行时间，
而不会受到 CPU 端的开销影响。对于 GPU 操作分析来说这一点至关重要，
因为 CPU 时间测量可能因异步执行而产生误导。

关键概念：
- torch.cuda.Event：GPU 端的标记点，可选启用计时功能。
- record()：在 GPU 流中的当前位置记录 event。
- synchronize()：等待所有 GPU 工作完成后，才继续在 CPU 上执行。
- elapsed_time()：返回两个已记录的 event 之间的时间差，单位为 milliseconds。
"""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn


class CUDATimer:
    """
    基于 CUDA event 的计时器，用于精确测量 GPU 操作。

    用法：
        timer = CUDATimer()
        timer.start()
        # ... GPU 操作 ...
        timer.stop()
        print(f"Operation took {timer.elapsed_ms():.2f} ms")
    """

    def __init__(self) -> None:
        self.start_event: torch.cuda.Event | None = None
        self.end_event: torch.cuda.Event | None = None
        self._elapsed: float = 0.0

    def start(self) -> None:
        """记录起始 event。"""
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()

    def stop(self) -> None:
        """记录结束 event 并同步。"""
        if self.end_event is not None:
            self.end_event.record()
            torch.cuda.synchronize()

    def elapsed_ms(self) -> float:
        """获取以 milliseconds 为单位的已用时间。"""
        if self.start_event is None or self.end_event is None:
            return 0.0
        return self.start_event.elapsed_time(self.end_event)

    def elapsed_s(self) -> float:
        """获取以秒为单位的已用时间。"""
        return self.elapsed_ms() / 1000.0


def benchmark_cuda_kernel(
    func, *args, warmup: int = 10, runs: int = 100, **kwargs
) -> dict[str, float]:
    """
    使用 CUDA events 对 CUDA kernel 函数进行 benchmark 测试。

    Args:
        func: 要测试的函数（必须接收和返回 CUDA tensors）。
        *args: 传递给函数的位置参数。
        warmup: 预热迭代次数。
        runs: 计时迭代次数。
        **kwargs: 传递给函数的关键字参数。

    Returns:
        包含 min、max、mean、median 时间（单位：milliseconds）的字典。
    """
    # 预热
    for _ in range(warmup):
        _ = func(*args, **kwargs)

    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(runs):
        timer = CUDATimer()
        timer.start()
        _ = func(*args, **kwargs)
        timer.stop()
        times.append(timer.elapsed_ms())

    times_sorted: list[float] = sorted(times)
    mean_time: float = sum(times) / len(times)
    median_time: float = times_sorted[len(times_sorted) // 2]

    return {
        "min_ms": times_sorted[0],
        "max_ms": times_sorted[-1],
        "mean_ms": mean_time,
        "median_ms": median_time,
        "std_ms": (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5,
        "num_runs": float(runs),
    }


def compare_cpu_vs_cuda_timing(
    model: nn.Module,
    input_ids: torch.Tensor,
    num_runs: int = 20,
) -> dict[str, Any]:
    """
    对比 CPU 端计时与 CUDA event 计时。

    演示了为什么 CUDA events 是必要的：CPU 时间测量
    包含了 Python 开销，可能无法反映实际的 GPU 执行时间。

    Args:
        model: 要进行 benchmark 测试的模型。
        input_ids: GPU 上的输入 tensor。
        num_runs: benchmark 运行的次数。

    Returns:
        包含 CPU 和 CUDA 计时对比结果的字典。
    """
    model.eval()

    # 预热
    for _ in range(5):
        _ = model(input_ids)

    torch.cuda.synchronize()

    # CPU 计时（对于 GPU 操作不够精确）
    cpu_times: list[float] = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = model(input_ids)
        torch.cuda.synchronize()
        cpu_times.append((time.perf_counter() - t0) * 1000)  # 转换为 ms

    # CUDA event 计时（精确）
    cuda_times: list[float] = []
    for _ in range(num_runs):
        timer = CUDATimer()
        timer.start()
        _ = model(input_ids)
        timer.stop()
        cuda_times.append(timer.elapsed_ms())

    cpu_mean: float = sum(cpu_times) / len(cpu_times)
    cuda_mean: float = sum(cuda_times) / len(cuda_times)

    return {
        "cpu_mean_ms": cpu_mean,
        "cpu_std_ms": (sum((t - cpu_mean) ** 2 for t in cpu_times) / len(cpu_times))
        ** 0.5,
        "cuda_mean_ms": cuda_mean,
        "cuda_std_ms": (sum((t - cuda_mean) ** 2 for t in cuda_times) / len(cuda_times))
        ** 0.5,
        "overhead_ms": cpu_mean - cuda_mean,
        "num_runs": num_runs,
    }


def profile_operation_pipeline() -> None:
    """
    演示如何使用 CUDA events 对多步骤 pipeline 进行分析。

    展示如何在前向传播过程中测量各个操作（linear、attention 等）
    以识别性能瓶颈。
    """
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping pipeline profiling demonstration.")
        return

    print("CUDA Event Pipeline Profiling")
    print("=" * 50)

    # 创建简单操作
    hidden = 768
    seq = 512
    batch = 4

    x = torch.randn(batch, seq, hidden, device="cuda")
    linear = nn.Linear(hidden, hidden * 4, device="cuda")
    linear2 = nn.Linear(hidden * 4, hidden, device="cuda")
    layernorm = nn.LayerNorm(hidden, device="cuda")

    # 预热
    for _ in range(5):
        y = linear(x)
        y = nn.functional.gelu(y)
        y = linear2(y)
        y = layernorm(y)

    torch.cuda.synchronize()

    # 对每个操作进行分析
    operations: dict[str, callable] = {
        "linear_up": lambda: linear(x),
        "gelu": lambda: nn.functional.gelu(linear(x)),
        "linear_down": lambda: nn.functional.gelu(linear(x)).pipe(linear2),
        "layernorm": lambda: layernorm(linear2(nn.functional.gelu(linear(x)))),
    }

    results: dict[str, dict[str, float]] = {}
    for name, op in operations.items():
        # 在完整 pipeline 的上下文中测量该操作
        def pipeline_step() -> torch.Tensor:
            h = linear(x)
            h = nn.functional.gelu(h)
            h = linear2(h)
            h = layernorm(h)
            return h

        # 完整 pipeline 的时间
        full_time = benchmark_cuda_kernel(pipeline_step, warmup=5, runs=20)
        full_ms: float = full_time["mean_ms"]

        # 不含该操作的时间（用于估算其贡献）
        # 这只是近似值；实际使用时建议用 torch.profiler 做详细分析
        results[name] = {
            "full_pipeline_ms": full_ms,
        }

    print(f"\n  Operation timing (seq_len={seq}, batch={batch}):")
    print(f"  Full pipeline mean: {full_ms:.3f} ms (all operations combined)")
    print(f"\n  Note: For detailed per-operation timing, use torch.profiler")
    print(f"  or the CUDATimer class within the model's forward pass.")


# 快速测试
if __name__ == "__main__":
    print("CUDA Event Timing Tools")
    print("=" * 50)

    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA devices: {torch.cuda.device_count()}")

        # 简单的 CUDA timer 测试
        timer = CUDATimer()
        timer.start()
        x = torch.randn(1000, 1000, device="cuda")
        y = torch.matmul(x, x.T)
        timer.stop()
        print(f"\nMatrix multiply (1000x1000 @ 1000x1000): {timer.elapsed_ms():.3f} ms")

        # 分析各操作
        profile_operation_pipeline()
    else:
        print("CUDA not available. Running CPU-only demonstration.")
        print("Install PyTorch with CUDA support for GPU timing features.")
        print()

        print("Key concepts:")
        print("  1. CUDA events provide GPU-side timestamps")
        print("  2. torch.cuda.Event(enable_timing=True) creates a timing event")
        print("  3. .record() marks the event in the GPU stream")
        print("  4. .synchronize() waits for GPU to finish before reading time")
        print("  5. .elapsed_time() returns time between two events in ms")
        print()
        print("  CPU timing (time.perf_counter()) includes Python overhead")
        print("  CUDA event timing measures actual GPU kernel execution time")
        print("  The difference can be significant (tens of microseconds per call)")

    print("\nCUDA timing module loaded successfully!")
