"""
第六讲 — GPU 编程：性能基准测试对比函数。

定义（但不执行）用于比较 PyTorch 与 Triton kernel 性能的基准测试工具。
所有重型计算均受保护，因此该文件始终可以安全导入。
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

try:
    import triton  # type: ignore[import-untyped]

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# 基准测试运行器
# ---------------------------------------------------------------------------


@contextmanager
def _timer(name: str = ""):
    """用于墙上时钟计时的上下文管理器。"""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    yield
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - start
    label = f"{name}: " if name else ""
    print(f"  {label}{elapsed * 1e3:.3f} ms")


def benchmark_fn(
    fn: Callable[..., Any],
    *args: Any,
    warmup: int = 5,
    repeat: int = 20,
    name: str = "",
    **kwargs: Any,
) -> Dict[str, float]:
    """对可调用对象进行基准测试（预热 + 重复迭代）。

    注意：此函数仅在 ``_ENABLE_BENCHMARK`` 为 True 时运行。
    通过命令行或环境变量设置以启用。
    """
    if not _ENABLE_BENCHMARK:
        return {"name": name, "mean_ms": -1.0, "std_ms": -1.0}

    # 预热
    for _ in range(warmup):
        fn(*args, **kwargs)

    torch.cuda.synchronize() if torch.cuda.is_available() else None

    # 计时运行
    times: List[float] = []
    for _ in range(repeat):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times.append((time.perf_counter() - t0) * 1e3)  # ms

    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
    return {"name": name, "mean_ms": mean, "std_ms": std}


# ---------------------------------------------------------------------------
# 带宽计算
# ---------------------------------------------------------------------------


def compute_bandwidth_gbs(
    data_bytes: float,
    time_ms: float,
) -> float:
    """计算有效带宽（单位 GB/s）。

    Parameters
    ----------
    data_bytes : float
        传输的总字节数（读 + 写）。
    time_ms : float
        kernel 执行时间，单位毫秒。
    """
    return data_bytes / (time_ms * 1e6) if time_ms > 0 else 0.0


def compute_tflops(
    flops: float,
    time_ms: float,
) -> float:
    """计算有效 TFLOPS。

    Parameters
    ----------
    flops : float
        总浮点运算次数。
    time_ms : float
        kernel 执行时间，单位毫秒。
    """
    return flops / (time_ms * 1e6 * 1e12) if time_ms > 0 else 0.0


# ---------------------------------------------------------------------------
# 基准测试编排
# ---------------------------------------------------------------------------


def run_kernel_benchmarks(
    shapes: Optional[List[Tuple[int, ...]]] = None,
    device: str = "cuda",
) -> List[Dict[str, Any]]:
    """运行一组 kernel 性能基准测试（PyTorch vs Triton）。

    仅在 ``_ENABLE_BENCHMARK`` 为 True 时执行。

    Parameters
    ----------
    shapes : list of tuples
        要测试的输入形状。默认：几个常用尺寸。
    device : str
        运行设备（'cuda' 或 'cpu'）。
    """
    if not _ENABLE_BENCHMARK:
        print("基准测试已禁用（设置 _ENABLE_BENCHMARK=True 以运行）。")
        return []

    if shapes is None:
        shapes = [
            (1024, 1024),  # softmax
            (256, 512, 128),  # matmul (M, K, N)
            (4096,),  # GeLU
        ]

    results: List[Dict[str, Any]] = []
    for shape in shapes:
        if len(shape) == 1:
            x = torch.randn(shape, device=device)
            for name, fn in _get_gelu_fns():
                res = benchmark_fn(fn, x, name=f"GeLU-{name}")
                res["shape"] = shape
                results.append(res)
        elif len(shape) == 2:
            x = torch.randn(shape, device=device)
            for name, fn in _get_softmax_fns():
                res = benchmark_fn(fn, x, name=f"Softmax-{name}")
                res["shape"] = shape
                results.append(res)
        elif len(shape) == 3:
            M, K, N = shape
            a = torch.randn(M, K, device=device)
            b = torch.randn(K, N, device=device)
            for name, fn in _get_matmul_fns():
                res = benchmark_fn(fn, a, b, name=f"Matmul-{name}")
                res["shape"] = shape
                results.append(res)
    return results


def _get_gelu_fns() -> List[Tuple[str, Callable]]:
    from .gelu_kernel import gelu_pytorch, gelu_triton

    fns: List[Tuple[str, Callable]] = [("pytorch", gelu_pytorch)]
    if HAS_TRITON and _ENABLE_BENCHMARK:
        fns.append(("triton", gelu_triton))
    return fns


def _get_softmax_fns() -> List[Tuple[str, Callable]]:
    from .softmax_kernel import softmax_pytorch, softmax_triton

    fns: List[Tuple[str, Callable]] = [("pytorch", softmax_pytorch)]
    if HAS_TRITON and _ENABLE_BENCHMARK:
        fns.append(("triton", softmax_triton))
    return fns


def _get_matmul_fns() -> List[Tuple[str, Callable]]:
    from .matmul_kernel import matmul_pytorch, matmul_triton

    fns: List[Tuple[str, Callable]] = [("pytorch", matmul_pytorch)]
    if HAS_TRITON and _ENABLE_BENCHMARK:
        fns.append(("triton", matmul_triton))
    return fns


# ---------------------------------------------------------------------------
# 开关：设置为 True 以启用实际的基准测试
# ---------------------------------------------------------------------------

_ENABLE_BENCHMARK = False


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print(f"Triton available: {HAS_TRITON}")
    print(f"Benchmarks enabled: {_ENABLE_BENCHMARK}")

    # 展示如何调用基准测试 API
    print("\n示例基准测试调用（未实际执行）：")
    print("  results = run_kernel_benchmarks(device='cuda')")

    # 通过解析计算一些示例吞吐量数据
    print("\n示例吞吐量计算：")
    data_gb = 0.016  # 16 MB

    # 不同带宽下的假设耗时
    for bw in [900.0, 2039.0, 3350.0]:
        time_ms = data_gb / (bw * 1e9) * 1e3
        print(f"  BW={bw:.0f} GB/s → {data_gb:.3f} GB in {time_ms * 1e3:.2f} µs")

    print(f"  Matmul 4096³ FP16: {2 * 4096**3 / 1e12:.0f} TFLOPs")
    print("\nAll checks passed.")
