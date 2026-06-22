#!/usr/bin/env python3
"""
CUDA streams and async operations benchmark.

Measures:
  - Throughput improvement with overlapping H2D+compute+D2H
  - Pinned vs pageable memory transfer bandwidth
  - Stream sync overhead
  - Multi-stream concurrency performance (2, 4, 8 streams)
  - Double-buffering pipeline improvement
  - CUDA C++ native stream kernel vs Python torch.cuda.Stream() comparison

Run: python 07_cuda_streams_async/benchmark_cuda_streams.py
"""

from __future__ import annotations

import sys
import time
from typing import Callable

import torch
import triton
import triton.language as tl

from benchmarks.benchmark_utils import (
    BenchmarkConfig,
    BenchmarkResult,
    benchmark_kernel,
    compare_kernels,
    generate_report,
)

# 检测 CUDA stream C++ 扩展是否可用
_STREAM_KERNELS_AVAILABLE = False
try:
    import cuda_stream_kernels  # type: ignore[import-not-found]

    _STREAM_KERNELS_AVAILABLE = True
except ImportError:
    pass

CONFIG = BenchmarkConfig(warmup_steps=3, measure_steps=15, repeat=3)


# ---------------------------------------------------------------------------
# Benchmark kernels
# ---------------------------------------------------------------------------


@triton.jit
def _bench_add_kernel(x_ptr, y_ptr, out_ptr, n_elements: int, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


@triton.jit
def _bench_mul_kernel(x_ptr, out_ptr, n_elements: int, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x * x, mask=mask)


def _launch_bench_add(
    x: torch.Tensor, y: torch.Tensor, out: torch.Tensor, stream: torch.cuda.Stream | None = None
) -> None:
    n = x.numel()
    BLOCK_SIZE = 256
    grid = triton.cdiv(n, BLOCK_SIZE)
    if stream is not None:
        with torch.cuda.stream(stream):
            _bench_add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)
    else:
        _bench_add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)


def _launch_bench_mul(
    x: torch.Tensor, out: torch.Tensor, stream: torch.cuda.Stream | None = None
) -> None:
    n = x.numel()
    BLOCK_SIZE = 256
    grid = triton.cdiv(n, BLOCK_SIZE)
    if stream is not None:
        with torch.cuda.stream(stream):
            _bench_mul_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
    else:
        _bench_mul_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)


# ---------------------------------------------------------------------------
# Pinned vs pageable memory bandwidth
# ---------------------------------------------------------------------------


def bench_pinned_vs_pageable() -> list[BenchmarkResult]:
    """Measure H2D transfer bandwidth for pinned vs pageable memory."""
    print(f"\n{'=' * 60}")
    print("  PINNED vs PAGEABLE MEMORY BANDWIDTH")
    print(f"{'=' * 60}")

    sizes = [
        1024 * 1024,
        4 * 1024 * 1024,
        16 * 1024 * 1024,
        64 * 1024 * 1024,
        128 * 1024 * 1024,
    ]
    results: list[BenchmarkResult] = []

    for n in sizes:
        elements = n // 4  # float32 = 4 bytes
        dev = torch.empty(elements, device="cuda", dtype=torch.float32)

        # Pinned memory H2D
        host_pinned = torch.randn(elements, dtype=torch.float32, pin_memory=True)
        r = benchmark_kernel(
            fn=lambda h, d: d.copy_(h, non_blocking=False),
            args=(host_pinned, dev),
            name=f"h2d_pinned_{n // (1024 * 1024)}MB",
            config=CONFIG,
        )
        results.append(r)

        # Pageable memory H2D
        host_pageable = torch.randn(elements, dtype=torch.float32)
        r = benchmark_kernel(
            fn=lambda h, d: d.copy_(h, non_blocking=False),
            args=(host_pageable, dev),
            name=f"h2d_pageable_{n // (1024 * 1024)}MB",
            config=CONFIG,
        )
        results.append(r)

    compare_kernels(results)
    return results


# ---------------------------------------------------------------------------
# Multi-stream concurrency
# ---------------------------------------------------------------------------


def bench_multi_stream_concurrency() -> list[BenchmarkResult]:
    """Benchmark performance with 1, 2, 4, 8 concurrent streams."""
    print(f"\n{'=' * 60}")
    print("  MULTI-STREAM CONCURRENCY")
    print(f"{'=' * 60}")

    n = 5_000_000
    num_streams_list = [1, 2, 4, 8]
    results: list[BenchmarkResult] = []

    for num_streams in num_streams_list:
        streams = [torch.cuda.Stream() for _ in range(num_streams)]

        # Prepare data for each stream
        xs = [torch.randn(n, device="cuda", dtype=torch.float32) for _ in range(num_streams)]
        ys = [torch.randn(n, device="cuda", dtype=torch.float32) for _ in range(num_streams)]
        outs = [torch.empty_like(xs[0]) for _ in range(num_streams)]

        def _run_multi() -> None:
            for i in range(num_streams):
                with torch.cuda.stream(streams[i]):
                    _launch_bench_add(xs[i], ys[i], outs[i])

        r = benchmark_kernel(
            fn=_run_multi,
            name=f"streams_{num_streams}",
            config=CONFIG,
        )
        results.append(r)

        # Sync all streams
        for s in streams:
            s.synchronize()
        torch.cuda.synchronize()

    compare_kernels(results)
    return results


# ---------------------------------------------------------------------------
# Stream sync overhead
# ---------------------------------------------------------------------------


def bench_stream_sync_overhead() -> list[BenchmarkResult]:
    """Measure synchronization overhead for different patterns."""
    print(f"\n{'=' * 60}")
    print("  STREAM SYNC OVERHEAD")
    print(f"{'=' * 60}")

    n = 2_000_000
    results: list[BenchmarkResult] = []

    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    # No sync (just launch)
    def _launch_only():
        _launch_bench_add(x, y, out)

    r = benchmark_kernel(fn=_launch_only, name="no_sync", config=CONFIG)
    results.append(r)

    # Stream sync
    def _stream_sync():
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            _launch_bench_add(x, y, out)
        s.synchronize()

    r = benchmark_kernel(fn=_stream_sync, name="stream_sync", config=CONFIG)
    results.append(r)

    # Global sync
    def _global_sync():
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            _launch_bench_add(x, y, out)
        torch.cuda.synchronize()

    r = benchmark_kernel(fn=_global_sync, name="global_sync", config=CONFIG)
    results.append(r)

    compare_kernels(results)
    return results


# ---------------------------------------------------------------------------
# Double-buffering pipeline benchmark
# ---------------------------------------------------------------------------


def bench_double_buffering() -> list[BenchmarkResult]:
    """Compare sequential vs double-buffered H2D+compute+D2H pipeline."""
    print(f"\n{'=' * 60}")
    print("  DOUBLE-BUFFERING PIPELINE")
    print(f"{'=' * 60}")

    chunk_size = 4_000_000
    num_chunks = 4
    results: list[BenchmarkResult] = []

    # Pinned host arrays
    host_ins = [
        torch.randn(chunk_size, dtype=torch.float32, pin_memory=True) for _ in range(num_chunks)
    ]
    host_outs = [
        torch.empty(chunk_size, dtype=torch.float32, pin_memory=True) for _ in range(num_chunks)
    ]

    dev_buf_a = torch.empty(chunk_size, device="cuda", dtype=torch.float32)
    dev_buf_b = torch.empty(chunk_size, device="cuda", dtype=torch.float32)
    dev_out_a = torch.empty(chunk_size, device="cuda", dtype=torch.float32)
    dev_out_b = torch.empty(chunk_size, device="cuda", dtype=torch.float32)

    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()

    # --- Sequential ---
    def _sequential():
        dev_seq = torch.empty(chunk_size, device="cuda", dtype=torch.float32)
        dev_seq_out = torch.empty(chunk_size, device="cuda", dtype=torch.float32)
        for i in range(num_chunks):
            dev_seq.copy_(host_ins[i])
            _launch_bench_mul(dev_seq, dev_seq_out)
            host_outs[i].copy_(dev_seq_out)

    r = benchmark_kernel(fn=_sequential, name="sequential_pipeline", config=CONFIG)
    results.append(r)

    # --- Double-buffered ---
    def _double_buffered():
        for i in range(0, num_chunks, 2):
            with torch.cuda.stream(stream_a):
                dev_buf_a.copy_(host_ins[i], non_blocking=True)
                _launch_bench_mul(dev_buf_a, dev_out_a)
                host_outs[i].copy_(dev_out_a, non_blocking=True)

            if i + 1 < num_chunks:
                with torch.cuda.stream(stream_b):
                    dev_buf_b.copy_(host_ins[i + 1], non_blocking=True)
                    _launch_bench_mul(dev_buf_b, dev_out_b)
                    host_outs[i + 1].copy_(dev_out_b, non_blocking=True)

        stream_a.synchronize()
        stream_b.synchronize()

    r = benchmark_kernel(fn=_double_buffered, name="double_buffered", config=CONFIG)
    results.append(r)

    compare_kernels(results)

    # Calculate improvement
    if len(results) >= 2:
        ratio = results[0].p50_ms / results[1].p50_ms if results[1].p50_ms > 0 else 0
        print(f"\n  Double-buffering speedup: {ratio:.2f}x")

    return results


# ---------------------------------------------------------------------------
# CUDA C++ native stream kernel benchmark
# ---------------------------------------------------------------------------


def bench_cuda_native_vs_python_streams() -> list[BenchmarkResult]:
    """对比 CUDA C++ 原生 stream kernel vs Python torch.cuda.Stream()。

    CUDA C++ 原生方式：
      - 直接在 .cu 文件中管理 cudaStream_t / cudaEvent_t
      - 无 Python 层开销，无 Torch stream 对象创建/销毁
      - 可直接使用 cudaMemcpyAsync 等底层 API

    Python torch.cuda.Stream() 方式：
      - 通过 PyTorch 的 Python 绑定间接使用 CUDA stream API
      - 有 Python 对象创建开销和 GIL 开销
    """
    if not _STREAM_KERNELS_AVAILABLE:
        print("\n  (cuda_stream_kernels 扩展未构建，跳过此 benchmark)")
        return []

    print(f"\n{'=' * 60}")
    print("  CUDA C++ NATIVE vs PYTHON STREAMS")
    print(f"{'=' * 60}")

    n = 5_000_000
    results: list[BenchmarkResult] = []

    # --- Python torch.cuda.Stream() 多 stream 并发 ---
    num_streams_list = [2, 4]
    for num_streams in num_streams_list:
        streams = [torch.cuda.Stream() for _ in range(num_streams)]
        xs = [torch.randn(n, device="cuda", dtype=torch.float32) for _ in range(num_streams)]
        ys = [torch.randn(n, device="cuda", dtype=torch.float32) for _ in range(num_streams)]
        outs = [torch.empty_like(xs[0]) for _ in range(num_streams)]

        def _run_python_streams():
            for i in range(num_streams):
                with torch.cuda.stream(streams[i]):
                    _launch_bench_add(xs[i], ys[i], outs[i])

        r = benchmark_kernel(
            fn=_run_python_streams,
            name=f"python_streams_{num_streams}",
            config=CONFIG,
        )
        results.append(r)

        for s in streams:
            s.synchronize()

    # --- CUDA C++ 原生 stream 并发 ---
    for num_streams in num_streams_list:
        a_list = [torch.randn(n, device="cuda", dtype=torch.float32) for _ in range(num_streams)]
        b_list = [torch.randn(n, device="cuda", dtype=torch.float32) for _ in range(num_streams)]
        out_list = [torch.empty_like(a_list[0]) for _ in range(num_streams)]

        def _run_cuda_native_streams():
            import cuda_stream_kernels

            cuda_stream_kernels.multi_stream_concurrent_exec(a_list, b_list, out_list)

        r = benchmark_kernel(
            fn=_run_cuda_native_streams,
            name=f"cuda_native_streams_{num_streams}",
            config=CONFIG,
        )
        results.append(r)

    # --- CUDA C++ 原生 WAR 同步对比 ---
    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)

    def _war_wrong():
        import cuda_stream_kernels

        cuda_stream_kernels.war_sync_correct_vs_wrong(a, b)

    r = benchmark_kernel(fn=_war_wrong, name="cuda_native_war_sync", config=CONFIG)
    results.append(r)

    compare_kernels(results)
    return results


def main():
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        sys.exit(1)

    print("=" * 70)
    print("  CUDA STREAMS & ASYNC BENCHMARKS")
    print("=" * 70)
    print(f"\n  Device: {torch.cuda.get_device_name(0)}")
    print(f"  PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print(f"  Triton: {triton.__version__}")
    if _STREAM_KERNELS_AVAILABLE:
        print("  CUDA C++ stream kernels: available")
    else:
        print("  CUDA C++ stream kernels: not built")

    all_results: list[BenchmarkResult] = []
    all_results.extend(bench_pinned_vs_pageable())
    all_results.extend(bench_multi_stream_concurrency())
    all_results.extend(bench_stream_sync_overhead())
    all_results.extend(bench_double_buffering())
    all_results.extend(bench_cuda_native_vs_python_streams())

    print(f"\n{'=' * 70}")
    print("  FINAL SUMMARY")
    print(f"{'=' * 70}")
    compare_kernels(all_results)

    report_md = generate_report(all_results, "07_cuda_streams_async/streams_report")
    print(f"\n{report_md}")


if __name__ == "__main__":
    main()
