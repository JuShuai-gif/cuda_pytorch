#!/usr/bin/env python3
"""
Comprehensive bandwidth benchmark for memory operations.

Tests:
  - Various tensor sizes (1KB to 1GB)
  - Contiguous vs strided access
  - Triton copy vs torch.clone
  - Achieved % of peak bandwidth
  - Bandwidth vs size chart data generation
  - CUDA C++ native float/float2/float4 vectorized access comparison
  - CUDA C++ strided access degradation measurement

Run: python 03_memory_bandwidth/benchmark_memory_bandwidth.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch
import triton

from analysis import get_peak_bandwidth
from benchmarks.benchmark_utils import (
    BenchmarkConfig,
    BenchmarkResult,
    benchmark_kernel,
    compare_kernels,
    generate_report,
)
from triton_copy import copy_kernel, copy_non_contiguous, copy_vectorized

# 检测 CUDA bandwidth C++ 扩展是否可用
_BANDWIDTH_KERNELS_AVAILABLE = False
try:
    import cuda_bandwidth_kernels  # type: ignore[import-not-found]

    _BANDWIDTH_KERNELS_AVAILABLE = True
except ImportError:
    pass


CONFIG = BenchmarkConfig(warmup_steps=5, measure_steps=30, repeat=3)


def bench_contiguous_vs_strided() -> list[BenchmarkResult]:
    """Benchmark contiguous elementwise add vs strided (transposed) add."""
    print(f"\n{'=' * 60}")
    print("  CONTIGUOUS vs STRIDED ACCESS")
    print(f"{'=' * 60}")

    sizes = [2**15, 2**18, 2**20, 2**22, 2**24]
    results = []

    for n in sizes:
        x = torch.ones(n, device="cuda", dtype=torch.float32)
        y = torch.ones(n, device="cuda", dtype=torch.float32)

        # Contiguous
        def contiguous_add():
            return x * y

        r_c = benchmark_kernel(
            fn=contiguous_add,
            name=f"contiguous_{n}",
            config=CONFIG,
        )
        results.append(r_c)

        # Strided (transposed 2D)
        side = int(n**0.5)
        if side >= 2 and side * side <= n:
            x2d = torch.ones(side, side, device="cuda", dtype=torch.float32)
            y2d = torch.ones(side, side, device="cuda", dtype=torch.float32)
            x_t = x2d.t()
            y_t = y2d.t()

            def strided_add():
                return x_t * y_t

            r_s = benchmark_kernel(
                fn=strided_add,
                name=f"strided_{n}",
                config=CONFIG,
            )
            results.append(r_s)

    print(f"\n  {'Size':>10}  {'Contiguous':>15}  {'Strided':>15}  {'Ratio':>8}")
    print(f"  {'-' * 10}  {'-' * 15}  {'-' * 15}  {'-' * 8}")
    for i in range(0, len(results), 2):
        if i + 1 < len(results):
            c_bw = results[i].bandwidth_gb_s
            s_bw = results[i + 1].bandwidth_gb_s
            ratio = c_bw / s_bw if s_bw > 0 else 0
            size_label = results[i].name.split("_")[-1]
            print(f"  {size_label:>10}  {c_bw:>13.1f} GB/s  {s_bw:>13.1f} GB/s  {ratio:>6.1f}x")

    return results


def bench_triton_copy_vs_torch() -> list[BenchmarkResult]:
    """Benchmark Triton copy kernels vs torch.clone."""
    print(f"\n{'=' * 60}")
    print("  TRITON COPY vs torch.clone")
    print(f"{'=' * 60}")

    sizes = [2**15, 2**18, 2**20, 2**22, 2**24]
    results = []

    for n in sizes:
        x = torch.randn(n, device="cuda", dtype=torch.float32)

        # Triton simple copy
        r = benchmark_kernel(
            fn=lambda t: copy_kernel(t),
            args=(x,),
            name=f"triton_copy_{n}",
            config=CONFIG,
        )
        results.append(r)

        # Triton vectorized copy
        r = benchmark_kernel(
            fn=lambda t: copy_vectorized(t, vec_size=4),
            args=(x,),
            name=f"triton_vec4_{n}",
            config=CONFIG,
        )
        results.append(r)

        # torch.clone
        r = benchmark_kernel(
            fn=lambda t: t.clone(),
            args=(x,),
            name=f"torch_clone_{n}",
            config=CONFIG,
        )
        results.append(r)

    compare_kernels(results)
    return results


def bench_strided_copy_degradation() -> list[BenchmarkResult]:
    """Show bandwidth degradation with increasing stride."""
    print(f"\n{'=' * 60}")
    print("  STRIDED COPY BANDWIDTH DEGRADATION")
    print(f"{'=' * 60}")

    n_base = 2**20
    strides = [1, 2, 4, 8, 16, 32, 64, 128]
    results = []

    for stride in strides:
        total_n = n_base * stride
        x = torch.randn(total_n, device="cuda", dtype=torch.float32)
        n_copy = total_n // stride

        def strided_copy():
            return copy_non_contiguous(x, stride=stride)

        r = benchmark_kernel(
            fn=strided_copy,
            name=f"stride_{stride}",
            config=CONFIG,
        )
        results.append(r)

    _, peak_bw = get_peak_bandwidth()
    print(f"\n  {'Stride':>8}  {'Bandwidth (GB/s)':>18}  {'% Peak':>8}")
    print(f"  {'-' * 8}  {'-' * 18}  {'-' * 8}")
    for r in results:
        stride = int(r.name.split("_")[-1])
        pct = r.bandwidth_gb_s / peak_bw * 100 if peak_bw > 0 else 0
        print(f"  {stride:>8}  {r.bandwidth_gb_s:>18.2f}  {pct:>7.1f}%")

    return results


def bench_size_sweep() -> list[BenchmarkResult]:
    """Sweep tensor sizes and record bandwidth."""
    print(f"\n{'=' * 60}")
    print("  BANDWIDTH vs SIZE SWEEP")
    print(f"{'=' * 60}")

    # From 1KB to 1GB in float32 elements
    sizes = [
        256,  # 1 KB
        1024,  # 4 KB
        4096,  # 16 KB
        16384,  # 64 KB
        65536,  # 256 KB
        262144,  # 1 MB
        1_048_576,  # 4 MB
        4_194_304,  # 16 MB
        16_777_216,  # 64 MB
        67_108_864,  # 256 MB
    ]
    results = []

    for n in sizes:
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        y = torch.randn(n, device="cuda", dtype=torch.float32)

        def elem_mult():
            return x * y

        r = benchmark_kernel(
            fn=elem_mult,
            name=f"elem_mult_{n}",
            config=BenchmarkConfig(warmup_steps=3, measure_steps=20, repeat=2),
        )
        results.append(r)

    _, peak_bw = get_peak_bandwidth()
    print(f"\n  {'Elements':>12}  {'Size':>10}  {'Bandwidth (GB/s)':>18}  {'% Peak':>8}")
    print(f"  {'-' * 12}  {'-' * 10}  {'-' * 18}  {'-' * 8}")
    for r in results:
        n = int(r.name.split("_")[-1])
        size_bytes = n * 4
        if size_bytes >= 1e9:
            size_str = f"{size_bytes / 1e9:.1f} GB"
        elif size_bytes >= 1e6:
            size_str = f"{size_bytes / 1e6:.1f} MB"
        elif size_bytes >= 1e3:
            size_str = f"{size_bytes / 1e3:.1f} KB"
        else:
            size_str = f"{size_bytes} B"
        pct = r.bandwidth_gb_s / peak_bw * 100 if peak_bw > 0 else 0
        print(f"  {n:>12,}  {size_str:>10}  {r.bandwidth_gb_s:>18.2f}  {pct:>7.1f}%")

    return results


def generate_bandwidth_csv(results: list[BenchmarkResult], output_path: str = ".") -> Path:
    """Generate bandwidth vs size CSV data for charting."""
    filepath = Path(output_path) / "bandwidth_vs_size.csv"
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n_elements", "size_bytes", "bandwidth_gb_s"])
        writer.writeheader()
        for r in results:
            n = int(r.name.split("_")[-1])
            writer.writerow(
                {
                    "n_elements": n,
                    "size_bytes": n * 4,
                    "bandwidth_gb_s": r.bandwidth_gb_s,
                }
            )
    return filepath


# ---------------------------------------------------------------------------
# CUDA C++ 原生带宽 benchmark
# ---------------------------------------------------------------------------


def bench_cuda_vectorized_access() -> list[BenchmarkResult]:
    """对比 CUDA C++ float / float2 / float4 向量化访问的带宽差异。"""
    if not _BANDWIDTH_KERNELS_AVAILABLE:
        print("\n  (cuda_bandwidth_kernels 扩展未构建，跳过)")
        return []

    print(f"\n{'=' * 60}")
    print("  CUDA C++ VECTORIZED ACCESS: float vs float2 vs float4")
    print(f"{'=' * 60}")

    sizes = [2**18, 2**20, 2**22, 2**24]
    results = []

    for n in sizes:
        x = torch.randn(n, device="cuda", dtype=torch.float32)

        # float 标量
        r = benchmark_kernel(
            fn=lambda t: cuda_bandwidth_kernels.bench_copy_float(t),
            args=(x,),
            name=f"cuda_copy_float_{n}",
            config=BenchmarkConfig(warmup_steps=3, measure_steps=15, repeat=2),
        )
        results.append(r)

        # float2
        r = benchmark_kernel(
            fn=lambda t: cuda_bandwidth_kernels.bench_copy_float2(t),
            args=(x,),
            name=f"cuda_copy_float2_{n}",
            config=BenchmarkConfig(warmup_steps=3, measure_steps=15, repeat=2),
        )
        results.append(r)

        # float4
        r = benchmark_kernel(
            fn=lambda t: cuda_bandwidth_kernels.bench_copy_float4(t),
            args=(x,),
            name=f"cuda_copy_float4_{n}",
            config=BenchmarkConfig(warmup_steps=3, measure_steps=15, repeat=2),
        )
        results.append(r)

    compare_kernels(results)
    return results


def bench_cuda_strided_degradation() -> list[BenchmarkResult]:
    """对比 CUDA C++ strided 访问的带宽衰减。"""
    if not _BANDWIDTH_KERNELS_AVAILABLE:
        return []

    print(f"\n{'=' * 60}")
    print("  CUDA C++ STRIDED ACCESS DEGRADATION")
    print(f"{'=' * 60}")

    n_base = 2**20
    strides = [1, 2, 4, 8, 16, 32, 64, 128]
    results = []

    for stride in strides:
        total_n = n_base * stride
        x = torch.randn(total_n, device="cuda", dtype=torch.float32)

        r = benchmark_kernel(
            fn=lambda t, s=stride: cuda_bandwidth_kernels.bench_strided_copy(t, s),
            args=(x,),
            name=f"cuda_stride_{stride}",
            config=BenchmarkConfig(warmup_steps=3, measure_steps=15, repeat=2),
        )
        results.append(r)

    _, peak_bw = get_peak_bandwidth()
    print(f"\n  {'Stride':>8}  {'Time (us)':>12}  {'Effective BW (GB/s)':>22}")
    print(f"  {'-' * 8}  {'-' * 12}  {'-' * 22}")
    for r in results:
        stride = int(r.name.split("_")[-1])
        n = n_base
        bytes_moved = n * 4 * 2  # read + write
        time_us = r.p50_ms * 1000.0  # ms -> us
        eff_bw = (bytes_moved / (time_us / 1e6)) / 1e9 if time_us > 0 else 0
        print(f"  {stride:>8}  {time_us:>10.1f} us  {eff_bw:>20.1f} GB/s")

    return results


def main():
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        sys.exit(1)

    device_name, peak_bw = get_peak_bandwidth()
    print("=" * 70)
    print("  MEMORY BANDWIDTH BENCHMARKS")
    print("=" * 70)
    print(f"\n  Device: {device_name}")
    print(f"  Estimated Peak Bandwidth: {peak_bw:.1f} GB/s")
    print(f"  PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print(f"  Triton: {triton.__version__}")
    if _BANDWIDTH_KERNELS_AVAILABLE:
        print("  CUDA C++ bandwidth kernels: available")
    else:
        print("  CUDA C++ bandwidth kernels: not built")

    all_results = []

    all_results.extend(bench_size_sweep())
    all_results.extend(bench_contiguous_vs_strided())
    all_results.extend(bench_triton_copy_vs_torch())
    all_results.extend(bench_strided_copy_degradation())
    all_results.extend(bench_cuda_vectorized_access())
    all_results.extend(bench_cuda_strided_degradation())

    print(f"\n{'=' * 70}")
    print("  FINAL SUMMARY")
    print(f"{'=' * 70}")
    compare_kernels(all_results)

    report_md = generate_report(all_results, "03_memory_bandwidth/bandwidth_report")
    print(f"\n{report_md}")


if __name__ == "__main__":
    main()
