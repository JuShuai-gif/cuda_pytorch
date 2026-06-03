#!/usr/bin/env python3
"""
Comprehensive operator fusion benchmark.

Compares each fused kernel vs:
  - PyTorch sequential (eager)
  - torch.compile(sequential)
  - Shows memory bandwidth savings
  - Speedup at various sizes

Run: python 04_operator_fusion/benchmark_operator_fusion.py
"""

from __future__ import annotations

import sys

import torch
import triton

from benchmarks.benchmark_utils import (
    BenchmarkConfig,
    BenchmarkResult,
    benchmark_kernel,
    compare_kernels,
    generate_report,
)
from kernel_add_relu import fused_add_relu, sequential_add_relu
from kernel_bias_gelu import fused_bias_gelu, sequential_bias_gelu
from kernel_residual_layernorm import fused_residual_layernorm, sequential_residual_layernorm
from kernel_rmsnorm import triton_rmsnorm, torch_rmsnorm

CONFIG = BenchmarkConfig(warmup_steps=5, measure_steps=30, repeat=3)


def bench_add_relu() -> list[BenchmarkResult]:
    """Benchmark fused add+relu vs sequential vs torch.compile."""
    print(f"\n{'=' * 60}")
    print("  ADD + RELU FUSION BENCHMARK")
    print(f"{'=' * 60}")

    dims = [1024, 4096, 8192, 32768, 131072, 524288, 4_194_304, 16_777_216]
    results = []

    for dim in dims:
        x = torch.randn(dim, device="cuda", dtype=torch.float32)
        bias = torch.randn(dim, device="cuda", dtype=torch.float32)

        # Fused
        r = benchmark_kernel(
            fn=lambda a, b: fused_add_relu(a, b),
            args=(x, bias),
            name=f"fused_add_relu_{dim}",
            config=CONFIG,
        )
        results.append(r)

        # Sequential
        r = benchmark_kernel(
            fn=lambda a, b: sequential_add_relu(a, b),
            args=(x, bias),
            name=f"seq_add_relu_{dim}",
            config=CONFIG,
        )
        results.append(r)

    # Summary
    print(f"\n  {'Size':>12}  {'Fused (GB/s)':>14}  {'Seq (GB/s)':>14}  {'Speedup':>8}")
    print(f"  {'-' * 12}  {'-' * 14}  {'-' * 14}  {'-' * 8}")
    for i in range(0, len(results), 2):
        fused = results[i]
        seq = results[i + 1]
        n = int(fused.name.split("_")[-1])
        speedup = seq.bandwidth_gb_s / fused.bandwidth_gb_s if fused.bandwidth_gb_s > 0 else 0
        print(
            f"  {n:>12,}  {fused.bandwidth_gb_s:>12.2f} GB/s"
            f"  {seq.bandwidth_gb_s:>12.2f} GB/s  {speedup:>6.2f}x"
        )

    return results


def bench_bias_gelu() -> list[BenchmarkResult]:
    """Benchmark fused bias+gelu vs sequential vs torch.compile."""
    print(f"\n{'=' * 60}")
    print("  BIAS + GELU FUSION BENCHMARK")
    print(f"{'=' * 60}")

    dims = [1024, 4096, 16384, 65536, 262144, 1_048_576, 4_194_304]
    results = []

    for dim in dims:
        x = torch.randn(dim, device="cuda", dtype=torch.float32)
        bias = torch.randn(dim, device="cuda", dtype=torch.float32)

        r = benchmark_kernel(
            fn=lambda a, b: fused_bias_gelu(a, b),
            args=(x, bias),
            name=f"fused_gelu_{dim}",
            config=CONFIG,
        )
        results.append(r)

        r = benchmark_kernel(
            fn=lambda a, b: sequential_bias_gelu(a, b),
            args=(x, bias),
            name=f"seq_gelu_{dim}",
            config=CONFIG,
        )
        results.append(r)

    print(f"\n  {'Size':>12}  {'Fused (GB/s)':>14}  {'Seq (GB/s)':>14}  {'Speedup':>8}")
    print(f"  {'-' * 12}  {'-' * 14}  {'-' * 14}  {'-' * 8}")
    for i in range(0, len(results), 2):
        fused = results[i]
        seq = results[i + 1]
        n = int(fused.name.split("_")[-1])
        speedup = seq.bandwidth_gb_s / fused.bandwidth_gb_s if fused.bandwidth_gb_s > 0 else 0
        print(
            f"  {n:>12,}  {fused.bandwidth_gb_s:>12.2f} GB/s"
            f"  {seq.bandwidth_gb_s:>12.2f} GB/s  {speedup:>6.2f}x"
        )

    return results


def bench_residual_layernorm() -> list[BenchmarkResult]:
    """Benchmark fused residual+layernorm vs sequential."""
    print(f"\n{'=' * 60}")
    print("  RESIDUAL + LAYERNORM FUSION BENCHMARK")
    print(f"{'=' * 60}")

    shapes = [
        (1, 1024),
        (1, 2048),
        (1, 4096),
        (4, 1024),
        (4, 2048),
        (4, 4096),
        (16, 1024),
        (16, 2048),
        (32, 1024),
        (32, 2048),
    ]
    results = []

    for B, D in shapes:
        x = torch.randn(B, D, device="cuda", dtype=torch.float32)
        residual = torch.randn(B, D, device="cuda", dtype=torch.float32)

        r = benchmark_kernel(
            fn=lambda a, b: fused_residual_layernorm(a, b, block_size=next_pow2(D)),
            args=(x, residual),
            name=f"fused_rln_{B}x{D}",
            config=CONFIG,
        )
        results.append(r)

        r = benchmark_kernel(
            fn=lambda a, b: sequential_residual_layernorm(a, b),
            args=(x, residual),
            name=f"seq_rln_{B}x{D}",
            config=CONFIG,
        )
        results.append(r)

    print(f"\n  {'Shape':>12}  {'Fused (ms)':>12}  {'Seq (ms)':>12}  {'Speedup':>8}")
    print(f"  {'-' * 12}  {'-' * 12}  {'-' * 12}  {'-' * 8}")
    for i in range(0, len(results), 2):
        fused = results[i]
        seq = results[i + 1]
        speedup = seq.p50_ms / fused.p50_ms if fused.p50_ms > 0 else 0
        print(
            f"  {f'{B}x{D}':>12}  {fused.p50_ms:>10.4f} ms"
            f"  {seq.p50_ms:>10.4f} ms  {speedup:>6.2f}x"
        )

    return results


def bench_rmsnorm() -> list[BenchmarkResult]:
    """Benchmark Triton RMSNorm vs PyTorch RMSNorm."""
    print(f"\n{'=' * 60}")
    print("  RMSNORM BENCHMARK")
    print(f"{'=' * 60}")

    shapes = [
        (1, 4096),
        (1, 8192),
        (4, 4096),
        (4, 8192),
        (16, 4096),
        (32, 4096),
        (128, 4096),
    ]
    results = []

    for B, D in shapes:
        x = torch.randn(B, D, device="cuda", dtype=torch.float32)
        weight = torch.randn(D, device="cuda", dtype=torch.float32)

        r = benchmark_kernel(
            fn=lambda a, w: triton_rmsnorm(a, w, block_size=next_pow2(D)),
            args=(x, weight),
            name=f"triton_rmsnorm_{B}x{D}",
            config=CONFIG,
        )
        results.append(r)

        r = benchmark_kernel(
            fn=lambda a, w: torch_rmsnorm(a, w),
            args=(x, weight),
            name=f"torch_rmsnorm_{B}x{D}",
            config=CONFIG,
        )
        results.append(r)

    print(f"\n  {'Shape':>12}  {'Triton (ms)':>12}  {'Torch (ms)':>12}  {'Ratio':>8}")
    print(f"  {'-' * 12}  {'-' * 12}  {'-' * 12}  {'-' * 8}")
    for i in range(0, len(results), 2):
        tms = results[i]
        pms = results[i + 1]
        ratio = pms.p50_ms / tms.p50_ms if tms.p50_ms > 0 else 0
        print(f"  {f'{B}x{D}':>12}  {tms.p50_ms:>10.4f} ms  {pms.p50_ms:>10.4f} ms  {ratio:>6.2f}x")

    return results


def bench_torch_compile_comparison() -> list[BenchmarkResult]:
    """Compare fused kernels with torch.compile."""
    print(f"\n{'=' * 60}")
    print("  TORCH.COMPILE COMPARISON")
    print(f"{'=' * 60}")

    n = 4_194_304
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    bias = torch.randn(n, device="cuda", dtype=torch.float32)
    results = []

    # Fused
    r = benchmark_kernel(
        fn=lambda a, b: fused_add_relu(a, b),
        args=(x, bias),
        name="fused_add_relu",
        config=CONFIG,
    )
    results.append(r)

    # Sequential
    r = benchmark_kernel(
        fn=lambda a, b: sequential_add_relu(a, b),
        args=(x, bias),
        name="sequential_add_relu",
        config=CONFIG,
    )
    results.append(r)

    # torch.compile sequential
    try:
        compiled_fn = torch.compile(sequential_add_relu, fullgraph=True)

        for _ in range(20):
            compiled_fn(x, bias)
        torch.cuda.synchronize()

        r = benchmark_kernel(
            fn=compiled_fn,
            args=(x, bias),
            name="compiled_add_relu",
            config=CONFIG,
        )
        results.append(r)
    except Exception as e:
        print(f"  Note: torch.compile benchmarking skipped ({e})")

    compare_kernels(results)
    return results


def next_pow2(n: int) -> int:
    """Return the next power of 2 >= n."""
    p = 1
    while p < n:
        p *= 2
    return p


def main():
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        sys.exit(1)

    print("=" * 70)
    print("  OPERATOR FUSION BENCHMARKS")
    print("=" * 70)
    print(f"\n  Device: {torch.cuda.get_device_name(0)}")
    print(f"  PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print(f"  Triton: {triton.__version__}")

    all_results = []
    all_results.extend(bench_add_relu())
    all_results.extend(bench_bias_gelu())
    all_results.extend(bench_residual_layernorm())
    all_results.extend(bench_rmsnorm())
    all_results.extend(bench_torch_compile_comparison())

    print(f"\n{'=' * 70}")
    print("  FINAL SUMMARY")
    print(f"{'=' * 70}")
    compare_kernels(all_results)

    report_md = generate_report(all_results, "04_operator_fusion/fusion_report")
    print(f"\n{report_md}")


if __name__ == "__main__":
    main()
