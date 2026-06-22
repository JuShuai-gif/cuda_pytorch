#!/usr/bin/env python3
"""
Memory management benchmark.

Measures:
  - In-place vs out-of-place peak memory and performance
  - Buffer pool allocation overhead vs fresh allocation
  - Memory transfer overhead
  - Memory-performance tradeoffs

Run: python 08_memory_management/benchmark_memory_management.py
"""

from __future__ import annotations

import sys
import time

import torch

from benchmarks.benchmark_utils import (
    BenchmarkConfig,
    BenchmarkResult,
    benchmark_kernel,
    compare_kernels,
    generate_report,
)
from memory_reuse import BufferPool

CONFIG = BenchmarkConfig(warmup_steps=5, measure_steps=20, repeat=3)


# ---------------------------------------------------------------------------
# In-place vs out-of-place benchmark
# ---------------------------------------------------------------------------


def bench_inplace_vs_outofplace() -> list[BenchmarkResult]:
    """Benchmark in-place vs out-of-place for common operations."""
    print(f"\n{'=' * 60}")
    print("  IN-PLACE vs OUT-OF-PLACE")
    print(f"{'=' * 60}")

    results: list[BenchmarkResult] = []
    ops = [
        "add",
        "mul",
        "sigmoid",
        "relu",
        "tanh",
        # "gelu", - PyTorch doesn't have inplace GELU
    ]
    n = 50_000_000

    for op_name in ops:
        torch.cuda.empty_cache()
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        y = torch.randn(n, device="cuda", dtype=torch.float32)

        out_func = getattr(torch, op_name)
        inplace_func = getattr(x, f"{op_name}_", None)

        # Out-of-place
        r = benchmark_kernel(
            fn=lambda a, b: (
                out_func(a, b) if op_name not in ("sigmoid", "relu", "tanh") else out_func(a)
            ),
            args=(x, y) if op_name not in ("sigmoid", "relu", "tanh") else (x,),
            name=f"{op_name}_out",
            config=CONFIG,
        )
        results.append(r)

        # In-place (if available)
        if inplace_func:
            r = benchmark_kernel(
                fn=lambda a, b: (
                    inplace_func(b)
                    if op_name not in ("sigmoid", "relu", "tanh")
                    else inplace_func()
                ),
                args=(x, y) if op_name not in ("sigmoid", "relu", "tanh") else (x,),
                name=f"{op_name}_in",
                config=CONFIG,
            )
            results.append(r)
        else:
            print(f"  No in-place version for {op_name}")

        del x, y
        torch.cuda.empty_cache()

    compare_kernels(results)
    return results


# ---------------------------------------------------------------------------
# Buffer pool overhead
# ---------------------------------------------------------------------------


def bench_buffer_pool_overhead() -> list[BenchmarkResult]:
    """Compare buffer pool allocation vs fresh allocation."""
    print(f"\n{'=' * 60}")
    print("  BUFFER POOL OVERHEAD")
    print(f"{'=' * 60}")

    results: list[BenchmarkResult] = []
    n = 10_000_000
    pool = BufferPool(max_size=n * 100)
    x = torch.randn(n, device="cuda", dtype=torch.float32)

    # Fresh allocation + compute
    def _fresh_alloc_compute():
        tmp = torch.empty(n, device="cuda", dtype=torch.float32)
        tmp.copy_(x)
        tmp.mul_(2.0)
        x.copy_(tmp)
        del tmp

    r = benchmark_kernel(fn=_fresh_alloc_compute, name="fresh_alloc_compute", config=CONFIG)
    results.append(r)

    # Buffer pool acquire + compute + release
    def _pool_compute():
        buf = pool.acquire(n)
        buf.copy_(x)
        buf.mul_(2.0)
        x.copy_(buf)
        pool.release()

    r = benchmark_kernel(fn=_pool_compute, name="pool_reuse_compute", config=CONFIG)
    results.append(r)

    compare_kernels(results)

    # Memory comparison
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(10):
        tmp = torch.empty(n, device="cuda", dtype=torch.float32)
        del tmp
    fresh_peak = torch.cuda.max_memory_allocated()
    torch.cuda.empty_cache()

    print(f"\n  Fresh alloc peak: {fresh_peak / 1e6:.1f} MB")

    del x
    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Memory transfer overhead
# ---------------------------------------------------------------------------


def bench_memory_transfer_overhead() -> list[BenchmarkResult]:
    """Measure overhead of create-copy-delete patterns."""
    print(f"\n{'=' * 60}")
    print("  MEMORY TRANSFER OVERHEAD")
    print(f"{'=' * 60}")

    results: list[BenchmarkResult] = []
    sizes = [1_000_000, 10_000_000, 50_000_000]

    for n in sizes:
        mb = (n * 4) / 1e6

        # H2D transfer
        host = torch.randn(n, dtype=torch.float32, pin_memory=True)
        dev = torch.empty(n, device="cuda", dtype=torch.float32)

        r = benchmark_kernel(
            fn=lambda h, d: d.copy_(h),
            args=(host, dev),
            name=f"h2d_{n // 1_000_000}M",
            config=CONFIG,
        )
        results.append(r)

        # D2H transfer
        r = benchmark_kernel(
            fn=lambda h, d: h.copy_(d),
            args=(host, dev),
            name=f"d2h_{n // 1_000_000}M",
            config=CONFIG,
        )
        results.append(r)

        # Clear
        del host, dev
        torch.cuda.empty_cache()

    compare_kernels(results)
    return results


# ---------------------------------------------------------------------------
# Memory-performance tradeoff demonstration
# ---------------------------------------------------------------------------


def bench_memory_performance_tradeoff() -> list[BenchmarkResult]:
    """Show memory-performance tradeoff for different strategies.

    Strategy A: Pre-allocate everything (high memory, low latency)
    Strategy B: Allocate on-demand + caching allocator
    Strategy C: Minimal memory, recompute when needed
    """
    print(f"\n{'=' * 60}")
    print("  MEMORY-PERFORMANCE TRADEOFF")
    print(f"{'=' * 60}")

    results: list[BenchmarkResult] = []
    n = 5_000_000
    num_iters = 50

    # Strategy A: Pre-allocate
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    prefilled = [torch.randn(n, device="cuda", dtype=torch.float32) for _ in range(num_iters)]
    torch.cuda.synchronize()
    peak_a = torch.cuda.max_memory_allocated()

    def _strategy_a():
        for t in prefilled:
            t.add_(1.0)

    r = benchmark_kernel(fn=_strategy_a, name="strategy_prealloc", config=CONFIG)
    results.append(r)
    del prefilled
    torch.cuda.empty_cache()

    # Strategy B: Allocate on demand with caching allocator
    torch.cuda.reset_peak_memory_stats()
    base = torch.randn(n, device="cuda", dtype=torch.float32)

    def _strategy_b():
        tmp = torch.empty(n, device="cuda", dtype=torch.float32)
        tmp.copy_(base)
        tmp.add_(1.0)
        base.copy_(tmp)
        del tmp

    # Warmup to prime caching allocator
    for _ in range(10):
        _strategy_b()

    torch.cuda.reset_peak_memory_stats()
    r = benchmark_kernel(fn=_strategy_b, name="strategy_on_demand", config=CONFIG)
    results.append(r)
    peak_b = torch.cuda.max_memory_allocated()
    del base
    torch.cuda.empty_cache()

    # Strategy C: Minimal memory (recompute)
    torch.cuda.reset_peak_memory_stats()
    seed = torch.randn(1, device="cuda", dtype=torch.float32)

    def _strategy_c():
        tmp = seed * torch.randn(1, device="cuda", dtype=torch.float32)
        del tmp

    r = benchmark_kernel(fn=_strategy_c, name="strategy_minimal", config=CONFIG)
    results.append(r)
    peak_c = torch.cuda.max_memory_allocated()
    del seed
    torch.cuda.empty_cache()

    compare_kernels(results)

    print(f"\n  Peak memory:")
    print(f"    Strategy A (pre-alloc): {peak_a / 1e6:.1f} MB")
    print(f"    Strategy B (on-demand): {peak_b / 1e6:.1f} MB")
    print(f"    Strategy C (minimal):   {peak_c / 1e6:.1f} MB")

    return results


def main():
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        sys.exit(1)

    print("=" * 70)
    print("  MEMORY MANAGEMENT BENCHMARKS")
    print("=" * 70)
    print(f"\n  Device: {torch.cuda.get_device_name(0)}")
    print(f"  PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print(f"  Total GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.2f} GB")

    all_results: list[BenchmarkResult] = []
    all_results.extend(bench_inplace_vs_outofplace())
    all_results.extend(bench_buffer_pool_overhead())
    all_results.extend(bench_memory_transfer_overhead())
    all_results.extend(bench_memory_performance_tradeoff())

    print(f"\n{'=' * 70}")
    print("  FINAL SUMMARY")
    print(f"{'=' * 70}")
    compare_kernels(all_results)

    report_md = generate_report(all_results, "08_memory_management/memory_report")
    print(f"\n{report_md}")


if __name__ == "__main__":
    main()
