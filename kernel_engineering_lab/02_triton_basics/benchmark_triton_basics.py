#!/usr/bin/env python3
"""
Benchmark Triton kernels against PyTorch equivalents.

Compares:
  - Triton vector_add vs torch.add
  - Triton SiLU/GELU/ReLU vs torch.nn.functional equivalents
  - Triton basic GEMM vs torch.matmul

Run: python 02_triton_basics/benchmark_triton_basics.py
"""

from __future__ import annotations

import sys

import torch
import triton

from benchmarks.benchmark_utils import (
    BenchmarkConfig,
    benchmark_kernel,
    benchmark_torch,
    compare_kernels,
)

from triton_vector_add import triton_vector_add
from triton_elementwise import triton_gelu, triton_relu, triton_silu
from triton_gemm_basic import triton_gemm


CONFIG = BenchmarkConfig(warmup_steps=5, measure_steps=50, repeat=3)


# ---------------------------------------------------------------------------
# Vector Add benchmark
# ---------------------------------------------------------------------------


def bench_vector_add():
    print(f"\n{'=' * 60}")
    print("  VECTOR ADD")
    print(f"{'=' * 60}")

    sizes = [2**15, 2**20, 2**24]  # 32K, 1M, 16M
    all_results = []

    for n in sizes:
        print(f"\n  --- size = {n:,} ---")
        a = torch.randn(n, device="cuda", dtype=torch.float32)
        b = torch.randn(n, device="cuda", dtype=torch.float32)

        results = []

        results.append(
            benchmark_kernel(
                fn=lambda x, y: triton_vector_add(x, y),
                args=(a, b),
                name=f"triton_add_{n}",
                config=CONFIG,
            )
        )
        results.append(
            benchmark_torch(
                torch.add,
                a,
                b,
                name=f"torch_add_{n}",
                config=CONFIG,
            )
        )
        all_results.extend(results)
        compare_kernels(results)

    return all_results


# ---------------------------------------------------------------------------
# Activation benchmark
# ---------------------------------------------------------------------------


def bench_activations():
    print(f"\n{'=' * 60}")
    print("  ACTIVATION FUNCTIONS")
    print(f"{'=' * 60}")

    sizes = [2**15, 2**20]  # 32K, 1M
    all_results = []

    activations = [
        ("SiLU", triton_silu, torch.nn.functional.silu),
        ("GELU", triton_gelu, lambda t: torch.nn.functional.gelu(t, approximate="tanh")),
        ("ReLU", triton_relu, torch.nn.functional.relu),
    ]

    for name, triton_fn, torch_fn in activations:
        for n in sizes:
            x = torch.randn(n, device="cuda", dtype=torch.float32)

            results = []
            results.append(
                benchmark_kernel(
                    fn=triton_fn,
                    args=(x,),
                    name=f"triton_{name.lower()}_{n}",
                    config=CONFIG,
                )
            )
            results.append(
                benchmark_torch(
                    torch_fn,
                    x,
                    name=f"torch_{name.lower()}_{n}",
                    config=CONFIG,
                )
            )
            all_results.extend(results)

    compare_kernels(all_results)
    return all_results


# ---------------------------------------------------------------------------
# GEMM benchmark
# ---------------------------------------------------------------------------


def bench_gemm():
    print(f"\n{'=' * 60}")
    print("  MATRIX MULTIPLY (GEMM)")
    print(f"{'=' * 60}")

    # Keep sizes modest since this is a naive tiled implementation
    gemm_sizes = [
        (256, 256, 256),
        (512, 512, 256),
        (1024, 1024, 256),
    ]
    all_results = []

    for M, N, K in gemm_sizes:
        print(f"\n  --- {M}x{K}x{N} ---")
        a = torch.randn(M, K, device="cuda", dtype=torch.float32)
        b = torch.randn(K, N, device="cuda", dtype=torch.float32)

        results = []

        results.append(
            benchmark_kernel(
                fn=triton_gemm,
                args=(a, b),
                name=f"triton_gemm_{M}x{K}x{N}",
                config=CONFIG,
            )
        )
        results.append(
            benchmark_torch(
                torch.matmul,
                a,
                b,
                name=f"torch_matmul_{M}x{K}x{N}",
                config=CONFIG,
            )
        )
        all_results.extend(results)
        compare_kernels(results)

    return all_results


def main():
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        sys.exit(1)

    print("GPU:", torch.cuda.get_device_name(0))
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print(f"Triton: {triton.__version__}")

    all_results = []
    all_results.extend(bench_vector_add())
    all_results.extend(bench_activations())
    all_results.extend(bench_gemm())

    print(f"\n{'=' * 60}")
    print("  SUMMARY (All Kernels)")
    print(f"{'=' * 60}")
    compare_kernels(all_results)


if __name__ == "__main__":
    main()
