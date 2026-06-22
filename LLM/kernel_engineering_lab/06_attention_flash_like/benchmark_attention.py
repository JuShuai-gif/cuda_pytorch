#!/usr/bin/env python3
"""
Comprehensive attention benchmark.

Compares naive PyTorch, naive Triton, tiled Triton, prefill, decode,
and torch.nn.functional.scaled_dot_product_attention across typical
LLM configurations. Measures latency, peak memory, and estimated bandwidth.

Run: python 06_attention_flash_like/benchmark_attention.py
"""

from __future__ import annotations

import math
import sys
from typing import Optional

import torch
import triton

from benchmarks.benchmark_utils import (
    BenchmarkConfig,
    BenchmarkResult,
    benchmark_kernel,
    compare_kernels,
    generate_report,
)
from flash_attention_kv_cache import attention_decode, attention_prefill
from naive_attention import naive_attention_torch, naive_attention_triton
from tiled_attention import tiled_attention

CONFIG = BenchmarkConfig(warmup_steps=5, measure_steps=20, repeat=3)


def _get_torch_sdpa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False
) -> torch.Tensor:
    """Wrapper for torch.nn.functional.scaled_dot_product_attention."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    is_causal = causal and q.shape[2] == k.shape[2]
    attn_mask = None
    if causal and not is_causal:
        Q_len, KV_len = q.shape[2], k.shape[2]
        diag = KV_len - Q_len + 1 if KV_len > Q_len else 1
        m = torch.triu(torch.ones(Q_len, KV_len, device=q.device, dtype=torch.bool), diagonal=diag)
        attn_mask = m

    if attn_mask is not None:
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, scale=scale, dropout_p=0.0
        )
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=is_causal, scale=scale, dropout_p=0.0
    )


def _peak_memory_delta(fn, *args, **kwargs) -> float:
    """Measure peak memory increase during fn execution."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    result = fn(*args, **kwargs)
    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated()
    return peak_bytes / 1e6  # MB


def _build_bench(
    name: str,
    fn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    **extra,
) -> BenchmarkResult:
    """Run a benchmark with the given attention function."""
    return benchmark_kernel(
        fn=lambda qq, kk, vv: fn(qq, kk, vv, **extra),
        args=(q, k, v),
        name=name,
        config=CONFIG,
    )


def bench_prefill_configs() -> list[BenchmarkResult]:
    """Benchmark prefill (Q_len == KV_len) at various sequence lengths."""
    print(f"\n{'=' * 60}")
    print("  PREFILL BENCHMARK (Q_len == KV_len)")
    print(f"{'=' * 60}")

    B, H = 1, 32
    head_dim = 128
    seq_lens = [128, 512, 1024, 2048, 4096]
    results = []

    for L in seq_lens:
        q = torch.randn(B, H, L, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, L, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, L, head_dim, device="cuda", dtype=torch.float16)

        try:
            r = _build_bench(f"torch_sdpa_L{L}", _get_torch_sdpa, q, k, v)
            results.append(r)
        except RuntimeError as e:
            print(f"  torch_sdpa_L{L}: OOM or error: {e}")

        try:
            r = _build_bench(f"tiled_L{L}", tiled_attention, q, k, v)
            results.append(r)
        except RuntimeError as e:
            print(f"  tiled_L{L}: OOM or error: {e}")

        try:
            r = _build_bench(f"prefill_L{L}", attention_prefill, q, k, v)
            results.append(r)
        except RuntimeError as e:
            print(f"  prefill_L{L}: OOM or error: {e}")

    compare_kernels(results)
    return results


def bench_decode_configs() -> list[BenchmarkResult]:
    """Benchmark decode (Q_len=1, varying KV cache size)."""
    print(f"\n{'=' * 60}")
    print("  DECODE BENCHMARK (Q_len=1)")
    print(f"{'=' * 60}")

    B, H = 1, 32
    head_dim = 128
    kv_lens = [128, 512, 1024, 2048, 4096]
    results = []

    for KV_len in kv_lens:
        q = torch.randn(B, H, 1, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, KV_len, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, KV_len, head_dim, device="cuda", dtype=torch.float16)

        # torch SDP
        try:
            r = _build_bench(f"torch_sdpa_decode_KV{KV_len}", _get_torch_sdpa, q, k, v)
            results.append(r)
        except RuntimeError as e:
            print(f"  torch_sdpa_decode_KV{KV_len}: OOM or error: {e}")

        # decode kernel
        try:
            r = benchmark_kernel(
                fn=lambda qq, kk, vv: attention_decode(qq, kk, vv),
                args=(q, k, v),
                name=f"decode_KV{KV_len}",
                config=CONFIG,
            )
            results.append(r)
        except RuntimeError as e:
            print(f"  decode_KV{KV_len}: OOM or error: {e}")

        # tiled attention (same kernel, works with Q_len=1)
        try:
            r = _build_bench(f"tiled_decode_KV{KV_len}", tiled_attention, q, k, v)
            results.append(r)
        except RuntimeError as e:
            print(f"  tiled_decode_KV{KV_len}: OOM or error: {e}")

    compare_kernels(results)
    return results


def bench_llama_configs() -> list[BenchmarkResult]:
    """Benchmark with LLaMA-scale configurations."""
    print(f"\n{'=' * 60}")
    print("  LLAMA-SCALE BENCHMARKS")
    print(f"{'=' * 60}")

    results = []

    # LLaMA-7B: hidden=4096, num_heads=32, head_dim=128
    # LLaMA-70B: hidden=8192, num_heads=64, head_dim=128
    configs = [
        ("llama7b", 1, 32, 128, 2048),
        ("llama70b", 1, 64, 128, 2048),
    ]

    for name, B, H, D, L in configs:
        q = torch.randn(B, H, L, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, L, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, L, D, device="cuda", dtype=torch.float16)

        for impl_name, impl in [
            ("torch_sdpa", _get_torch_sdpa),
            ("tiled", tiled_attention),
            ("prefill", attention_prefill),
        ]:
            try:
                r = _build_bench(f"{impl_name}_{name}_L{L}", impl, q, k, v)
                results.append(r)
            except RuntimeError as e:
                print(f"  {impl_name}_{name}_L{L}: OOM or error: {e}")

    compare_kernels(results)
    return results


def bench_memory_scaling() -> list[BenchmarkResult]:
    """Measure peak memory usage at different sequence lengths."""
    print(f"\n{'=' * 60}")
    print("  MEMORY SCALING ANALYSIS")
    print(f"{'=' * 60}")

    B, H = 1, 32
    head_dim = 128
    seq_lens = [256, 512, 1024, 2048, 4096]
    results = []

    print(
        f"\n  {'Seq Len':>10}  {'Naive Torch (MB)':>18}  {'Tiled (MB)':>14}  {'Torch SDP (MB)':>16}"
    )
    print(f"  {'-' * 10}  {'-' * 18}  {'-' * 14}  {'-' * 16}")

    for L in seq_lens:
        q = torch.randn(B, H, L, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, L, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, L, head_dim, device="cuda", dtype=torch.float16)

        try:
            mem_naive = _peak_memory_delta(naive_attention_torch, q, k, v)
        except RuntimeError:
            mem_naive = float("inf")

        try:
            mem_tiled = _peak_memory_delta(tiled_attention, q, k, v)
        except RuntimeError:
            mem_tiled = float("inf")

        try:
            mem_sdpa = _peak_memory_delta(_get_torch_sdpa, q, k, v)
        except RuntimeError:
            mem_sdpa = float("inf")

        print(f"  {L:>10}  {mem_naive:>16.1f} MB  {mem_tiled:>12.1f} MB  {mem_sdpa:>14.1f} MB")

    return results


def bench_dtype_comparison() -> list[BenchmarkResult]:
    """Compare performance across dtypes."""
    print(f"\n{'=' * 60}")
    print("  DTYPE COMPARISON")
    print(f"{'=' * 60}")

    B, H, L, D = 1, 16, 512, 64
    dtypes = [torch.float32, torch.float16]
    if torch.cuda.get_device_capability()[0] >= 8:
        dtypes.append(torch.bfloat16)

    results = []

    for dtype in dtypes:
        q = torch.randn(B, H, L, D, device="cuda", dtype=dtype)
        k = torch.randn(B, H, L, D, device="cuda", dtype=dtype)
        v = torch.randn(B, H, L, D, device="cuda", dtype=dtype)

        for impl_name, impl in [
            ("torch_sdpa", _get_torch_sdpa),
            ("tiled", tiled_attention),
        ]:
            try:
                r = _build_bench(f"{impl_name}_{str(dtype)[6:]}_L{L}", impl, q, k, v)
                results.append(r)
            except RuntimeError as e:
                print(f"  {impl_name}_{str(dtype)[6:]}: error: {e}")

    compare_kernels(results)
    return results


def bench_causal_comparison() -> list[BenchmarkResult]:
    """Compare causal vs non-causal attention."""
    print(f"\n{'=' * 60}")
    print("  CAUSAL vs NON-CAUSAL COMPARISON")
    print(f"{'=' * 60}")

    B, H, L, D = 1, 16, 512, 64
    results = []

    q = torch.randn(B, H, L, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, L, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, L, D, device="cuda", dtype=torch.float16)

    for causal in (False, True):
        label = "causal" if causal else "noncausal"
        for impl_name, impl in [
            ("torch_sdpa", lambda qq, kk, vv: _get_torch_sdpa(qq, kk, vv, causal=causal)),
            ("tiled", lambda qq, kk, vv: tiled_attention(qq, kk, vv, causal_mask=causal)),
            ("prefill", lambda qq, kk, vv: attention_prefill(qq, kk, vv, causal_mask=causal)),
        ]:
            try:
                r = benchmark_kernel(
                    fn=impl,
                    args=(q, k, v),
                    name=f"{impl_name}_{label}",
                    config=CONFIG,
                )
                results.append(r)
            except RuntimeError as e:
                print(f"  {impl_name}_{label}: error: {e}")

    compare_kernels(results)
    return results


def main():
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        sys.exit(1)

    print("=" * 70)
    print("  ATTENTION KERNEL BENCHMARKS")
    print("=" * 70)
    print(f"\n  Device: {torch.cuda.get_device_name(0)}")
    print(f"  PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print(f"  Triton: {triton.__version__}")
    print(f"  Compute Capability: {torch.cuda.get_device_capability(0)}")

    all_results: list[BenchmarkResult] = []
    all_results.extend(bench_prefill_configs())
    all_results.extend(bench_decode_configs())
    all_results.extend(bench_llama_configs())
    all_results.extend(bench_dtype_comparison())
    all_results.extend(bench_causal_comparison())
    bench_memory_scaling()

    print(f"\n{'=' * 70}")
    print("  FINAL SUMMARY")
    print(f"{'=' * 70}")
    compare_kernels(all_results)

    report_md = generate_report(all_results, "06_attention_flash_like/attention_report")
    print(f"\n{report_md}")


if __name__ == "__main__":
    main()
