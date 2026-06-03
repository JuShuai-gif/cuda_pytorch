#!/usr/bin/env python3
"""
Comprehensive matmul tiling benchmark.

Sweeps block sizes, num_warps, LLM-relevant shapes.
Compares naive, tiled, optimized-triton, torch.matmul, torch.compile(torch.matmul).
Tests batched matmul vs torch.bmm.
Generates GFLOPS report with % of theoretical peak.

Run: python 05_matmul_tiling/benchmark_matmul_tiling.py
"""

from __future__ import annotations

import sys
from typing import Callable

import torch
import triton

from batched_matmul import batched_matmul
from benchmarks.benchmark_utils import (
    BenchmarkConfig,
    BenchmarkResult,
    benchmark_kernel,
    compare_kernels,
    generate_report,
)
from naive_matmul import naive_matmul
from tiled_matmul import tiled_matmul
from triton_matmul_optimized import optimized_matmul

CONFIG = BenchmarkConfig(warmup_steps=5, measure_steps=30, repeat=3)


def _compute_matmul_gflops(m: int, n: int, k: int, elapsed_s: float) -> float:
    """Compute achieved GFLOPS for a matmul of shape (M, K) x (K, N).

    Each multiply-add (fma) counts as 2 FLOPs. Total: 2 * M * N * K.
    """
    total_flops = 2.0 * m * n * k
    if elapsed_s > 0:
        return total_flops / elapsed_s / 1e9
    return 0.0


def _get_peak_gflops() -> float:
    """Estimate theoretical peak GFLOPS from GPU name and CUDA compute capability.

    Returns FP32 peak. For fp16, multiply by the tensor core factor.
    """
    if not torch.cuda.is_available():
        return 0.0

    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    major, minor = cc

    sm_count = 0
    try:
        props = torch.cuda.get_device_properties(0)
        sm_count = props.multi_processor_count
    except Exception:
        sm_count = 80

    clock_mhz = 1000.0
    fma_per_cycle_per_sm = 128.0

    peak_fp32 = sm_count * clock_mhz * 1e6 * fma_per_cycle_per_sm / 1e9

    # Refine per architecture
    if major == 7 and minor == 5:  # Turing (RTX 2080 Ti)
        peak_fp32 = sm_count * 64 * 2 * 1500e6 / 1e9
    elif major == 8 and minor == 0:  # A100
        peak_fp32 = 19500.0
    elif major == 8 and minor == 6:  # RTX 3090 / A40
        peak_fp32 = sm_count * 128 * 2 * 1700e6 / 1e9
    elif major == 8 and minor == 9:  # RTX 4090, Ada Lovelace
        peak_fp32 = sm_count * 128 * 2 * 2500e6 / 1e9
    elif major == 9 and minor == 0:  # H100
        peak_fp32 = 67000.0
    elif major >= 10:  # B200 and beyond
        peak_fp32 = 90000.0
    elif major == 7 and minor == 0:  # V100
        peak_fp32 = 15700.0
    elif major == 7 and minor == 5:
        peak_fp32 = 16300.0
    elif major == 8 and minor == 6:
        peak_fp32 = 35500.0
    elif major >= 8:
        peak_fp32 = sm_count * 128 * 2 * 1500e6 / 1e9

    if "RTX 3090" in name or "RTX A6000" in name:
        peak_fp32 = 35500.0
    if "RTX 4090" in name:
        peak_fp32 = 82700.0
    if "RTX 4080" in name:
        peak_fp32 = 48700.0
    if "RTX 3080" in name:
        peak_fp32 = 29700.0
    if "T4" in name:
        peak_fp32 = 8100.0
    if "A10" in name:
        peak_fp32 = 31200.0
    if "L40" in name or "L40S" in name:
        peak_fp32 = 91400.0
    if "A5000" in name:
        peak_fp32 = 27800.0
    if "A6000" in name and "RTX" not in name:
        peak_fp32 = 38700.0
    if "H100" in name:
        if "PCIe" in name or "NVL" in name:
            peak_fp32 = 51000.0
        else:
            peak_fp32 = 67000.0
    if "B200" in name or "B100" in name:
        peak_fp32 = 90000.0

    return peak_fp32


def _wrap_matmul_kernel(
    fn: Callable, a: torch.Tensor, b: torch.Tensor, **kwargs: object
) -> Callable[[], torch.Tensor]:
    """Wrap a matmul kernel so benchmark_kernel can time it."""

    def _runner() -> torch.Tensor:
        return fn(a, b, **kwargs)

    return _runner


def bench_block_size_sweep() -> list[BenchmarkResult]:
    """Sweep BLOCK_M, BLOCK_N, BLOCK_K for the tiled matmul."""
    print(f"\n{'=' * 60}")
    print("  BLOCK SIZE SWEEP (tiled_matmul)")
    print(f"{'=' * 60}")

    M, N, K = 1024, 1024, 512
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)

    block_sizes = [16, 32, 64, 128]
    results = []

    for bm in block_sizes:
        for bn in block_sizes:
            for bk in block_sizes:

                def runner():
                    return tiled_matmul(a, b, block_m=bm, block_n=bn, block_k=bk)

                name = f"tiled_bm{bm}_bn{bn}_bk{bk}"
                r = benchmark_kernel(fn=runner, name=name, config=CONFIG)
                elapsed = r.p50_ms / 1000.0
                r.gflops = _compute_matmul_gflops(M, N, K, elapsed)
                results.append(r)

    results.sort(key=lambda r: r.gflops, reverse=True)

    print(f"\n  Top 10 configurations by GFLOPS (matrix {M}x{K}x{N}):")
    print(f"  {'Rank':>4}  {'Config':>30}  {'GFLOPS':>10}  {'Time (ms)':>10}")
    print(f"  {'-' * 4}  {'-' * 30}  {'-' * 10}  {'-' * 10}")
    for i, r in enumerate(results[:10]):
        print(f"  {i + 1:>4}  {r.name:>30}  {r.gflops:>8.2f} GF  {r.p50_ms:>8.4f} ms")

    return results


def bench_num_warps_sweep() -> list[BenchmarkResult]:
    """Sweep num_warps for the optimized matmul."""
    print(f"\n{'=' * 60}")
    print("  NUM WARPS SWEEP (optimized_matmul)")
    print(f"{'=' * 60}")

    M, N, K = 1024, 1024, 512
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)

    warp_counts = [2, 4, 8]
    block_configs = [(64, 64, 32), (128, 128, 32), (128, 128, 64)]
    results = []

    for nw in warp_counts:
        for bm, bn, bk in block_configs:

            def runner():
                return optimized_matmul(a, b, block_m=bm, block_n=bn, block_k=bk, num_warps=nw)

            name = f"opt_w{nw}_bm{bm}_bn{bn}_bk{bk}"
            r = benchmark_kernel(fn=runner, name=name, config=CONFIG)
            elapsed = r.p50_ms / 1000.0
            r.gflops = _compute_matmul_gflops(M, N, K, elapsed)
            results.append(r)

    results.sort(key=lambda r: r.gflops, reverse=True)

    print(f"\n  Top 5 warp configurations by GFLOPS:")
    print(f"  {'Rank':>4}  {'Config':>30}  {'GFLOPS':>10}  {'Time (ms)':>10}")
    print(f"  {'-' * 4}  {'-' * 30}  {'-' * 10}  {'-' * 10}")
    for i, r in enumerate(results[:5]):
        print(f"  {i + 1:>4}  {r.name:>30}  {r.gflops:>8.2f} GF  {r.p50_ms:>8.4f} ms")

    return results


def bench_llm_shapes() -> list[BenchmarkResult]:
    """Benchmark LLM-relevant matmul shapes: QKV projections, FFN, output projection."""
    print(f"\n{'=' * 60}")
    print("  LLM-RELEVANT SHAPES")
    print(f"{'=' * 60}")

    shapes = [
        # (M, N, K) description - M=hidden_dim, N=output_dim, K=hidden_dim
        # QKV projection (hidden x 3*hidden)
        (1024, 3072, 1024, "QKV_proj_1K"),
        (4096, 12288, 4096, "QKV_proj_4K"),
        (8192, 24576, 8192, "QKV_proj_8K"),
        # Output projection (3*hidden/heads x head_dim)
        (1024, 1024, 1024, "Out_proj_1K"),
        (4096, 4096, 4096, "Out_proj_4K"),
        (8192, 8192, 8192, "Out_proj_8K"),
        # Head-specific: batch=1, head_dim variations
        (1, 64, 64, "head_b1_d64_k64"),
        (1, 128, 128, "head_b1_d128_k128"),
        (1024, 64, 1024, "attn_score_1K_d64"),
        (4096, 64, 4096, "attn_score_4K_d64"),
        (1024, 128, 1024, "attn_score_1K_d128"),
        (4096, 128, 4096, "attn_score_4K_d128"),
    ]

    peak = _get_peak_gflops()
    results = []

    for M, N, K, desc in shapes:
        a = torch.randn(M, K, device="cuda", dtype=torch.float32)
        b = torch.randn(K, N, device="cuda", dtype=torch.float32)

        # Naive
        def r_naive():
            return naive_matmul(a, b)

        r = benchmark_kernel(fn=r_naive, name=f"naive_{desc}", config=CONFIG)
        r.gflops = _compute_matmul_gflops(M, N, K, r.p50_ms / 1000.0)
        results.append(r)

        # Tiled
        def r_tiled():
            return tiled_matmul(a, b)

        r = benchmark_kernel(fn=r_tiled, name=f"tiled_{desc}", config=CONFIG)
        r.gflops = _compute_matmul_gflops(M, N, K, r.p50_ms / 1000.0)
        results.append(r)

        # Optimized Triton
        def r_opt():
            return optimized_matmul(a, b, block_m=128, block_n=128, block_k=32, num_warps=4)

        r = benchmark_kernel(fn=r_opt, name=f"opt_{desc}", config=CONFIG)
        r.gflops = _compute_matmul_gflops(M, N, K, r.p50_ms / 1000.0)
        results.append(r)

    print(f"\n  Peak theoretical FP32 GFLOPS: {peak:.0f}")
    print(
        f"  {'Description':>22}  {'Naive GF':>8}  {'Tiled GF':>8}  {'Opt GF':>8}  {'Best %Peak':>10}"
    )
    print(f"  {'-' * 22}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 10}")

    for i in range(0, len(results), 3):
        n = results[i]
        t = results[i + 1]
        o = results[i + 2]
        best = max(n.gflops, t.gflops, o.gflops)
        pct = best / peak * 100.0 if peak > 0 else 0
        desc = n.name.split("_", 1)[1] if "_" in n.name else n.name
        print(f"  {desc:>22}  {n.gflops:>8.1f}  {t.gflops:>8.1f}  {o.gflops:>8.1f}  {pct:>9.1f}%")

    return results


def bench_matmul_methods() -> list[BenchmarkResult]:
    """Compare all matmul methods including torch.matmul and torch.compile."""
    print(f"\n{'=' * 60}")
    print("  MATMUL METHOD COMPARISON")
    print(f"{'=' * 60}")

    M, N, K = 4096, 4096, 4096
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)

    results = []

    # Naive Triton
    def r_naive():
        return naive_matmul(a, b)

    r = benchmark_kernel(fn=r_naive, name="naive_triton", config=CONFIG)
    r.gflops = _compute_matmul_gflops(M, N, K, r.p50_ms / 1000.0)
    results.append(r)

    # Tiled Triton
    def r_tiled():
        return tiled_matmul(a, b, block_m=128, block_n=128, block_k=32)

    r = benchmark_kernel(fn=r_tiled, name="tiled_triton", config=CONFIG)
    r.gflops = _compute_matmul_gflops(M, N, K, r.p50_ms / 1000.0)
    results.append(r)

    # Optimized Triton
    def r_opt():
        return optimized_matmul(a, b, block_m=128, block_n=128, block_k=32, num_warps=4)

    r = benchmark_kernel(fn=r_opt, name="optimized_triton", config=CONFIG)
    r.gflops = _compute_matmul_gflops(M, N, K, r.p50_ms / 1000.0)
    results.append(r)

    # torch.matmul
    def r_torch():
        return torch.matmul(a, b)

    r = benchmark_kernel(fn=r_torch, name="torch_matmul", config=CONFIG)
    r.gflops = _compute_matmul_gflops(M, N, K, r.p50_ms / 1000.0)
    results.append(r)

    # torch.compile(torch.matmul)
    try:
        compiled_fn = torch.compile(lambda x, y: torch.matmul(x, y), fullgraph=True)
        for _ in range(10):
            compiled_fn(a, b)
        torch.cuda.synchronize()

        def r_compiled():
            return compiled_fn(a, b)

        r = benchmark_kernel(fn=r_compiled, name="torch_compile_matmul", config=CONFIG)
        r.gflops = _compute_matmul_gflops(M, N, K, r.p50_ms / 1000.0)
        results.append(r)
    except Exception as e:
        print(f"  Note: torch.compile benchmarking skipped ({e})")

    peak = _get_peak_gflops()

    print(f"\n  Matrix size: {M} x {K} x {N}")
    print(f"  Peak theoretical FP32: {peak:.0f} GFLOPS")
    print(f"  {'Method':>25}  {'Time (ms)':>10}  {'GFLOPS':>10}  {'% Peak':>8}")
    print(f"  {'-' * 25}  {'-' * 10}  {'-' * 10}  {'-' * 8}")
    for r in results:
        pct = r.gflops / peak * 100.0 if peak > 0 else 0
        print(f"  {r.name:>25}  {r.p50_ms:>8.4f} ms  {r.gflops:>8.1f} GF  {pct:>7.1f}%")

    return results


def bench_batched_matmul() -> list[BenchmarkResult]:
    """Benchmark batched matmul vs torch.bmm at various batch sizes."""
    print(f"\n{'=' * 60}")
    print("  BATCHED MATMUL vs torch.bmm")
    print(f"{'=' * 60}")

    configs = [
        (1, 512, 512, 512),
        (4, 512, 512, 512),
        (16, 256, 256, 256),
        (32, 128, 128, 128),
        (64, 64, 64, 64),
        (8, 1024, 1024, 512),
        (4, 4096, 4096, 1024),
    ]

    results = []

    for B, M, N, K in configs:
        a = torch.randn(B, M, K, device="cuda", dtype=torch.float32)
        b_t = torch.randn(B, K, N, device="cuda", dtype=torch.float32)

        # Triton batched
        def r_triton():
            return batched_matmul(a, b_t, block_m=128, block_n=128, block_k=32)

        r = benchmark_kernel(fn=r_triton, name=f"triton_bmm_{B}x{M}x{N}x{K}", config=CONFIG)
        r.gflops = _compute_matmul_gflops(M, N, K, r.p50_ms / 1000.0) * B
        results.append(r)

        # torch.bmm
        def r_torch():
            return torch.bmm(a, b_t)

        r = benchmark_kernel(fn=r_torch, name=f"torch_bmm_{B}x{M}x{N}x{K}", config=CONFIG)
        r.gflops = _compute_matmul_gflops(M, N, K, r.p50_ms / 1000.0) * B
        results.append(r)

    print(f"\n  {'Config':>25}  {'Triton (ms)':>12}  {'torch.bmm (ms)':>12}  {'Ratio':>8}")
    print(f"  {'-' * 25}  {'-' * 12}  {'-' * 12}  {'-' * 8}")
    for i in range(0, len(results), 2):
        tr = results[i]
        th = results[i + 1]
        ratio = th.p50_ms / tr.p50_ms if tr.p50_ms > 0 else 0
        label = tr.name.replace("triton_bmm_", "")
        print(f"  {label:>25}  {tr.p50_ms:>10.4f} ms  {th.p50_ms:>10.4f} ms  {ratio:>6.2f}x")

    return results


def bench_prefill_vs_decode() -> list[BenchmarkResult]:
    """Benchmark matmul shapes specific to prefill (batch>1) vs decode (batch=1)."""
    print(f"\n{'=' * 60}")
    print("  PREFILL vs DECODE PATTERNS")
    print(f"{'=' * 60}")

    # Prefill: batch sequence length is large, compute-bound
    # Decode: batch=1, memory-bound for small head_dim
    configs = [
        # (M, N, K, description)
        (4096, 4096, 4096, "prefill_h4K_o4K"),  # Large prefill
        (8192, 8192, 4096, "prefill_h8K_o8K"),
        (1, 4096, 4096, "decode_b1_h4K"),  # Decode, single token
        (1, 11008, 4096, "decode_b1_ffn_up"),  # FFN up projection
        (11008, 4096, 4096, "decode_b1_ffn_down"),  # FFN down projection
        (1, 64, 4096, "decode_b1_head64"),  # Single head
        (1, 128, 8192, "decode_b1_head128"),
    ]

    results = []
    peak = _get_peak_gflops()

    for M, N, K, desc in configs:
        a = torch.randn(M, K, device="cuda", dtype=torch.float32)
        b = torch.randn(K, N, device="cuda", dtype=torch.float32)

        # Use optimized Triton for best result
        def r_opt():
            return optimized_matmul(a, b, block_m=128, block_n=128, block_k=32, num_warps=4)

        r = benchmark_kernel(fn=r_opt, name=f"opt_{desc}", config=CONFIG)
        r.gflops = _compute_matmul_gflops(M, N, K, r.p50_ms / 1000.0)
        results.append(r)

    print(f"\n  {'Pattern':>25}  {'Time (ms)':>10}  {'GFLOPS':>10}  {'% Peak':>8}  {'Shape':>18}")
    print(f"  {'-' * 25}  {'-' * 10}  {'-' * 10}  {'-' * 8}  {'-' * 18}")
    for i, ((M, N, K, desc), r) in enumerate(zip(configs, results)):
        pct = r.gflops / peak * 100.0 if peak > 0 else 0
        shape = f"{M}x{N}x{K}"
        print(f"  {desc:>25}  {r.p50_ms:>8.4f} ms  {r.gflops:>8.1f} GF  {pct:>7.1f}%  {shape:>18}")

    return results


def main():
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        sys.exit(1)

    try:
        import triton  # noqa: F401
    except ImportError:
        print("Triton is not installed. Exiting.")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    peak = _get_peak_gflops()

    print("=" * 70)
    print("  MATMUL TILING BENCHMARKS")
    print("=" * 70)
    print(f"\n  Device: {device_name}")
    print(f"  Estimated Peak FP32: {peak:.0f} GFLOPS")
    print(f"  PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print(f"  Triton: {triton.__version__}")

    all_results = []
    all_results.extend(bench_block_size_sweep())
    all_results.extend(bench_num_warps_sweep())
    all_results.extend(bench_llm_shapes())
    all_results.extend(bench_matmul_methods())
    all_results.extend(bench_batched_matmul())
    all_results.extend(bench_prefill_vs_decode())

    print(f"\n{'=' * 70}")
    print("  FINAL SUMMARY - All Results")
    print(f"{'=' * 70}")
    compare_kernels(all_results)

    print(f"\n{'=' * 70}")
    print("  BEST CONFIGURATION")
    print(f"{'=' * 70}")

    valid = [r for r in all_results if r.gflops > 0 and "tiled" not in r.name]
    valid.sort(key=lambda r: r.gflops, reverse=True)
    if valid:
        best = valid[0]
        best_pct = best.gflops / peak * 100.0 if peak > 0 else 0
        print(f"  Best kernel: {best.name}")
        print(f"  Achieved: {best.gflops:.1f} GFLOPS ({best_pct:.1f}% of theoretical peak)")
        print(f"  Latency: {best.p50_ms:.4f} ms (p50)")
        print(f"  Device: {best.device}")

    report_md = generate_report(all_results, "05_matmul_tiling/matmul_report")
    print(f"\n{report_md}")


if __name__ == "__main__":
    main()
