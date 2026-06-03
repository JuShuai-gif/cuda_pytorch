"""
Unified benchmark for all project kernels.

Provides a single entry point to benchmark every kernel developed across
modules and compare against baseline implementations.

Categories:
  1. Elementwise: add, mul, relu, gelu, silu, sigmoid, tanh
  2. Activation fusions: add+relu, bias+gelu, residual+layernorm
  3. Normalization: layernorm, rmsnorm
  4. Attention: naive, tiled, flash-like for prefill/decode
  5. Matmul: naive, tiled, optimized across block sizes
  6. Memory: copy bw, strided access bandwidth
  7. Stream ops: H2D, D2H, overlap pipeline

Usage:
    python 12_inference_pipeline/benchmark_all_ops.py
    python 12_inference_pipeline/benchmark_all_ops.py --output report
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import torch
import triton

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "02_triton_basics"))
sys.path.insert(0, str(_PROJECT_ROOT / "03_memory_bandwidth"))
sys.path.insert(0, str(_PROJECT_ROOT / "04_operator_fusion"))
sys.path.insert(0, str(_PROJECT_ROOT / "06_attention_flash_like"))
sys.path.insert(0, str(_PROJECT_ROOT / "07_cuda_streams_async"))

from benchmarks.benchmark_utils import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkConfig,
    benchmark_kernel,
    compare_kernels,
    generate_report,
)
from triton_elementwise import triton_relu, triton_gelu, triton_silu
from triton_vector_add import triton_vector_add
from triton_gemm_basic import triton_gemm
from kernel_add_relu import fused_add_relu, sequential_add_relu
from kernel_bias_gelu import fused_bias_gelu, sequential_bias_gelu
from kernel_residual_layernorm import fused_residual_layernorm, sequential_residual_layernorm
from kernel_rmsnorm import triton_rmsnorm, torch_rmsnorm
from naive_attention import naive_attention_torch, naive_attention_triton
from tiled_attention import tiled_attention, _scaled_dot_product_attention_ref
from flash_attention_kv_cache import attention_prefill, attention_decode

CONFIG = BenchmarkConfig(warmup_steps=5, measure_steps=20, repeat=3)


def _to_result(name: str, fn: Callable, *args, **kwargs) -> BenchmarkResult:
    """Helper to benchmark a function with given args."""
    return benchmark_kernel(fn, args=args, kwargs=kwargs, name=name, config=CONFIG)


# ---------------------------------------------------------------------------
# Category 1: Elementwise kernels
# ---------------------------------------------------------------------------


def bench_elementwise() -> list[BenchmarkResult]:
    """Benchmark elementwise activation kernels."""
    print(f"\n{'=' * 60}")
    print("  CATEGORY 1: ELEMENTWISE KERNELS")
    print(f"{'=' * 60}")

    results = []
    sizes = [1024, 4096, 16384, 65536, 262144, 1_048_576, 4_194_304, 16_777_216]

    for n in sizes:
        x = torch.randn(n, device="cuda", dtype=torch.float32)

        results.append(_to_result(f"add_vec_{n}", lambda a, b: a + b, x, x))
        results.append(_to_result(f"triton_add_vec_{n}", triton_vector_add, x, x))

        results.append(_to_result(f"relu_torch_{n}", torch.relu, x))
        results.append(_to_result(f"relu_triton_{n}", triton_relu, x))

        results.append(
            _to_result(
                f"gelu_torch_{n}", lambda t: torch.nn.functional.gelu(t, approximate="tanh"), x
            )
        )
        results.append(_to_result(f"gelu_triton_{n}", triton_gelu, x))

        results.append(_to_result(f"silu_torch_{n}", torch.nn.functional.silu, x))
        results.append(_to_result(f"silu_triton_{n}", triton_silu, x))

    return results


# ---------------------------------------------------------------------------
# Category 2: Activation fusions
# ---------------------------------------------------------------------------


def bench_activation_fusions() -> list[BenchmarkResult]:
    """Benchmark fused activation kernels."""
    print(f"\n{'=' * 60}")
    print("  CATEGORY 2: ACTIVATION FUSIONS")
    print(f"{'=' * 60}")

    results = []
    dims = [1024, 4096, 8192, 32768, 131072, 524288]

    for dim in dims:
        x = torch.randn(dim, device="cuda", dtype=torch.float32)
        bias = torch.randn(dim, device="cuda", dtype=torch.float32)

        results.append(_to_result(f"add_relu_seq_{dim}", sequential_add_relu, x, bias))
        results.append(_to_result(f"add_relu_fused_{dim}", fused_add_relu, x, bias))

        results.append(_to_result(f"bias_gelu_seq_{dim}", sequential_bias_gelu, x, bias))
        results.append(_to_result(f"bias_gelu_fused_{dim}", fused_bias_gelu, x, bias))

    # Residual + LayerNorm (2D required)
    rln_dims = [(4, 512), (4, 1024), (8, 2048), (16, 4096), (32, 1024), (64, 768)]
    for rows, cols in rln_dims:
        x2 = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
        res = torch.randn(rows, cols, device="cuda", dtype=torch.float32)

        results.append(
            _to_result(
                f"res_ln_seq_{rows}x{cols}",
                sequential_residual_layernorm,
                x2,
                res,
            )
        )
        results.append(
            _to_result(
                f"res_ln_fused_{rows}x{cols}",
                fused_residual_layernorm,
                x2,
                res,
                cols,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Category 3: Normalization
# ---------------------------------------------------------------------------


def bench_normalization() -> list[BenchmarkResult]:
    """Benchmark normalization kernels."""
    print(f"\n{'=' * 60}")
    print("  CATEGORY 3: NORMALIZATION")
    print(f"{'=' * 60}")

    results = []
    rms_dims = [(4, 512), (4, 1024), (8, 2048), (16, 4096), (32, 4096), (64, 8192)]

    for rows, cols in rms_dims:
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
        weight = torch.ones(cols, device="cuda", dtype=torch.float32)

        results.append(_to_result(f"rmsnorm_torch_{rows}x{cols}", torch_rmsnorm, x, weight))
        results.append(_to_result(f"rmsnorm_triton_{rows}x{cols}", triton_rmsnorm, x, weight, cols))

        # Also benchmark torch layernorm for comparison
        results.append(
            _to_result(
                f"layernorm_torch_{rows}x{cols}",
                lambda t: torch.nn.functional.layer_norm(t, [t.shape[-1]]),
                x,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Category 4: Attention
# ---------------------------------------------------------------------------


def bench_attention() -> list[BenchmarkResult]:
    """Benchmark attention kernels across prefill/decode scenarios."""
    print(f"\n{'=' * 60}")
    print("  CATEGORY 4: ATTENTION")
    print(f"{'=' * 60}")

    results = []
    configs = [
        (1, 4, 64, 64, 64),
        (1, 8, 128, 128, 64),
        (1, 4, 256, 256, 64),
        (2, 8, 128, 128, 64),
        (1, 4, 512, 512, 64),
    ]

    for B, H, Q_len, KV_len, D in configs:
        q = torch.randn(B, H, Q_len, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, H, KV_len, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, H, KV_len, D, device="cuda", dtype=torch.float32)

        label = f"{B}x{H}x{Q_len}x{KV_len}x{D}"

        results.append(_to_result(f"attn_naive_torch_{label}", naive_attention_torch, q, k, v))
        results.append(_to_result(f"attn_naive_triton_{label}", naive_attention_triton, q, k, v))
        results.append(_to_result(f"attn_tiled_{label}", tiled_attention, q, k, v))
        results.append(_to_result(f"attn_prefill_{label}", attention_prefill, q, k, v))
        results.append(
            _to_result(
                f"attn_torch_sdpa_{label}",
                lambda qq, kk, vv: torch.nn.functional.scaled_dot_product_attention(
                    qq, kk, vv, scale=1.0 / math.sqrt(D)
                ),
                q,
                k,
                v,
            )
        )

    # Decode pattern: Q_len=1, various KV lengths
    decode_configs = [(1, 4, 64, 64), (1, 4, 128, 64), (1, 8, 256, 64), (1, 4, 512, 64)]
    for B, H, KV_len, D in decode_configs:
        q = torch.randn(B, H, 1, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, H, KV_len, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, H, KV_len, D, device="cuda", dtype=torch.float32)

        label = f"{B}x{H}x1x{KV_len}x{D}"
        results.append(_to_result(f"attn_decode_{label}", attention_decode, q, k, v))
        results.append(_to_result(f"attn_tiled_decode_{label}", tiled_attention, q, k, v))
        results.append(
            _to_result(
                f"attn_torch_sdpa_decode_{label}",
                lambda qq, kk, vv: torch.nn.functional.scaled_dot_product_attention(
                    qq, kk, vv, scale=1.0 / math.sqrt(D)
                ),
                q,
                k,
                v,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Category 5: Matmul
# ---------------------------------------------------------------------------


def bench_matmul() -> list[BenchmarkResult]:
    """Benchmark matmul kernels."""
    print(f"\n{'=' * 60}")
    print("  CATEGORY 5: MATMUL")
    print(f"{'=' * 60}")

    results = []
    sizes = [(256, 256, 256), (512, 512, 256), (1024, 1024, 256), (2048, 1024, 512)]

    for M, N, K in sizes:
        a = torch.randn(M, K, device="cuda", dtype=torch.float32)
        b = torch.randn(K, N, device="cuda", dtype=torch.float32)

        label = f"{M}x{N}x{K}"
        results.append(_to_result(f"matmul_torch_{label}", torch.matmul, a, b))

        for bm, bn, bk in [(32, 32, 32), (64, 64, 32), (64, 64, 64)]:
            results.append(
                _to_result(
                    f"matmul_triton_{label}_B{bm}x{bn}x{bk}",
                    triton_gemm,
                    a,
                    b,
                    bm,
                    bn,
                    bk,
                )
            )

    return results


# ---------------------------------------------------------------------------
# Category 6: Memory bandwidth
# ---------------------------------------------------------------------------


def bench_memory() -> list[BenchmarkResult]:
    """Benchmark memory operations."""
    print(f"\n{'=' * 60}")
    print("  CATEGORY 6: MEMORY BANDWIDTH")
    print(f"{'=' * 60}")

    results = []
    sizes = [1_048_576, 4_194_304, 16_777_216, 67_108_864]

    try:
        from triton_copy import copy_kernel, copy_vectorized, copy_non_contiguous

        for n in sizes:
            x = torch.randn(n, device="cuda", dtype=torch.float32)

            results.append(_to_result(f"copy_triton_{n}", copy_kernel, x))
            results.append(_to_result(f"copy_vec_{n}", copy_vectorized, x, 4))
            results.append(_to_result(f"copy_clone_{n}", x.clone))

        # Strided access
        for stride in [1, 4, 16, 64]:
            n = 1_048_576 * stride
            x = torch.randn(n, device="cuda", dtype=torch.float32)
            results.append(
                _to_result(
                    f"copy_stride{stride}",
                    copy_non_contiguous,
                    x,
                    stride,
                )
            )

    except ImportError:
        print("  [SKIP] triton_copy module not available")

    return results


# ---------------------------------------------------------------------------
# Category 7: Stream operations
# ---------------------------------------------------------------------------


def bench_streams() -> list[BenchmarkResult]:
    """Benchmark stream-related operations."""
    print(f"\n{'=' * 60}")
    print("  CATEGORY 7: STREAM OPERATIONS")
    print(f"{'=' * 60}")

    results = []
    sizes = [1_048_576, 4_194_304, 16_777_216]

    for n in sizes:
        x_cpu = torch.randn(n, dtype=torch.float32)

        # H2D transfer
        results.append(
            _to_result(
                f"h2d_{n}",
                lambda src: src.to("cuda", non_blocking=True),
                x_cpu,
            )
        )

        # D2H transfer
        x_gpu = x_cpu.to("cuda")
        results.append(
            _to_result(
                f"d2h_{n}",
                lambda src: src.to("cpu", non_blocking=True),
                x_gpu,
            )
        )

        # Overlap: two streams compute kernel + H2D transfer
        def overlap_workload(a, b, stream1, stream2):
            with torch.cuda.stream(stream1):
                c1 = torch.relu(a)
            with torch.cuda.stream(stream2):
                c2 = b.to("cuda", non_blocking=True)
            torch.cuda.synchronize()
            return c1, c2

        a = torch.randn(n, device="cuda", dtype=torch.float32)
        b = torch.randn(n, dtype=torch.float32)
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()

        results.append(_to_result(f"overlap_h2d_kernel_{n}", overlap_workload, a, b, s1, s2))

    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_all_benchmarks(output_path: Optional[str] = None) -> list[BenchmarkResult]:
    """Run benchmarks for all kernel types and generate unified report.

    Returns:
        List of BenchmarkResult objects.
    """
    all_results: list[BenchmarkResult] = []

    print("\n" + "=" * 70)
    print("  UNIFIED KERNEL BENCHMARK - ALL CATEGORIES")
    print("=" * 70)
    print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Triton: {triton.__version__}")
    print("=" * 70)

    t_start = time.perf_counter()

    try:
        all_results.extend(bench_elementwise())
    except Exception as e:
        print(f"  [ERROR] Elementwise benchmarks failed: {e}")

    try:
        all_results.extend(bench_activation_fusions())
    except Exception as e:
        print(f"  [ERROR] Activation fusion benchmarks failed: {e}")

    try:
        all_results.extend(bench_normalization())
    except Exception as e:
        print(f"  [ERROR] Normalization benchmarks failed: {e}")

    try:
        all_results.extend(bench_attention())
    except Exception as e:
        print(f"  [ERROR] Attention benchmarks failed: {e}")

    try:
        all_results.extend(bench_matmul())
    except Exception as e:
        print(f"  [ERROR] Matmul benchmarks failed: {e}")

    try:
        all_results.extend(bench_memory())
    except Exception as e:
        print(f"  [ERROR] Memory benchmarks failed: {e}")

    try:
        all_results.extend(bench_streams())
    except Exception as e:
        print(f"  [ERROR] Stream benchmarks failed: {e}")

    elapsed = time.perf_counter() - t_start

    print(f"\n{'=' * 70}")
    print(f"  BENCHMARK COMPLETE: {len(all_results)} results in {elapsed:.1f}s")
    print(f"{'=' * 70}")

    # Print comparison table
    if all_results:
        compare_kernels(all_results)

    # Generate report
    if output_path:
        generate_report(all_results, output_path)

    return all_results


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. Cannot run benchmarks.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Unified benchmark for all project kernels")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Base path for output report files (without extension)",
    )
    args = parser.parse_args()

    run_all_benchmarks(output_path=args.output)
