#!/usr/bin/env python3
"""
运行工业级 benchmark suite。

借鉴 CUTLASS profiler 和 PyTorch benchmark 的最佳实践，
自动对比所有已实现的 kernel 和 PyTorch 原生实现。

用法:
    python benchmarks/run_benchmark_suite.py           # 运行全部
    python benchmarks/run_benchmark_suite.py --profile  # 带 PyTorch profiler
    python benchmarks/run_benchmark_suite.py --output results/report  # 保存报告
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

# 添加项目根目录到 sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.benchmark_framework import (
    BenchmarkResult,
    BenchmarkSuite,
    KernelConfig,
    ProblemSize,
    estimate_attention_bytes,
    estimate_attention_flops,
    estimate_matmul_bytes,
    estimate_matmul_flops,
    estimate_norm_bytes,
    estimate_norm_flops,
    estimate_elementwise_bytes,
    estimate_elementwise_flops,
)
from benchmarks.gpu_info import detect_gpu, list_all_gpus


# ============================================================================
# CUDA Extension 自动构建
# ============================================================================


def _ensure_extension_built() -> bool:
    """确保 CUDA 扩展已编译。"""
    try:
        import cuda_kernels  # noqa: F401

        return True
    except ImportError:
        setup_py = PROJECT_ROOT / "01_cuda_basics" / "setup.py"
        result = subprocess.run(
            [sys.executable, str(setup_py), "build_ext", "--inplace"],
            cwd=str(setup_py.parent),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Extension build failed:\n{result.stderr}")
            return False
        return True


# ============================================================================
# Benchmark 各个 kernel 类别
# ============================================================================


def bench_matmul_suite(suite: BenchmarkSuite) -> List[BenchmarkResult]:
    """矩阵乘法 benchmark suite。"""
    import cuda_kernels

    results = []
    configs = [
        (1, 4096, 4096, "decode projection"),
        (64, 4096, 4096, "small prefill"),
        (128, 8192, 4096, "FFN gate"),
        (128, 4096, 14336, "LLaMA FFN up"),
        (128, 14336, 4096, "LLaMA FFN down"),
        (1024, 4096, 4096, "medium prefill"),
        (1, 4096, 8192, "decode large"),
        (2048, 4096, 4096, "large prefill"),
    ]

    for M, N, K, desc in configs:
        torch.manual_seed(42)
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B = torch.randn(K, N, device="cuda", dtype=torch.float16)
        C = torch.empty(M, N, device="cuda", dtype=torch.float16)

        flops = estimate_matmul_flops(M, N, K)  # 闭包捕获
        mem_bytes = estimate_matmul_bytes(M, N, K)  # 闭包捕获

        ps = ProblemSize(shape=(M, N, K), description=desc)

        # CUDA tiled matmul
        r_cuda = suite.benchmark_kernel(
            fn=lambda a, b, c: cuda_kernels.tiled_matmul(a, b, c),
            args=(A, B, C),
            config=KernelConfig(name="cuda_tiled_matmul", implementation="cuda"),
            problem_size=ps,
            flop_fn=lambda *a, fl=flops: fl,
            bytes_fn=lambda *a, mb=mem_bytes: mb,
        )
        results.append(r_cuda)

        # torch.matmul
        r_torch = suite.benchmark_torch_op(
            fn=lambda a, b: torch.matmul(a.float(), b.float()).to(torch.float16),
            a=A,
            b=B,
            name="torch.matmul",
            problem_size=ps,
            flop_fn=lambda *a, fl=flops: fl,
            bytes_fn=lambda *a, mb=mem_bytes: mb,
        )
        results.append(r_torch)

        # torch.compile matmul
        def compiled_matmul(a, b):
            return torch.matmul(a.float(), b.float()).to(torch.float16)

        compiled_fn = torch.compile(compiled_matmul, dynamic=False)
        for _ in range(5):
            compiled_fn(A, B)
        torch.cuda.synchronize()

        r_compiled = suite.benchmark_kernel(
            fn=compiled_fn,
            args=(A, B),
            config=KernelConfig(name="torch.compile_matmul", implementation="torch_compile"),
            problem_size=ps,
            flop_fn=lambda *a, fl=flops: fl,
            bytes_fn=lambda *a, mb=mem_bytes: mb,
        )
        results.append(r_compiled)

        print(
            f"  matmul {desc:25s} cuda={r_cuda.latency_p50_us:.0f}us "
            f"torch={r_torch.latency_p50_us:.0f}us "
            f"compile={r_compiled.latency_p50_us:.0f}us"
        )

    return results


def bench_attention_suite(suite: BenchmarkSuite) -> List[BenchmarkResult]:
    """Attention benchmark suite。"""
    import cuda_kernels

    results = []
    configs = [
        (2, 4, 64, 64, False),
        (1, 8, 128, 64, False),
        (2, 4, 128, 128, False),
        (1, 8, 256, 64, False),
        (1, 8, 512, 64, False),
    ]

    for batch, n_heads, seq_len, head_dim, causal in configs:
        torch.manual_seed(42)
        scale = 1.0 / math.sqrt(head_dim)
        shape = (batch, n_heads, seq_len, head_dim)

        Q = torch.randn(shape, device="cuda", dtype=torch.float16)
        K = torch.randn(shape, device="cuda", dtype=torch.float16)
        V = torch.randn(shape, device="cuda", dtype=torch.float16)
        O = torch.empty(shape, device="cuda", dtype=torch.float16)

        flops = estimate_attention_flops(batch, n_heads, seq_len, head_dim)
        mem_bytes = estimate_attention_bytes(batch, n_heads, seq_len, head_dim)

        ps = ProblemSize(
            shape=(batch, n_heads, seq_len, head_dim),
            description=f"attn {'causal' if causal else 'full'}",
        )

        # FlashAttention
        r_fa = suite.benchmark_kernel(
            fn=lambda q, k, v, o, s, c: cuda_kernels.flash_attention_fwd(q, k, v, o, s, c),
            args=(Q, K, V, O, scale, causal),
            config=KernelConfig(
                name="flash_attention",
                implementation="cuda",
            ),
            problem_size=ps,
            flop_fn=lambda *a, fl=flops: fl,
            bytes_fn=lambda *a, mb=mem_bytes: mb,
        )
        results.append(r_fa)

        # PyTorch sdpa
        r_torch = suite.benchmark_torch_op(
            torch.nn.functional.scaled_dot_product_attention,
            Q,
            K,
            V,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=causal,
            scale=scale,
            name="torch.sdpa",
            problem_size=ps,
            flop_fn=lambda *a, fl=flops: fl,
            bytes_fn=lambda *a, mb=mem_bytes: mb,
        )
        results.append(r_torch)

        print(
            f"  attn {seq_len:4d}x{head_dim:3d} {'causal' if causal else 'full':7s} "
            f"flash={r_fa.latency_p50_us:.0f}us "
            f"torch={r_torch.latency_p50_us:.0f}us"
        )

    return results


def bench_norm_suite(suite: BenchmarkSuite) -> List[BenchmarkResult]:
    """Normalization benchmark suite (RMSNorm, LayerNorm)。"""
    import cuda_kernels

    results = []
    configs = [
        (16, 768),
        (128, 768),
        (16, 4096),
        (128, 4096),
        (256, 4096),
    ]

    for rows, hidden_dim in configs:
        eps = 1e-5
        x = torch.randn(rows, hidden_dim, device="cuda", dtype=torch.float16)
        weight = torch.randn(hidden_dim, device="cuda", dtype=torch.float16)
        bias = torch.randn(hidden_dim, device="cuda", dtype=torch.float16)
        out = torch.empty(rows, hidden_dim, device="cuda", dtype=torch.float16)

        flops = estimate_norm_flops(rows, hidden_dim)
        mem_bytes = estimate_norm_bytes(rows, hidden_dim)

        ps = ProblemSize(shape=(rows, hidden_dim), description="norm")

        # RMSNorm CUDA
        r_rms = suite.benchmark_kernel(
            fn=lambda x_, w_, o_: cuda_kernels.rmsnorm_fwd(x_, w_, o_, eps),
            args=(x, weight, out),
            config=KernelConfig(name="cuda_rmsnorm", implementation="cuda"),
            problem_size=ps,
            flop_fn=lambda *a, f=flops: f,
            bytes_fn=lambda *a, mb=mem_bytes: mb,
        )
        results.append(r_rms)

        # LayerNorm CUDA
        r_ln = suite.benchmark_kernel(
            fn=lambda x_, w_, b_, o_: cuda_kernels.layernorm_fwd(x_, w_, b_, o_, eps),
            args=(x, weight, bias, out),
            config=KernelConfig(name="cuda_layernorm", implementation="cuda"),
            problem_size=ps,
            flop_fn=lambda *a, f=flops: f,
            bytes_fn=lambda *a, mb=mem_bytes: mb,
        )
        results.append(r_ln)

        # torch.nn.LayerNorm
        ln = torch.nn.LayerNorm(hidden_dim, eps=eps, device="cuda", dtype=torch.float16)
        with torch.no_grad():
            ln.weight.copy_(weight.float().to(torch.float16))
            ln.bias.copy_(bias.float().to(torch.float16))

        r_torch_ln = suite.benchmark_torch_op(
            ln,
            x,
            name="torch.LayerNorm",
            problem_size=ps,
            flop_fn=lambda *a, f=flops: f,
            bytes_fn=lambda *a, mb=mem_bytes: mb,
        )
        results.append(r_torch_ln)

        print(
            f"  norm {rows:4d}x{hidden_dim:5d} "
            f"rms={r_rms.latency_p50_us:.0f}us "
            f"ln={r_ln.latency_p50_us:.0f}us "
            f"torch={r_torch_ln.latency_p50_us:.0f}us"
        )

    return results


def bench_activation_suite(suite: BenchmarkSuite) -> List[BenchmarkResult]:
    """Activation benchmark suite (SiLU, GELU, SwiGLU)。"""
    import cuda_kernels

    results = []
    sizes = [1024, 65536, 1048576, 8388608]  # 1K, 64K, 1M, 8M

    for n in sizes:
        x = torch.randn(n, device="cuda", dtype=torch.float16)
        out = torch.empty(n, device="cuda", dtype=torch.float16)

        flops = estimate_elementwise_flops(n)
        mem_bytes = estimate_elementwise_bytes(n)

        ps = ProblemSize(shape=(n,), description="activation")

        # SiLU
        r_silu = suite.benchmark_kernel(
            fn=lambda x_, o_: cuda_kernels.silu_fwd(x_, o_),
            args=(x, out),
            config=KernelConfig(name="cuda_silu", implementation="cuda"),
            problem_size=ps,
            flop_fn=lambda *a, f=flops: f,
            bytes_fn=lambda *a, mb=mem_bytes: mb,
        )
        results.append(r_silu)

        # GELU
        r_gelu = suite.benchmark_kernel(
            fn=lambda x_, o_: cuda_kernels.gelu_fwd(x_, o_),
            args=(x, out),
            config=KernelConfig(name="cuda_gelu", implementation="cuda"),
            problem_size=ps,
            flop_fn=lambda *a, f=flops: f,
            bytes_fn=lambda *a, mb=mem_bytes: mb,
        )
        results.append(r_gelu)

        print(
            f"  activation {n:10d} "
            f"silu={r_silu.latency_p50_us:.0f}us "
            f"gelu={r_gelu.latency_p50_us:.0f}us"
        )

    return results


def bench_reduction_suite(suite: BenchmarkSuite) -> List[BenchmarkResult]:
    """Reduction benchmark suite。"""
    import cuda_kernels

    results = []
    sizes = [1024, 32768, 1048576, 16777216]

    for n in sizes:
        x = torch.randn(n, device="cuda", dtype=torch.float32)

        ps = ProblemSize(shape=(n,), description="reduction")

        r_warp = suite.benchmark_kernel(
            fn=lambda x_: cuda_kernels.warp_reduce_sum(x_),
            args=(x,),
            config=KernelConfig(name="warp_reduce_sum", implementation="cuda"),
            problem_size=ps,
            flop_fn=lambda *a, f=n: f,
            bytes_fn=lambda *a, mb=n * 4: mb,
        )
        results.append(r_warp)

        r_full = suite.benchmark_kernel(
            fn=lambda x_: cuda_kernels.full_warp_reduction(x_),
            args=(x,),
            config=KernelConfig(name="full_warp_reduction", implementation="cuda"),
            problem_size=ps,
            flop_fn=lambda *a, f=n: f,
            bytes_fn=lambda *a, mb=n * 4: mb,
        )
        results.append(r_full)

        r_torch = suite.benchmark_torch_op(
            torch.sum,
            x,
            name="torch.sum",
            problem_size=ps,
            flop_fn=lambda *a, f=n: f,
            bytes_fn=lambda *a, mb=n * 4: mb,
        )
        results.append(r_torch)

        print(
            f"  reduction {n:10d} "
            f"warp={r_warp.latency_p50_us:.0f}us "
            f"full={r_full.latency_p50_us:.0f}us "
            f"torch={r_torch.latency_p50_us:.0f}us"
        )

    return results


def bench_paged_attention_suite(suite: BenchmarkSuite) -> List[BenchmarkResult]:
    """PagedAttention benchmark suite。"""
    import cuda_kernels

    results = []
    configs = [
        (4, 64, 16, 32),
        (8, 64, 16, 128),
        (4, 128, 16, 128),
        (8, 64, 16, 256),
    ]

    for num_heads, head_dim, block_size, context_len in configs:
        torch.manual_seed(42)
        scale = 1.0 / math.sqrt(head_dim)
        num_blocks = (context_len + block_size - 1) // block_size * 2

        K_cache, V_cache = cuda_kernels.allocate_kv_cache(
            num_blocks, block_size, num_heads, head_dim
        )
        K_cache.normal_()
        V_cache.normal_()

        num_needed = (context_len + block_size - 1) // block_size
        block_table = list(range(num_needed))
        while len(block_table) < 4:
            block_table.append(-1)

        context_lens = torch.tensor([context_len], device="cuda", dtype=torch.int32)
        block_tables = torch.tensor([block_table], device="cuda", dtype=torch.int32)

        Q = torch.randn(num_heads, head_dim, device="cuda", dtype=torch.float16)
        O = torch.empty(num_heads, head_dim, device="cuda", dtype=torch.float16)

        ps = ProblemSize(
            shape=(num_heads, head_dim, context_len),
            description="paged_attention",
        )

        r_paged = suite.benchmark_kernel(
            fn=lambda q, kc, vc, bt, cl, o, s: cuda_kernels.paged_attention(
                q, kc, vc, bt, cl, o, s
            ),
            args=(Q, K_cache, V_cache, block_tables, context_lens, O, scale),
            config=KernelConfig(name="paged_attention", implementation="cuda"),
            problem_size=ps,
        )
        results.append(r_paged)

        # 手动参考实现
        K_ref_list = []
        V_ref_list = []
        collected = 0
        for bi in block_table:
            if bi < 0:
                break
            take = min(block_size, context_len - collected)
            K_ref_list.append(K_cache[bi, :take].reshape(-1, num_heads, head_dim))
            V_ref_list.append(V_cache[bi, :take].reshape(-1, num_heads, head_dim))
            collected += take

        K_ref = torch.cat(K_ref_list, dim=0)
        V_ref = torch.cat(V_ref_list, dim=0)

        def manual_attn(q, k, v, s):
            qf = q.float().unsqueeze(0)
            kf = k.float().permute(1, 2, 0)
            vf = v.float().permute(1, 0, 2)
            scores = torch.bmm(qf, kf) * s
            attn_w = torch.softmax(scores, dim=-1)
            return torch.bmm(attn_w, vf).squeeze(0).to(torch.float16)

        r_manual = suite.benchmark_kernel(
            fn=manual_attn,
            args=(Q, K_ref, V_ref, scale),
            config=KernelConfig(name="manual_attn", implementation="torch"),
            problem_size=ps,
        )
        results.append(r_manual)

        print(
            f"  paged_attn ctx={context_len:4d}d={head_dim:3d} "
            f"paged={r_paged.latency_p50_us:.0f}us "
            f"manual={r_manual.latency_p50_us:.0f}us"
        )

    return results


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Industrial GPU Kernel Benchmark Suite")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output base path for reports (saves .md, .csv, .json).",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable PyTorch profiler during benchmarks.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer repetitions for faster results.",
    )
    parser.add_argument(
        "--list-gpus",
        action="store_true",
        help="List all known GPU specs and exit.",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["matmul", "attention", "norm", "activation", "reduction", "paged"],
        help="Skip specific benchmark categories.",
    )
    args = parser.parse_args()

    if args.list_gpus:
        list_all_gpus()
        print()
        detected = detect_gpu()
        if detected:
            print(f"Detected: {detected.model}")
            print(f"  Peak FP32:     {detected.peak_fp32_tflops} TFLOPS")
            print(f"  Peak TC FP16:  {detected.peak_tensor_core_fp16_tflops} TFLOPS")
            print(f"  Peak BW:       {detected.memory_bandwidth_gbps} GB/s")
            print(f"  Ridge FP32:    {detected.ridge_point_fp32:.1f} FLOP/Byte")
            print(f"  Ridge TC FP16: {detected.ridge_point_tc_fp16:.1f} FLOP/Byte")
        return

    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        sys.exit(1)

    # 确保扩展已编译
    if not _ensure_extension_built():
        print("Cannot build CUDA extension. Exiting.")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print(f"  Industrial GPU Kernel Benchmark Suite")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print(f"{'=' * 70}\n")

    w, r = (3, 25) if args.quick else (10, 100)
    suite = BenchmarkSuite(warmup=w, repeat=r)

    all_results: List[BenchmarkResult] = []

    skip_set = set(args.skip)

    if "matmul" not in skip_set:
        print("--- Matmul Benchmarks ---")
        all_results.extend(bench_matmul_suite(suite))
        print()

    if "attention" not in skip_set:
        print("--- Attention Benchmarks ---")
        all_results.extend(bench_attention_suite(suite))
        print()

    if "norm" not in skip_set:
        print("--- Normalization Benchmarks ---")
        all_results.extend(bench_norm_suite(suite))
        print()

    if "activation" not in skip_set:
        print("--- Activation Benchmarks ---")
        all_results.extend(bench_activation_suite(suite))
        print()

    if "reduction" not in skip_set:
        print("--- Reduction Benchmarks ---")
        all_results.extend(bench_reduction_suite(suite))
        print()

    if "paged" not in skip_set:
        print("--- PagedAttention Benchmarks ---")
        all_results.extend(bench_paged_attention_suite(suite))
        print()

    # 生成报告
    print(f"\n{'=' * 70}")
    print("  Results Summary")
    print(f"{'=' * 70}\n")
    suite.print_comparison_table(all_results)

    output_path = args.output
    if output_path is None:
        output_path = str(PROJECT_ROOT / "benchmark_report")

    suite.generate_report(results=all_results, output_path=output_path)


if __name__ == "__main__":
    main()
