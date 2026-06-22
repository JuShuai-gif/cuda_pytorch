"""
Attention Profiling Script - Chapter 03

Demonstrates:
1. torch.profiler for operator-level profiling
2. nvtx annotations for custom ranges
3. CUDA events for micro-benchmarking
4. Memory bandwidth and FLOPs estimation

Run with:
    python attention_profile.py
    nsys profile -o ch03_profile python attention_profile.py
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    tensorboard_trace_handler,
)

OUT_DIR = Path(__file__).resolve().parent / "profiles"
OUT_DIR.mkdir(exist_ok=True)


def estimate_flops(N: int, d: int) -> float:
    """Estimate total FLOPs for one attention forward pass."""
    qk = 2.0 * N * N * d  # Q @ K^T
    softmax = 4.0 * N * N  # exp + sum + normalize
    pv = 2.0 * N * N * d  # P @ V
    return qk + softmax + pv


def estimate_memory(N: int, d: int, dtype_size: int = 2) -> dict:
    """Estimate memory usage in MB."""
    return {
        "Q": N * d * dtype_size / (1024**2),
        "K": N * d * dtype_size / (1024**2),
        "V": N * d * dtype_size / (1024**2),
        "S": N * N * dtype_size / (1024**2),
        "P": N * N * dtype_size / (1024**2),
        "O": N * d * dtype_size / (1024**2),
        "total_io": (3 * N * d + 2 * N * N + N * d) * dtype_size / (1024**2),
    }


def run_torch_profiler(N: int, d_k: int, d_v: int):
    """Profile attention with torch.profiler."""
    Q = torch.randn(1, 1, N, d_k, device="cuda", dtype=torch.float16)
    K = torch.randn(1, 1, N, d_k, device="cuda", dtype=torch.float16)
    V = torch.randn(1, 1, N, d_v, device="cuda", dtype=torch.float16)

    print(f"\n--- Torch Profiler: N={N}, d_k={d_k} ---")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
        on_trace_ready=tensorboard_trace_handler(str(OUT_DIR / f"trace_N{N}")),
    ) as prof:
        with record_function("naive_attention"):
            # Step 1: Q @ K^T
            with record_function("QK_matmul"):
                S = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

            # Step 2: Softmax
            with record_function("softmax"):
                P = F.softmax(S, dim=-1)

            # Step 3: P @ V
            with record_function("PV_matmul"):
                O = torch.matmul(P, V)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Compare with fused SDPA
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as prof_fused:
        with record_function("fused_sdpa"):
            O_fused = F.scaled_dot_product_attention(Q, K, V)

    print("\n--- Fused SDPA ---")
    print(prof_fused.key_averages().table(sort_by="cuda_time_total", row_limit=10))


def run_micro_benchmark(N: int, d_k: int, d_v: int, iters: int = 100):
    """Micro-benchmark with CUDA events."""
    Q = torch.randn(1, 1, N, d_k, device="cuda", dtype=torch.float16)
    K = torch.randn(1, 1, N, d_k, device="cuda", dtype=torch.float16)
    V = torch.randn(1, 1, N, d_v, device="cuda", dtype=torch.float16)

    # Warmup
    for _ in range(10):
        _ = F.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()

    # Benchmark naive
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        S = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        P = F.softmax(S, dim=-1)
        O = torch.matmul(P, V)
    end.record()
    torch.cuda.synchronize()
    naive_ms = start.elapsed_time(end) / iters

    # Benchmark fused
    start_fused = torch.cuda.Event(enable_timing=True)
    end_fused = torch.cuda.Event(enable_timing=True)

    start_fused.record()
    for _ in range(iters):
        _ = F.scaled_dot_product_attention(Q, K, V)
    end_fused.record()
    torch.cuda.synchronize()
    fused_ms = start_fused.elapsed_time(end_fused) / iters

    # Compute metrics
    flops = estimate_flops(N, d_k)
    mem = estimate_memory(N, d_k)

    naive_tflops = (flops / 1e12) / (naive_ms / 1000)
    fused_tflops = (flops / 1e12) / (fused_ms / 1000)

    prop = torch.cuda.get_device_properties(0)
    peak_bw_gbs = prop.total_memory * 0

    print(f"\n--- Micro-benchmark: N={N}, d_k={d_k} ---")
    print(f"  Naive:          {naive_ms:10.3f} ms  ({naive_tflops:8.2f} TFLOPS)")
    print(f"  Fused SDPA:     {fused_ms:10.3f} ms  ({fused_tflops:8.2f} TFLOPS)")
    print(f"  Speedup:        {naive_ms / fused_ms:10.2f}x")
    print(f"  Total FLOPs:    {flops / 1e9:10.2f} GFLOPs")
    print(f"  IO Memory:      {mem['total_io']:10.2f} MB")
    print(f"  S+P peak mem:   {mem['S'] + mem['P']:10.2f} MB")
    print(
        f"  Arithmetic Intensity: {flops / (mem['total_io'] * 1024 * 1024):.1f} FLOPs/Byte"
    )

    return {
        "N": N,
        "naive_ms": naive_ms,
        "fused_ms": fused_ms,
        "naive_tflops": naive_tflops,
        "fused_tflops": fused_tflops,
        "speedup": naive_ms / fused_ms,
    }


def main():
    parser = argparse.ArgumentParser(description="Chapter 03: Attention Profiling")
    parser.add_argument(
        "--seq-len",
        type=int,
        nargs="+",
        default=[256, 512, 1024, 2048],
        help="Sequence lengths to benchmark",
    )
    parser.add_argument("--d-k", type=int, default=64, help="Key dimension")
    parser.add_argument("--d-v", type=int, default=64, help="Value dimension")
    parser.add_argument("--profile", action="store_true", help="Run torch.profiler")
    parser.add_argument("--benchmark", action="store_true", help="Run micro-benchmark")
    parser.add_argument("--all", action="store_true", help="Run all profiling modes")
    args = parser.parse_args()

    do_profile = args.profile or args.all
    do_benchmark = args.benchmark or args.all

    if not do_profile and not do_benchmark:
        do_benchmark = True  # Default

    print("=" * 70)
    print("Chapter 03: Attention Profiling")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    print(
        f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"
    )
    print("=" * 70)

    results = []

    for N in args.seq_len:
        if do_profile:
            run_torch_profiler(N, args.d_k, args.d_v)

        if do_benchmark:
            r = run_micro_benchmark(N, args.d_k, args.d_v)
            results.append(r)

    # Summary
    if results:
        print("\n" + "=" * 70)
        print("Summary:")
        print(
            f"{'N':>8s} {'Naive(ms)':>12s} {'Fused(ms)':>12s} {'Speedup':>10s} {'TFLOPS(naive)':>15s} {'TFLOPS(fused)':>15s}"
        )
        print("-" * 70)
        for r in results:
            print(
                f"{r['N']:8d} {r['naive_ms']:12.3f} {r['fused_ms']:12.3f} "
                f"{r['speedup']:10.2f}x {r['naive_tflops']:15.2f} {r['fused_tflops']:15.2f}"
            )

    print("\nNext step: Run with Nsight Systems for timeline visualization:")
    print(
        "  nsys profile -o ch03_profile python attention_profile.py --all --seq-len 1024 2048"
    )


if __name__ == "__main__":
    main()
