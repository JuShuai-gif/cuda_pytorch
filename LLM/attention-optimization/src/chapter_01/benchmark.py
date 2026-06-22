"""
Benchmark script for Chapter 01: Naive Attention.

Compares:
1. Our naive Python implementation
2. PyTorch's reference F.scaled_dot_product_attention
3. Reports latency, throughput, memory usage.

Output: CSV data and matplotlib charts.
"""

import argparse
import csv
import math
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.benchmark as torch_bench

matplotlib.use("Agg")

OUT_DIR = Path(__file__).resolve().parent / "plots"
OUT_DIR.mkdir(exist_ok=True)


def bench_naive(Q, K, V, warmup=10, iters=100):
    """Time our naive implementation."""
    d_k = Q.size(-1)
    # Warmup
    for _ in range(warmup):
        S = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        P = F.softmax(S, dim=-1)
        O = torch.matmul(P, V)

    if Q.device.type == "cuda":
        torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        S = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        P = F.softmax(S, dim=-1)
        O = torch.matmul(P, V)
        end_events[i].record()

    if Q.device.type == "cuda":
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        return np.mean(times), np.std(times)

    # CPU fallback
    t0 = time.perf_counter()
    for _ in range(iters):
        S = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        P = F.softmax(S, dim=-1)
        O = torch.matmul(P, V)
    elapsed = (time.perf_counter() - t0) / iters * 1000
    return elapsed, 0.0


def bench_reference(Q, K, V, warmup=10, iters=100):
    """Time PyTorch's scaled_dot_product_attention."""
    for _ in range(warmup):
        _ = F.scaled_dot_product_attention(Q, K, V)

    if Q.device.type == "cuda":
        torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        _ = F.scaled_dot_product_attention(Q, K, V)
        end_events[i].record()

    if Q.device.type == "cuda":
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        return np.mean(times), np.std(times)

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = F.scaled_dot_product_attention(Q, K, V)
    elapsed = (time.perf_counter() - t0) / iters * 1000
    return elapsed, 0.0


def compute_peak_memory(N: int, d_k: int, dtype_size: int) -> float:
    """Compute peak memory usage for the NxN attention matrix."""
    # S + P = 2 * N^2 elements
    return (2 * N * N * dtype_size) / (1024 * 1024)  # MB


def compute_flops(N: int, d_k: int, d_v: int) -> float:
    """Total FLOPs for one attention forward pass."""
    qk = 2.0 * N * N * d_k
    softmax_flops = 4.0 * N * N
    pv = 2.0 * N * N * d_v
    return qk + softmax_flops + pv


def run_benchmarks(device: str = "cuda", output_csv: str = "results.csv"):
    dtype = torch.float16 if device == "cuda" else torch.float32
    d_k, d_v = 64, 64
    seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192]
    device_obj = torch.device(
        device if torch.cuda.is_available() and device == "cuda" else "cpu"
    )

    results = []

    for N in seq_lens:
        try:
            Q = torch.randn(1, 1, N, d_k, device=device_obj, dtype=dtype)
            K = torch.randn(1, 1, N, d_k, device=device_obj, dtype=dtype)
            V = torch.randn(1, 1, N, d_v, device=device_obj, dtype=dtype)

            naive_mean, naive_std = bench_naive(Q, K, V)
            ref_mean, ref_std = bench_reference(Q, K, V)
            peak_mem = compute_peak_memory(N, d_k, dtype.itemsize)
            flops = compute_flops(N, d_k, d_v)

            # GFLOPS for naive
            gflops = (flops / 1e9) / (naive_mean / 1000) if naive_mean > 0 else 0

            results.append(
                {
                    "seq_len": N,
                    "d_k": d_k,
                    "naive_ms": naive_mean,
                    "naive_std": naive_std,
                    "ref_ms": ref_mean,
                    "ref_std": ref_std,
                    "peak_mem_MB": peak_mem,
                    "gflops": gflops,
                    "flops": flops,
                }
            )
            print(
                f"N={N:5d} | naive: {naive_mean:8.3f}ms | ref: {ref_mean:8.3f}ms | "
                f"peak_mem: {peak_mem:8.1f}MB"
            )

        except RuntimeError as e:
            print(f"N={N:5d} | OOM: {e}")

    # Write CSV
    csv_path = Path(output_csv)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults written to {csv_path}")

    # Plot
    plot_results(results)
    return results


def plot_results(results):
    if not results:
        return

    Ns = [r["seq_len"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Chapter 01: Naive Attention Benchmark", fontsize=14)

    # Latency
    ax = axes[0, 0]
    naive_ms = [r["naive_ms"] for r in results]
    ref_ms = [r["ref_ms"] for r in results]
    ax.plot(Ns, naive_ms, "o-", label="Naive (our impl)")
    ax.plot(Ns, ref_ms, "s-", label="PyTorch SDPA (fused)")
    ax.set_xlabel("Sequence Length N")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency vs Sequence Length")
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Memory
    ax = axes[0, 1]
    mem = [r["peak_mem_MB"] for r in results]
    ax.plot(Ns, mem, "o-", color="red")
    ax.fill_between(Ns, 0, mem, alpha=0.2, color="red")
    ax.set_xlabel("Sequence Length N")
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_title("Peak Memory (O(N^2))")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # O(N^2) verification
    ax = axes[1, 0]
    ax_N2 = [n**2 for n in Ns]
    ax.plot(Ns, naive_ms, "o-", label="Actual latency")
    # Fit quadratic
    coeffs = np.polyfit([n**2 for n in Ns], naive_ms, 1)
    fit = [coeffs[0] * n**2 + coeffs[1] for n in Ns]
    ax.plot(Ns, fit, "--", label=f"O(N^2) fit: {coeffs[0]:.2e}*N²")
    ax.set_xlabel("Sequence Length N")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("O(N^2) Verification")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # GFLOPS
    ax = axes[1, 1]
    gflops_vals = [r["gflops"] for r in results]
    ax.bar(range(len(Ns)), gflops_vals, tick_label=[str(n) for n in Ns])
    ax.set_xlabel("Sequence Length N")
    ax.set_ylabel("GFLOPS")
    ax.set_title("Throughput (GFLOPS)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = OUT_DIR / "ch01_benchmark.png"
    fig.savefig(png_path, dpi=150)
    print(f"Plot saved to {png_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Chapter 01 Attention Benchmark")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output", default="results_ch01.csv")
    args = parser.parse_args()

    run_benchmarks(device=args.device, output_csv=args.output)


if __name__ == "__main__":
    main()
