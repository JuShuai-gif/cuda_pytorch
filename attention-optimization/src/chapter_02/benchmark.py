"""
Benchmark for Chapter 02: GPU vs CPU Attention.

Compares:
1. C++ CPU naive attention (Chapter 01)
2. CUDA naive attention (Chapter 02, global memory)
3. CUDA naive attention (Chapter 02, shared memory)
4. PyTorch scaled_dot_product_attention

Also computes arithmetic intensity and plots roofline.
"""

import argparse
import csv
import math
import subprocess
import sys
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


# A100 specifications
A100_PEAK_FP16_TFLOPS = 312.0
A100_HBM_BW_TBPS = 2.0  # TB/s


def run_cuda_binary(binary_path: str, args: list = None) -> str:
    """Run a compiled CUDA binary and capture stdout."""
    try:
        result = subprocess.run(
            [binary_path] + (args or []), capture_output=True, text=True, timeout=120
        )
        return result.stdout
    except FileNotFoundError:
        return ""
    except subprocess.TimeoutExpired:
        return "TIMEOUT"


def bench_torch_sdpa(Q, K, V, warmup=10, iters=100):
    """Benchmark PyTorch's fused scaled_dot_product_attention."""
    for _ in range(warmup):
        _ = F.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        _ = F.scaled_dot_product_attention(Q, K, V)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters


def compute_roofline_bound(ai: float, peak_tflops: float, peak_bw_tbps: float) -> float:
    """Compute the roofline performance bound for a given arithmetic intensity."""
    compute_bound = peak_tflops  # TFLOPS
    memory_bound = ai * peak_bw_tbps * 1000.0  # AI * BW(TB/s) * 1000
    return min(compute_bound, memory_bound)


def plot_roofline(results: list):
    """Plot roofline model with measured points."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Roofline curve
    ai_range = np.logspace(-1, 4, 200)  # 0.1 to 10000
    perf = [
        compute_roofline_bound(ai, A100_PEAK_FP16_TFLOPS, A100_HBM_BW_TBPS)
        for ai in ai_range
    ]
    ax.loglog(ai_range, perf, "k-", linewidth=2, label=f"A100 Roofline")

    # Mark ridge point
    ridge_ai = A100_PEAK_FP16_TFLOPS / (A100_HBM_BW_TBPS * 1000)
    ax.axvline(
        ridge_ai,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"Ridge point: AI={ridge_ai:.1f}",
    )

    # Plot measured points
    colors = ["red", "blue", "green", "orange"]
    for i, r in enumerate(results):
        ax.scatter(
            r["ai"],
            r["gflops"],
            color=colors[i % len(colors)],
            s=80,
            label=f"N={r['seq_len']} ({r['label']})",
        )

    ax.set_xlabel("Arithmetic Intensity (FLOPs/Byte)")
    ax.set_ylabel("Performance (GFLOPS)")
    ax.set_title("Roofline Analysis: Naive Attention on A100")
    ax.legend()
    ax.grid(True, alpha=0.3)

    png_path = OUT_DIR / "ch02_roofline.png"
    fig.savefig(png_path, dpi=150)
    print(f"Roofline plot saved to {png_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda_binary",
        default="./build/chapter_02/naive_attention_gpu",
        help="Path to compiled CUDA binary",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    d_k, d_v = 64, 64
    results = []

    # Run CUDA binary once to capture output
    cuda_output = run_cuda_binary(args.cuda_binary)
    print("CUDA Binary Output:")
    print(cuda_output)
    print()

    for N in [64, 128, 256, 512, 1024, 2048]:
        Q = torch.randn(1, 1, N, d_k, device=device, dtype=dtype)
        K = torch.randn(1, 1, N, d_k, device=device, dtype=dtype)
        V = torch.randn(1, 1, N, d_v, device=device, dtype=dtype)

        # FLOPs
        flops = 2 * N * N * d_k + 4 * N * N + 2 * N * N * d_v
        # Bytes (approximate, ignoring Q, K, V which are small)
        bytes_moved = (N * d_k + N * d_k + N * d_v + 2 * N * N + N * d_v) * 2  # FP16
        ai = flops / bytes_moved if bytes_moved > 0 else 0

        # PyTorch SDPA
        try:
            sdpa_ms = bench_torch_sdpa(Q, K, V)
            gflops_sdpa = (flops / 1e9) / (sdpa_ms / 1000)

            results.append(
                {
                    "seq_len": N,
                    "label": "torch.SDPA",
                    "flops": flops,
                    "bytes": bytes_moved,
                    "ai": ai,
                    "time_ms": sdpa_ms,
                    "gflops": gflops_sdpa,
                }
            )
            print(
                f"N={N:4d} | SDPA: {sdpa_ms:8.3f}ms | {gflops_sdpa:8.1f} GFLOPS | AI={ai:.1f}"
            )
        except RuntimeError as e:
            print(f"N={N:4d} | SDPA: OOM")

    # Plot
    if results:
        plot_roofline(results)

    # Summary
    print("\n" + "=" * 60)
    print("Key Insights from Chapter 02:")
    print(f"  Ridge point (A100): AI = {ridge_ai:.1f} FLOPs/Byte")
    print(f"  Naive attention AI ≈ d_k/2 = {d_k / 2:.0f} (memory-bound!)")
    print("  → Bottleneck is HBM bandwidth, not compute")
    print("  → Solution: Reduce HBM traffic via tiling + fusion")
    print("=" * 60)


if __name__ == "__main__":
    main()
