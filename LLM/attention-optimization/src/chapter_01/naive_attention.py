"""
Naive Scaled Dot-Product Attention implementation in Python.

This implementation follows the exact formula:
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

Each intermediate step is materialized explicitly to demonstrate
the O(N^2) memory bottleneck that FlashAttention solves.
"""

import math
import time
import sys

import torch
import torch.nn.functional as F


def attention_ref(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Reference implementation using PyTorch's fused SDPA."""
    return F.scaled_dot_product_attention(Q, K, V)


def attention_naive_step_by_step(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
) -> tuple[torch.Tensor, dict]:
    """
    Naive attention with each step materialized.

    Tracks memory allocations and wall-clock time per step.

    Returns:
        output: [batch, num_heads, seq_len, d_v]
        stats: dict with per-step timing and peak memory
    """
    stats = {}
    d_k = Q.size(-1)

    # Step 1: Q @ K^T -> S: [B, H, N, N]
    t0 = time.perf_counter()
    S = torch.matmul(Q, K.transpose(-2, -1))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    stats["step1_qk"] = time.perf_counter() - t0
    stats["size_S_MB"] = S.numel() * S.element_size() / (1024 * 1024)

    # Step 2: Scale
    S = S / math.sqrt(d_k)

    # Step 3: Softmax row-wise
    t0 = time.perf_counter()
    P = F.softmax(S, dim=-1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    stats["step2_softmax"] = time.perf_counter() - t0
    stats["size_P_MB"] = P.numel() * P.element_size() / (1024 * 1024)

    # Step 4: P @ V -> O: [B, H, N, d_v]
    t0 = time.perf_counter()
    O = torch.matmul(P, V)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    stats["step3_pv"] = time.perf_counter() - t0

    stats["peak_memory_MB"] = (
        (2 * S.numel() + P.numel()) * S.element_size() / (1024 * 1024)
    )

    return O, stats


def attention_naive(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Single-line naive attention for benchmarking.

    Equivalent to the step-by-step version but measures end-to-end time.
    """
    d_k = Q.size(-1)
    S = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    P = F.softmax(S, dim=-1)
    O = torch.matmul(P, V)
    return O


def report(seq_len: int, d_k: int, stats: dict):
    """Print formatted benchmark results."""
    print(
        f"  seq_len={seq_len:6d}, d_k={d_k:4d} | "
        f"S/P size: {stats['size_S_MB']:8.2f} MB | "
        f"Peak mem: {stats['peak_memory_MB']:8.2f} MB | "
        f"QK^T: {stats['step1_qk'] * 1000:8.3f} ms | "
        f"Softmax: {stats['step2_softmax'] * 1000:8.3f} ms | "
        f"PV: {stats['step3_pv'] * 1000:8.3f} ms"
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Running on: {device}, dtype: {dtype}")
    print("=" * 80)
    print(
        f"{'seq_len':>8s}  {'d_k':>4s}  {'S/P(MB)':>10s}  {'Peak(MB)':>10s}  "
        f"{'QK^T(ms)':>10s}  {'Softmax(ms)':>12s}  {'PV(ms)':>10s}"
    )
    print("-" * 80)

    for seq_len in [256, 512, 1024, 2048, 4096, 8192]:
        try:
            Q = torch.randn(1, 1, seq_len, 64, device=device, dtype=dtype)
            K = torch.randn(1, 1, seq_len, 64, device=device, dtype=dtype)
            V = torch.randn(1, 1, seq_len, 64, device=device, dtype=dtype)

            # Warmup
            for _ in range(5):
                _ = attention_naive(Q, K, V)
            if device == "cuda":
                torch.cuda.synchronize()

            # Benchmark
            t_start = time.perf_counter()
            _, stats = attention_naive_step_by_step(Q, K, V)
            t_end = time.perf_counter()

            print(
                f"{seq_len:8d}  {64:4d}  "
                f"{stats['size_S_MB']:10.2f}  {stats['peak_memory_MB']:10.2f}  "
                f"{stats['step1_qk'] * 1000:10.3f}  {stats['step2_softmax'] * 1000:12.3f}  "
                f"{stats['step3_pv'] * 1000:10.3f}"
            )

            # Correctness check
            ref_out = attention_ref(Q, K, V)
            our_out, _ = attention_naive_step_by_step(Q, K, V)
            max_diff = (ref_out - our_out).abs().max().item()
            assert max_diff < 5e-2, f"Numerical error too large: {max_diff}"

        except RuntimeError as e:
            print(f"{seq_len:8d}  OOM: {str(e)[:60]}")

    print("=" * 80)

    # Theoretical analysis
    print("\n--- Complexity Analysis ---")
    print("Time: O(N^2) - dominated by QK^T and PV matmuls")
    print("Memory: O(N^2) - S and P matrices are the bottleneck")
    print(f"\nFor N=8192, d_k=64 on {device}:")
    n = 8192
    print(f"  S matrix elements: {n} x {n} = {n * n:,}")
    print(f"  FP16 memory for S: {n * n * 2 / (1024 * 1024):.1f} MB")


if __name__ == "__main__":
    main()
