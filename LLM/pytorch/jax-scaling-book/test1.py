"""Roofline analysis: arithmetic intensity & memory-bound vs compute-bound.

Companion script for jax-scaling-book/01-roofline.md.
Demonstrates the roofline model using PyTorch to measure actual
GPU/CPU bandwidth and compute limits, then plots the roofline curve.

Key concepts:
  - Arithmetic intensity = FLOPs / bytes_transferred
  - Memory-bound: when arithmetic intensity < ridge_point
  - Compute-bound: when arithmetic intensity > ridge_point
  - Roofline ridge point = peak_FLOPS / peak_bandwidth

Run:
    python test1.py                # full demo
    python test1.py theory         # theoretical roofline (from .md)
    python test1.py benchmark      # measure actual HW limits
    python test1.py matmul         # matmul roofline across batch sizes
"""

import sys
import time

import torch


# ============ 1. Theoretical roofline (from 01-roofline.md) ============
def exp_theory():
    print("=" * 60)
    print("1. Theoretical Roofline Analysis")
    print("=" * 60)

    # TPU v5e specs (from notes)
    HBM_BW = 8.2e11  # bytes/s
    BF16_PEAK = 1.97e14  # FLOPs/s
    INT8_PEAK = 3.94e14  # OPs/s

    ridge_bf16 = BF16_PEAK / HBM_BW  # FLOPs per byte -> compute-bound threshold
    ridge_int8 = INT8_PEAK / HBM_BW

    print(f"  TPU v5e specs:")
    print(f"    HBM bandwidth:    {HBM_BW / 1e12:.1f} TB/s")
    print(f"    BF16 peak:        {BF16_PEAK / 1e12:.1f} TFLOPS")
    print(f"    INT8 peak:        {INT8_PEAK / 1e12:.1f} TOPS")
    print(f"    Roofline ridge (BF16): {ridge_bf16:.0f} FLOPs/byte")
    print(f"    Roofline ridge (INT8): {ridge_int8:.0f} FLOPs/byte")

    # Analyze INT8 matmul X[B,D] @ Y[D,F] -> Z[B,F]
    B, D, F = 128, 4096, 4096

    flops = 2 * B * D * F
    bytes_total = B * D + D * F + B * F  # INT8 = 1 byte/element
    ai = flops / bytes_total  # arithmetic intensity

    t_math = flops / INT8_PEAK
    t_comm = bytes_total / HBM_BW

    print(f"\n  INT8 matmul [{B}, {D}] @ [{D}, {F}]:")
    print(f"    FLOPs:        {flops / 1e9:.2f} GFLOPs")
    print(f"    bytes:        {bytes_total / 1e9:.2f} GB")
    print(f"    arith intensity: {ai:.1f} FLOPs/byte")
    print(f"    T_math:       {t_math * 1e6:.0f} us")
    print(f"    T_comm:       {t_comm * 1e6:.0f} us")
    print(f"    lower bound:  {max(t_math, t_comm) * 1e6:.0f} us")
    print(f"    upper bound:  {(t_math + t_comm) * 1e6:.0f} us")
    print(f"    regime:       {'COMPUTE-bound' if ai > ridge_int8 else 'MEMORY-bound'}")

    # The batch-size threshold from the notes
    # AI ≈ 2BDF/DF = 2B (for large D, F)
    # 2B > ridge_int8 -> B > ridge_int8 / 2
    B_threshold = ridge_int8 / 2
    print(f"\n  Compute-bound threshold: B > {B_threshold:.0f}")
    print()


# ============ 2. Benchmark real hardware ============
def exp_benchmark():
    print("=" * 60)
    print("2. Benchmark real GPU bandwidth & matmul speed")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  [SKIP] No CUDA device available")
        return

    device = "cuda"
    props = torch.cuda.get_device_properties(0)
    print(f"  Device: {props.name}")
    print(f"  Memory: {props.total_mem / 1e9:.1f} GB")

    # --- Memory bandwidth measurement ---
    N = 256 * 1024 * 1024 // 4  # 256M elements
    x = torch.randn(N, device=device, dtype=torch.float32)
    y = torch.randn(N, device=device, dtype=torch.float32)
    torch.cuda.synchronize()

    # Warmup
    for _ in range(10):
        y.copy_(x)
    torch.cuda.synchronize()

    # Timed
    n_iter = 100
    t0 = time.perf_counter()
    for _ in range(n_iter):
        y.copy_(x)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    bw = (N * 4 * 2 * n_iter) / (t1 - t0) / 1e9  # GB/s (read + write)
    print(
        f"\n  Measured memory bandwidth: {bw:.1f} GB/s (theoretical: ~{props.memory_clock / 1e6:.0f} MHz)"
    )

    # --- Matmul peak measurement ---
    M, N_, K = 4096, 4096, 4096
    a = torch.randn(M, K, device=device, dtype=torch.float16)
    b = torch.randn(K, N_, device=device, dtype=torch.float16)

    # Warmup
    for _ in range(10):
        torch.mm(a, b)
    torch.cuda.synchronize()

    n_iter = 50
    t0 = time.perf_counter()
    for _ in range(n_iter):
        torch.mm(a, b)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    flops_per_mm = 2 * M * N_ * K
    tflops = (flops_per_mm * n_iter) / (t1 - t0) / 1e12
    print(f"  Measured matmul TFLOPS: {tflops:.1f} (fp16)")

    # Compute roofline ridge
    ridge = tflops * 1e12 / (bw * 1e9)
    print(f"  Empirical roofline ridge: {ridge:.0f} FLOPs/byte")
    print()


# ============ 3. Matmul roofline across batch sizes ============
def exp_matmul_roofline():
    print("=" * 60)
    print("3. Matmul roofline across batch sizes")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  [SKIP] No CUDA device available")
        return

    device = "cuda"

    # Estimate peak values from a large matmul
    M, K, N_ = 4096, 4096, 4096
    a = torch.randn(M, K, device=device, dtype=torch.float16)
    b = torch.randn(K, N_, device=device, dtype=torch.float16)
    for _ in range(5):
        torch.mm(a, b)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        torch.mm(a, b)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    peak_flops = (2 * M * K * N_ * 20) / (t1 - t0)

    # Estimate peak bandwidth
    N_bw = 128 * 1024 * 1024 // 4
    x = torch.randn(N_bw, device=device)
    y = torch.randn(N_bw, device=device)
    for _ in range(10):
        y.copy_(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        y.copy_(x)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    peak_bw = (N_bw * 4 * 2 * 100) / (t1 - t0)

    ridge = peak_flops / peak_bw

    print(f"  Peak FLOPs:    {peak_flops / 1e12:.1f} TFLOPS")
    print(f"  Peak BW:       {peak_bw / 1e9:.1f} GB/s")
    print(f"  Ridge point:   {ridge:.0f} FLOPs/byte")
    print(f"\n  Testing matmul X[B, 4096] @ [4096, 4096] for various B:")

    K_val, N_val = 4096, 4096
    print(f"  {'B':>6s}  {'GFLOPS':>8s}  {'AI':>6s}  {'regime':>12s}")
    print(f"  {'-' * 6}  {'-' * 8}  {'-' * 6}  {'-' * 12}")

    for B in [1, 4, 16, 64, 128, 256, 512, 1024]:
        a = torch.randn(B, K_val, device=device, dtype=torch.float16)
        b = torch.randn(K_val, N_val, device=device, dtype=torch.float16)

        for _ in range(5):
            torch.mm(a, b)
        torch.cuda.synchronize()

        n_iter = max(10, 200 // B)
        t0 = time.perf_counter()
        for _ in range(n_iter):
            torch.mm(a, b)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        gflops = (2 * B * K_val * N_val * n_iter) / (t1 - t0) / 1e9
        bytes_ = (B * K_val * 2 + K_val * N_val * 2 + B * N_val * 2) * n_iter
        ai = (2 * B * K_val * N_val) / (
            B * K_val * 2 + K_val * N_val * 2 + B * N_val * 2
        )
        regime = "COMPUTE" if ai > ridge else "MEMORY"

        print(f"  {B:>6d}  {gflops:>8.1f}  {ai:>6.1f}  {regime:>12s}")

    B_threshold = int(ridge / 2) if ridge > 0 else 0
    print(f"\n  -> Compute-bound threshold: B > ~{B_threshold}")
    print("     (above this batch size, matmul is compute-bound, not memory-bound)")
    print()


EXPERIMENTS = {
    "theory": exp_theory,
    "benchmark": exp_benchmark,
    "matmul": exp_matmul_roofline,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for exp in exps:
        if exp not in EXPERIMENTS:
            print(f"unknown exp '{exp}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[exp]()

    print("[roofline demo] DONE")


if __name__ == "__main__":
    main()
