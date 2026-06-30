"""Quantization fundamentals demo: from FP16 to INT8/INT4.

Covers core concepts from quantization/notes.md:
  1. Symmetric quantization   (absmax, zero-centered)
  2. Asymmetric quantization  (with zero_point)
  3. Per-channel vs per-tensor quantization
  4. Memory bandwidth vs compute tradeoff
  5. Outlier detection (identifying hard-to-quantize channels)

Run:
    python test1.py              # full demo
    python test1.py symmetric    # symmetric quant only
    python test1.py asymmetric   # asymmetric quant only
    python test1.py channel      # per-channel vs per-tensor
    python test1.py bandwidth    # bandwidth tradeoff demo
    python test1.py outlier      # outlier detection
"""

import sys

import torch


# ============ 1. Symmetric quantization (absmax) ============
def exp_symmetric():
    print("=" * 60)
    print("1. Symmetric quantization (absmax)")
    print("=" * 60)

    values = torch.tensor([5.47, 8.21, 3.145, 10.94, -2.5, 0.0], dtype=torch.float32)

    # INT8 range: [-127, 127]
    q_min, q_max = -127, 127

    # Scale = max(|values|) / q_max
    abs_max = values.abs().max()
    scale = abs_max / q_max
    print(f"  original:    {values}")
    print(f"  abs_max:     {abs_max:.4f}")
    print(f"  scale:       {scale:.4f}")

    # Quantize: q = round(x / scale), clamp to [q_min, q_max]
    q = torch.clamp(torch.round(values / scale), q_min, q_max).to(torch.int8)
    print(f"  quantized:   {q}")

    # Dequantize: x_hat = q * scale
    x_hat = q.float() * scale
    print(f"  dequantized: {x_hat}")

    # Error
    errors = (values - x_hat).abs()
    print(f"  abs errors:  {errors}")
    print(f"  max error:   {errors.max():.4f}")
    print(f"  mean error:  {errors.mean():.4f}")
    print("  -> zero maps to zero (symmetric), one scale for all values")
    print()


# ============ 2. Asymmetric quantization ============
def exp_asymmetric():
    print("=" * 60)
    print("2. Asymmetric quantization (with zero_point)")
    print("=" * 60)

    values = torch.tensor([5.47, 8.21, 3.145, 10.94, -2.5, 0.0], dtype=torch.float32)

    q_min, q_max = -128, 127

    r_min, r_max = values.min(), values.max()
    scale = (q_max - q_min) / (r_max - r_min)
    zero_point = int(q_min - torch.round(r_min / scale))

    print(f"  original:     {values}")
    print(f"  [r_min, r_max]: [{r_min:.4f}, {r_max:.4f}]")
    print(f"  scale:         {scale:.4f}")
    print(f"  zero_point:    {zero_point}")

    # Quantize
    q = torch.clamp(torch.round(values / scale + zero_point), q_min, q_max).to(
        torch.int8
    )
    print(f"  quantized:    {q}")

    # Dequantize
    x_hat = (q.float() - zero_point) * scale
    print(f"  dequantized:  {x_hat}")

    errors = (values - x_hat).abs()
    print(f"  abs errors:   {errors}")
    print(f"  max error:    {errors.max():.4f}")
    print(f"  mean error:   {errors.mean():.4f}")
    print("  -> full range [-128,127] utilized, but zero not preserved exactly")
    print()

    # Compare with symmetric on same data
    abs_max = values.abs().max()
    scale_sym = abs_max / 127
    q_sym = torch.clamp(torch.round(values / scale_sym), -127, 127).to(torch.int8)
    x_hat_sym = q_sym.float() * scale_sym
    print(f"  symmetric  max error: {(values - x_hat_sym).abs().max():.4f}")
    print(f"  asymmetric max error: {errors.max():.4f}")
    print()


# ============ 3. Per-channel vs per-tensor quantization ============
def exp_channel():
    print("=" * 60)
    print("3. Per-channel vs per-tensor quantization")
    print("=" * 60)

    # Simulate a weight matrix with varied channel ranges
    torch.manual_seed(42)
    W = torch.randn(4, 256, dtype=torch.float32)
    # Make channel 1 have large outliers, channel 3 very small
    W[1] *= 10.0
    W[3] *= 0.1

    print(f"  Weight shape: {W.shape}")
    print(f"  Channel ranges:")
    for c in range(4):
        print(
            f"    ch{c}: [{W[c].min():.2f}, {W[c].max():.2f}]  absmax={W[c].abs().max():.2f}"
        )

    # Per-tensor: single scale for entire matrix
    abs_max_global = W.abs().max()
    scale_global = abs_max_global / 127
    W_q_global = torch.clamp(torch.round(W / scale_global), -127, 127)
    W_hat_global = W_q_global.float() * scale_global
    err_global = (W - W_hat_global).abs()

    print(f"\n  Per-tensor quantization:  scale={scale_global:.4f}")
    print(
        f"    max error: {err_global.max():.4f},  mean error: {err_global.mean():.4f}"
    )
    for c in range(4):
        print(f"    ch{c} error: {err_global[c].mean():.4f}")

    # Per-channel: each row has its own scale
    abs_max_per_ch = W.abs().max(dim=1).values  # [4]
    scales_per_ch = abs_max_per_ch / 127  # [4]
    # Broadcast scales for quantization
    scales_broadcast = scales_per_ch.unsqueeze(1)  # [4, 1]
    W_q_ch = torch.clamp(torch.round(W / scales_broadcast), -127, 127)
    W_hat_ch = W_q_ch.float() * scales_broadcast
    err_ch = (W - W_hat_ch).abs()

    print(f"\n  Per-channel quantization:  scales={scales_per_ch}")
    print(f"    max error: {err_ch.max():.4f},  mean error: {err_ch.mean():.4f}")
    for c in range(4):
        print(f"    ch{c} error: {err_ch[c].mean():.4f}")

    print(
        "\n  -> per-channel preserves small channels that per-tensor would squash to zero"
    )
    print(
        "     LLM.int8() / AWQ use per-channel (or per-group) for exactly this reason"
    )
    print()


# ============ 4. Bandwidth vs compute tradeoff ============
def exp_bandwidth():
    print("=" * 60)
    print("4. Memory bandwidth vs compute tradeoff (A100 example)")
    print("=" * 60)

    # A100 specs
    mem_bw = 2000e9  # 2000 GB/s in bytes/s
    compute = 312e12  # 312 TFLOPS

    N = int(100e9)  # 100B parameters

    # FP16: 2 bytes per param
    data_fp16 = N * 2
    time_load_fp16 = data_fp16 / mem_bw
    time_compute = (N * 2) / compute  # OPS ≈ 2 * N for matmul

    print(f"  Model: {N / 1e9:.0f}B params")
    print(f"\n  FP16 (2 bytes/param):")
    print(f"    data to load:  {data_fp16 / 1e9:.1f} GB")
    print(f"    load time:     {time_load_fp16 * 1000:.2f} ms  <- bottleneck")
    print(f"    compute time:  {time_compute * 1000:.2f} ms")
    print(f"    ratio:         {time_load_fp16 / time_compute:.0f}x (memory-bound)")

    # FP8: 1 byte per param
    data_fp8 = N * 1
    time_load_fp8 = data_fp8 / mem_bw
    print(f"\n  FP8 (1 byte/param):")
    print(f"    data to load:  {data_fp8 / 1e9:.1f} GB")
    print(f"    load time:     {time_load_fp8 * 1000:.2f} ms")
    print(f"    ratio:         {time_load_fp8 / time_compute:.0f}x")
    print(
        f"    speedup:       {time_load_fp16 / time_load_fp8:.1f}x (purely from less data movement)"
    )

    # INT4: 0.5 bytes per param
    data_int4 = N * 0.5
    time_load_int4 = data_int4 / mem_bw
    print(f"\n  INT4 (0.5 bytes/param):")
    print(f"    data to load:  {data_int4 / 1e9:.1f} GB")
    print(f"    load time:     {time_load_int4 * 1000:.2f} ms")
    print(f"    speedup:       {time_load_fp16 / time_load_int4:.1f}x")

    print("\n  -> quantization speeds up inference by reducing data movement")
    print("     (NOT by making arithmetic faster)")
    print()


# ============ 5. Outlier detection ============
def exp_outlier():
    print("=" * 60)
    print("5. Outlier detection: identifying hard-to-quantize channels")
    print("=" * 60)

    torch.manual_seed(123)
    # Simulate activation-like tensor: normal values with some outlier channels
    X = torch.randn(1024, 4096, dtype=torch.float32)
    # Inject outliers in specific channels (like real LLM activations)
    outlier_channels = torch.randint(0, 4096, (20,))
    X[:, outlier_channels] *= 15.0

    print(f"  Activation shape: {X.shape}")

    # Per-channel absmax
    per_ch_absmax = X.abs().max(dim=0).values  # [4096]
    threshold = per_ch_absmax.mean() * 3  # 3x mean as outlier threshold
    outlier_mask = per_ch_absmax > threshold
    outlier_indices = torch.where(outlier_mask)[0]

    print(f"  mean absmax:   {per_ch_absmax.mean():.2f}")
    print(f"  std  absmax:   {per_ch_absmax.std():.2f}")
    print(f"  threshold:     {threshold:.2f}")
    print(
        f"  outlier chs:   {len(outlier_indices)} / {per_ch_absmax.shape[0]} "
        f"({len(outlier_indices) / per_ch_absmax.shape[0] * 100:.1f}%)"
    )

    if len(outlier_indices) > 0:
        print(
            f"  outlier absmax values: {per_ch_absmax[outlier_indices][:5].tolist()}..."
        )

    # Quantization error comparison
    # Per-tensor: single scale for all
    scale_global = X.abs().max() / 127
    X_q_global = (
        torch.clamp(torch.round(X / scale_global), -127, 127).float() * scale_global
    )
    err_global = (X - X_q_global).norm() / X.norm()

    # Per-channel with outlier protection (like LLM.int8() approach):
    # keep outlier channels in FP16, quantize rest
    scale_ch = per_ch_absmax / 127  # [4096]
    scale_ch_bcast = scale_ch.unsqueeze(0)  # [1, 4096]
    X_q_ch = (
        torch.clamp(torch.round(X / scale_ch_bcast), -127, 127).float() * scale_ch_bcast
    )
    err_ch = (X - X_q_ch).norm() / X.norm()

    print(f"\n  Quantization error (normalized):")
    print(f"    per-tensor:  {err_global:.6f}")
    print(f"    per-channel: {err_ch:.6f}")

    # Mixed: outliers in FP16, rest per-channel quantized
    X_mixed = X.clone()
    for ch in outlier_indices:
        # Keep outlier channel in FP16 (skip quantizing it)
        continue
    normal_mask = ~outlier_mask
    normal_cols = X[:, normal_mask]
    scales_normal = per_ch_absmax[normal_mask] / 127
    normal_q = torch.clamp(
        torch.round(normal_cols / scales_normal.unsqueeze(0)), -127, 127
    ).float() * scales_normal.unsqueeze(0)
    X_mixed[:, normal_mask] = normal_q
    err_mixed = (X - X_mixed).norm() / X.norm()

    print(f"    mixed (outliers fp16): {err_mixed:.6f}")

    print("\n  -> Outlier channels dominate per-tensor quantization error")
    print("     LLM.int8() strategy: keep ~1% outlier channels in FP16,")
    print("     quantize the remaining 99% -> near-lossless INT8 inference")
    print()


EXPERIMENTS = {
    "symmetric": exp_symmetric,
    "asymmetric": exp_asymmetric,
    "channel": exp_channel,
    "bandwidth": exp_bandwidth,
    "outlier": exp_outlier,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for exp in exps:
        if exp not in EXPERIMENTS:
            print(f"unknown exp '{exp}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[exp]()

    print("[quantization demo] DONE")


if __name__ == "__main__":
    main()
