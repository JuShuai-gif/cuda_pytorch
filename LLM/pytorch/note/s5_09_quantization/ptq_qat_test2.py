"""PTQ/QAT advanced: per-channel, backend config, custom qconfig.

Companion script for ptq_qat/ptq_qat.md.
  1. per-channel vs per-tensor:     accuracy comparison
  2. custom qconfig:               different configs per layer
  3. backend config:               fbgemm vs qnnpack
  4. HistogramObserver:            better calibration than MinMax
  5. weight-only quantization:     dynamic quantization for LLM

Run:
    python test2.py                 # full demo
    python test2.py per_channel     # per-channel vs per-tensor
    python test2.py custom          # custom qconfig per layer
    python test2.py backend         # backend comparison
    python test2.py histogram       # HistogramObserver
    python test2.py weight_only     # weight-only dynamic quant
"""

import sys
import torch
import torch.nn as nn
import torch.ao.quantization as quant


# ============ 1. Per-channel vs per-tensor ============
def exp_per_channel():
    print("=" * 60)
    print("1. Per-channel vs per-tensor quantization")
    print("=" * 60)

    torch.manual_seed(42)
    W = torch.randn(16, 64) * torch.arange(1, 17, dtype=torch.float32).unsqueeze(1)

    # Per-tensor
    scale_tensor = W.abs().max() / 127
    W_q_tensor = torch.quantize_per_tensor(W, scale_tensor, 0, torch.qint8)
    err_tensor = (W - W_q_tensor.dequantize()).norm()

    # Per-channel (axis=0 -> per output channel)
    scales_ch = W.abs().max(dim=1).values / 127
    zp_ch = torch.zeros(16, dtype=torch.int64)
    W_q_ch = torch.quantize_per_channel(W, scales_ch, zp_ch, axis=0, dtype=torch.qint8)
    err_ch = (W - W_q_ch.dequantize()).norm()

    print(f"  Per-tensor error (norm): {err_tensor:.4f}")
    print(f"  Per-channel error (norm): {err_ch:.4f}")
    print(f"  Improvement: {err_tensor / err_ch:.1f}x")

    # Show why: channels have different ranges
    channel_ranges = W.abs().max(dim=1).values
    print(
        f"\n  Channel ranges: min={channel_ranges.min():.1f} max={channel_ranges.max():.1f}"
    )
    print(f"  Ratio max/min:  {channel_ranges.max() / channel_ranges.min():.1f}")
    print("  -> Large range variation = per-channel much better")
    print()


# ============ 2. Custom qconfig per layer ============
def exp_custom():
    print("=" * 60)
    print("2. Custom qconfig per layer")
    print("=" * 60)

    class CustomNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.quant = quant.QuantStub()
            self.conv1 = nn.Conv2d(3, 8, 3)
            self.conv2 = nn.Conv2d(8, 16, 3)
            self.fc = nn.Linear(16 * 4 * 4, 10)
            self.dequant = quant.DeQuantStub()

        def forward(self, x):
            x = self.quant(x)
            x = self.conv1(x).relu()
            x = self.conv2(x).relu()
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            x = self.dequant(x)
            return x

    model = CustomNet().eval()

    # Per-channel qconfig for conv layers
    per_ch_qconfig = quant.QConfig(
        activation=quant.MovingAverageMinMaxObserver.with_args(
            qscheme=torch.per_tensor_affine
        ),
        weight=quant.PerChannelMinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_channel_symmetric
        ),
    )

    # Default for fc
    default_qconfig = quant.get_default_qconfig("fbgemm")

    # Assign
    model.conv1.qconfig = per_ch_qconfig
    model.conv2.qconfig = per_ch_qconfig
    model.fc.qconfig = default_qconfig

    print("  QConfigs assigned:")
    for name, mod in model.named_modules():
        if hasattr(mod, "qconfig") and mod.qconfig is not None:
            q = mod.qconfig
            print(f"    {name}: activation={type(q.activation).__name__}")

    # Prepare and check
    prepared = quant.prepare(model, inplace=False)
    for _ in range(3):
        prepared(torch.randn(2, 3, 8, 8))

    converted = quant.convert(prepared, inplace=False)
    print(f"\n  Converted successfully: {type(converted).__name__}")
    print("  -> Per-layer qconfig allows fine-grained quantization strategy")
    print()


# ============ 3. Backend config ============
def exp_backend():
    print("=" * 60)
    print("3. Backend: fbgemm vs qnnpack")
    print("=" * 60)

    model = nn.Sequential(
        quant.QuantStub(),
        nn.Conv2d(3, 8, 3),
        nn.ReLU(),
        quant.DeQuantStub(),
    ).eval()

    for backend in ["fbgemm", "qnnpack"]:
        try:
            qcfg = quant.get_default_qconfig(backend)
            m = nn.Sequential(
                quant.QuantStub(), nn.Conv2d(3, 8, 3), nn.ReLU(), quant.DeQuantStub()
            ).eval()
            m.qconfig = qcfg
            prepared = quant.prepare(m, inplace=False)
            for _ in range(3):
                prepared(torch.randn(2, 3, 8, 8))
            converted = quant.convert(prepared, inplace=False)
            print(f"  {backend:10s}: OK — conv type = {type(converted[1]).__name__}")
        except Exception as e:
            print(f"  {backend:10s}: ERROR — {e}")

    print("  -> fbgemm = x86 server (AVX2/VNNI), qnnpack = ARM mobile")
    print()


# ============ 4. HistogramObserver ============
def exp_histogram():
    print("=" * 60)
    print("4. HistogramObserver: better calibration")
    print("=" * 60)

    # Synthetic data with outlier
    torch.manual_seed(42)
    values = torch.cat([torch.randn(1000) * 0.5, torch.tensor([10.0, -8.0])])

    obs_minmax = quant.MinMaxObserver(
        dtype=torch.quint8, qscheme=torch.per_tensor_affine
    )
    obs_hist = quant.HistogramObserver(
        dtype=torch.quint8, qscheme=torch.per_tensor_affine
    )

    for batch in values.split(10):
        obs_minmax(batch)
        obs_hist(batch)

    s1, z1 = obs_minmax.calculate_qparams()
    s2, z2 = obs_hist.calculate_qparams()

    print(f"  MinMax:     scale={s1:.4f}, zp={z1}")
    print(f"  Histogram:  scale={s2:.4f}, zp={z2}")

    # Compare quantization error
    err_minmax = (
        (
            values
            - torch.fake_quantize_per_tensor_affine(values, float(s1), int(z1), 0, 255)
        )
        .abs()
        .mean()
    )
    err_hist = (
        (
            values
            - torch.fake_quantize_per_tensor_affine(values, float(s2), int(z2), 0, 255)
        )
        .abs()
        .mean()
    )

    print(f"\n  MinMax error:    {err_minmax:.4f}")
    print(f"  Histogram error: {err_hist:.4f}")
    print("  -> HistogramObserver better handles outliers (uses distribution)")
    print()


# ============ 5. Weight-only quantization ============
def exp_weight_only():
    print("=" * 60)
    print("5. Weight-only (dynamic) quantization")
    print("=" * 60)

    model = nn.Sequential(
        quant.QuantStub(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        quant.DeQuantStub(),
    ).eval()

    # Weight-only qconfig: activation stays float, weight gets quantized
    w_only_qconfig = quant.QConfig(
        activation=quant.PlaceholderObserver.with_args(dtype=torch.float),
        weight=quant.MinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_tensor_symmetric
        ),
    )

    model.qconfig = w_only_qconfig
    prepared = quant.prepare(model, inplace=False)
    for _ in range(3):
        prepared(torch.randn(4, 64))
    converted = quant.convert(prepared, inplace=False)

    # Weight should be int8, activation stays float
    for name, mod in converted.named_modules():
        if hasattr(mod, "weight"):
            print(
                f"  {name}: weight dtype={mod.weight().dtype if callable(mod.weight) else mod.weight.dtype}"
            )

    # Verify output is still float
    with torch.no_grad():
        y = converted(torch.randn(4, 64))
    print(f"  Output dtype: {y.dtype}")
    print("  -> Weight-only quant reduces model size, activations stay fp32")
    print()


EXPERIMENTS = {
    "per_channel": exp_per_channel,
    "custom": exp_custom,
    "backend": exp_backend,
    "histogram": exp_histogram,
    "weight_only": exp_weight_only,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[ptq_qat test2] DONE")


if __name__ == "__main__":
    main()
