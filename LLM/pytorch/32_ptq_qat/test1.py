"""PTQ & QAT quantization demo: prepare, calibrate, convert, fuse.

Companion script for ptq_qat/ptq_qat.md. Covers:
  1. PTQ pipeline:      prepare → calibrate → convert
  2. QAT pipeline:      prepare_qat → train → convert
  3. accuracy compare:  float vs PTQ vs QAT error
  4. module fusion:     fuse conv+bn+relu before quant
  5. scale/zero_point:  inspect quantization parameters
  6. observer:          MinMax vs MovingAverage vs Histogram

Note: This uses the older eager-mode API (torch.ao.quantization)
since FX graph mode requires more setup. The concepts are identical.

Run:
    python test1.py                # full demo
    python test1.py ptq            # PTQ pipeline
    python test1.py qat            # QAT pipeline (train + convert)
    python test1.py accuracy       # float vs PTQ vs QAT error compare
    python test1.py observe        # observer statistics
    python test1.py fuse           # module fusion
    python test1.py params         # scale/zero_point inspection

=== DEBUG 常见问题 ===
  Q: prepare() 后 model 输出变了?
  A: Observer 在 forward 时只观察不修改, 但 collect statistics;
     用 model_prepared.apply(disable_observer) 关闭观察

  Q: convert() 报 "Unsupported qconfig"?
  A: 检查后端: fbgemm (x86 server), qnnpack (ARM/mobile);
     x86 CPU 用 get_default_qconfig("fbgemm")

  Q: QAT 后精度不如 PTQ?
  A: (1) lr 过大导致 scale 不稳定 (2) observer 在 eval 时仍更新
     (3) calibration 数据不够代表性 (4) 首层/末层保持 fp32

  Q: 量化后精度下降严重?
  A: 逐步排查: per-channel 量化 → 融合 conv+bn → 敏感层保持 fp32
"""

import sys
import copy

import torch
import torch.nn as nn
import torch.ao.quantization as quant


# ============ 1. PTQ pipeline ============
def exp_ptq():
    print("=" * 60)
    print("1. PTQ pipeline: prepare → calibrate → convert")
    print("=" * 60)

    class TinyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.quant = quant.QuantStub()
            self.conv = nn.Conv2d(3, 8, 3, bias=False)
            self.relu = nn.ReLU()
            self.dequant = quant.DeQuantStub()

        def forward(self, x):
            x = self.quant(x)
            x = self.conv(x)
            x = self.relu(x)
            x = self.dequant(x)
            return x

    model = TinyCNN().eval()

    # Set qconfig
    model.qconfig = quant.get_default_qconfig("fbgemm")

    # Step 1: prepare (insert observers)
    model_prepared = quant.prepare(model, inplace=False)

    # Step 2: calibrate (observer collects min/max)
    print(f"  Before calibration:")
    act_obs = model_prepared.conv.activation_post_process
    print(f"    activation observer: min={act_obs.min_val}, max={act_obs.max_val}")

    for _ in range(10):
        x = torch.randn(4, 3, 16, 16) * 0.5 + 0.5  # range ~[0, 1]
        model_prepared(x)

    act_obs = model_prepared.conv.activation_post_process
    print(f"  After calibration:")
    print(
        f"    activation observer: min={act_obs.min_val:.3f}, max={act_obs.max_val:.3f}"
    )

    # Step 3: convert (replace with quantized modules)
    model_quantized = quant.convert(model_prepared, inplace=False)

    print(f"\n  Original model:")
    print(f"    {model}")

    # Compare weights
    for (name_f, param_f), (name_q, m_q) in zip(
        model.named_parameters(), model_quantized.named_buffers()
    ):
        if "weight" in name_f:
            print(f"    {name_f}: float32, shape={list(param_f.shape)}")
        if hasattr(m_q, "dtype"):
            print(f"    {name_q}: {m_q.dtype}, shape={list(m_q.shape)}")

    # Verify output
    x = torch.randn(4, 3, 16, 16)
    with torch.no_grad():
        y_float = model(x)
        y_int8 = model_quantized(x)

    diff = (y_float - y_int8).abs()
    print(f"\n  Output comparison:")
    print(f"    max abs diff: {diff.max().item():.4f}")
    print(f"    mean abs diff: {diff.mean().item():.4f}")
    print()


# ============ 2. Observer types ============
def exp_observe():
    print("=" * 60)
    print("2. Observer statistics: MinMax vs MovingAverage")
    print("=" * 60)

    x = torch.randn(100, 16)

    # MinMaxObserver: tracks overall min/max
    obs_minmax = (
        quant.MinMaxObserver().cuda()
        if torch.cuda.is_available()
        else quant.MinMaxObserver()
    )

    # MovingAverageMinMaxObserver: EMA of min/max
    obs_mov = quant.MovingAverageMinMaxObserver(averaging_constant=0.01)

    for i in range(50):
        batch = x[i * 2 : (i + 1) * 2]  # feed 2 samples at a time
        obs_minmax(batch)
        obs_mov(batch)

    print(
        f"  MinMaxObserver:            min={obs_minmax.min_val:.3f} max={obs_minmax.max_val:.3f}"
    )
    print(
        f"  MovingAverageObserver:     min={obs_mov.min_val:.3f} max={obs_mov.max_val:.3f}"
    )
    print(
        f"  Actual data range:         min={x.min().item():.3f} max={x.max().item():.3f}"
    )

    # Compute scale + zero_point from observer statistics
    qmin, qmax = 0, 255  # quint8 asymmetric

    def compute_qparams(observer):
        min_val, max_val = observer.min_val.cpu(), observer.max_val.cpu()
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - round(min_val / scale)
        zero_point = int(max(qmin, min(qmax, zero_point)))
        return scale.item(), zero_point

    s1, z1 = compute_qparams(obs_minmax)
    s2, z2 = compute_qparams(obs_mov)
    print(f"\n  MinMax scale/zp:      scale={s1:.4f}, zero_point={z1}")
    print(f"  MovingAvg scale/zp:   scale={s2:.4f}, zero_point={z2}")

    # Quantize-dequantize roundtrip
    x_sample = torch.tensor([0.5, -0.3, 1.2, -1.5])
    x_q = torch.quantize_per_tensor(x_sample, s1, z1, torch.quint8)
    x_dq = x_q.dequantize()
    print(f"\n  Roundtrip example:")
    print(f"    original: {x_sample}")
    print(f"    dequant:  {x_dq}")
    print(f"    error:    {(x_sample - x_dq).abs()}")
    print()


# ============ 3. Module fusion ============
def exp_fuse():
    print("=" * 60)
    print("3. Module fusion: conv+bn+relu → single fused module")
    print("=" * 60)

    class ConvBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3)
            self.bn = nn.BatchNorm2d(8)
            self.relu = nn.ReLU(inplace=False)

        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    model = ConvBlock().eval()

    # Before fusion: 3 separate modules
    print(f"  Before fusion:")
    for name, _ in model.named_modules():
        if name:
            print(f"    {name}")

    # Set running stats for BN
    with torch.no_grad():
        for _ in range(5):
            model(torch.randn(2, 3, 8, 8))

    # Get reference output
    x = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        y_before = model(x)

    # Fuse
    fused_model = quant.fuse_modules(model, ["conv", "bn", "relu"])

    print(f"\n  After fusion (conv+bn+relu):")
    for name, _ in fused_model.named_modules():
        if name:
            print(f"    {name}")

    # Verify: fused output ≈ original output
    with torch.no_grad():
        y_after = fused_model(x)

    diff = (y_before - y_after).abs().max().item()
    print(f"\n  Max abs diff after fusion: {diff:.2e}")
    print(f"  Folding correct: {'YES' if diff < 1e-5 else 'NO'}")

    # Check param counts
    before_params = sum(p.numel() for p in model.parameters())
    after_params = sum(p.numel() for p in fused_model.parameters())
    print(f"  Params before:  {before_params}")
    print(f"  Params after:   {after_params} (BN params gone)")
    print()


# ============ 4. Quantization parameters ============
def exp_params():
    print("=" * 60)
    print("4. Quantization parameters: scale & zero_point")
    print("=" * 60)

    # Manual torch.quantize_per_tensor
    x = torch.tensor([-1.0, 0.0, 1.0, 2.0, 3.5])

    # Symmetric INT8 quantization
    scale = x.abs().max() / 127
    x_q_sym = torch.quantize_per_tensor(x, scale, 0, torch.qint8)
    print(f"  Symmetric INT8 (scale={scale:.4f}, zp=0):")
    print(f"    quantized:  {x_q_sym.int_repr()}")
    print(f"    dequanted:  {x_q_sym.dequantize()}")
    print(f"    error:      {(x - x_q_sym.dequantize()).abs()}")

    # Asymmetric quint8 quantization
    qmin, qmax = 0, 255
    r_min, r_max = x.min().item(), x.max().item()
    scale_asym = (r_max - r_min) / (qmax - qmin)
    zp_asym = qmin - round(r_min / scale_asym)
    x_q_asym = torch.quantize_per_tensor(x, scale_asym, zp_asym, torch.quint8)
    print(f"\n  Asymmetric quint8 (scale={scale_asym:.4f}, zp={zp_asym}):")
    print(f"    quantized:  {x_q_asym.int_repr()}")
    print(f"    dequanted:  {x_q_asym.dequantize()}")
    print(f"    error:      {(x - x_q_asym.dequantize()).abs()}")

    # Per-channel quantization
    W = torch.randn(4, 8) * torch.tensor([1.0, 5.0, 0.2, 10.0]).unsqueeze(1)
    W_q_per_tensor = torch.quantize_per_tensor(W, W.abs().max() / 127, 0, torch.qint8)
    W_q_per_ch = torch.quantize_per_channel(
        W,
        W.abs().max(dim=1).values / 127,
        torch.zeros(4, dtype=torch.int64),
        axis=0,
        dtype=torch.qint8,
    )

    err_tensor = (W - W_q_per_tensor.dequantize()).norm()
    err_ch = (W - W_q_per_ch.dequantize()).norm()
    print(f"\n  Per-tensor vs per-channel:")
    print(f"    per-tensor error (norm): {err_tensor:.4f}")
    print(f"    per-channel error (norm): {err_ch:.4f}")
    print(
        f"    improvement:              {err_tensor / err_ch:.1f}x better with per-channel"
    )
    print("  -> per-channel quantization handles varying channel magnitudes")
    print()


# ============ 5. QAT pipeline ============
def exp_qat():
    print("=" * 60)
    print("5. QAT pipeline: prepare_qat → train → convert")
    print("=" * 60)

    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.quant = quant.QuantStub()
            self.fc1 = nn.Linear(8, 4)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(4, 2)
            self.dequant = quant.DeQuantStub()

        def forward(self, x):
            x = self.quant(x)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.dequant(x)
            return x

    torch.manual_seed(42)
    x_train = torch.randn(64, 8)
    y_train = torch.randn(64, 2)

    # QAT model
    model_qat = TinyNet()
    model_qat.qconfig = quant.get_default_qat_qconfig("fbgemm")
    model_qat_prepared = quant.prepare_qat(model_qat, inplace=False)

    # Train briefly (QAT learns to compensate quantization error)
    opt = torch.optim.SGD(model_qat_prepared.parameters(), lr=0.01)
    for _ in range(50):
        opt.zero_grad()
        out = model_qat_prepared(x_train)
        loss = torch.nn.functional.mse_loss(out, y_train)
        loss.backward()
        opt.step()

    model_qat_prepared.eval()
    model_qat_int8 = quant.convert(model_qat_prepared, inplace=False)
    print(f"  QAT training: 50 steps, final loss={loss.item():.4f}")

    # Compare: float model with same initial weights
    model_float = TinyNet()
    model_float.load_state_dict(
        {
            k: v
            for k, v in model_qat.state_dict().items()
            if k in model_float.state_dict()
        }
    )
    model_float.eval()

    # PTQ model (no training)
    model_ptq = TinyNet()
    model_ptq.load_state_dict(model_qat.state_dict())
    model_ptq.qconfig = quant.get_default_qconfig("fbgemm")
    model_ptq.eval()
    model_ptq_prepared = quant.prepare(model_ptq, inplace=False)
    for _ in range(3):
        model_ptq_prepared(x_train)
    model_ptq_int8 = quant.convert(model_ptq_prepared, inplace=False)

    with torch.no_grad():
        y_float = model_float(x_train)
        y_ptq = model_ptq_int8(x_train)
        y_qat = model_qat_int8(x_train)

    err_ptq = (y_float - y_ptq).norm() / y_float.norm()
    err_qat = (y_float - y_qat).norm() / y_float.norm()
    print(f"  PTQ relative error: {err_ptq:.6f}")
    print(f"  QAT relative error: {err_qat:.6f}")
    print(
        f"  QAT/PTQ ratio:      {err_qat / err_ptq:.2f}x (QAT better)"
        if err_qat < err_ptq
        else f"  (QAT needs more training to beat PTQ)"
    )
    print("  -> QAT trains with FakeQuantize to learn quantization error compensation")
    print()


# ============ 6. Accuracy comparison ============
def exp_accuracy():
    print("=" * 60)
    print("6. Float vs PTQ vs QAT accuracy (toy regression)")
    print("=" * 60)

    torch.manual_seed(123)
    X = torch.linspace(-3, 3, 120).unsqueeze(1)
    Y = X.pow(3) * 0.5 + X * 0.3 + torch.randn(120, 1) * 0.5

    class Regressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.quant = quant.QuantStub()
            self.net = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
            )
            self.dequant = quant.DeQuantStub()

        def forward(self, x):
            x = self.quant(x)
            x = self.net(x)
            x = self.dequant(x)
            return x

    # Float baseline
    model_f = Regressor()
    opt_f = torch.optim.Adam(model_f.parameters(), lr=0.01)
    for _ in range(200):
        opt_f.zero_grad()
        loss = torch.nn.functional.mse_loss(model_f(X), Y)
        loss.backward()
        opt_f.step()
    model_f.eval()
    with torch.no_grad():
        err_f = torch.nn.functional.mse_loss(model_f(X), Y).item()
    print(f"  Float baseline MSE: {err_f:.6f}")

    # PTQ
    model_p = Regressor()
    model_p.load_state_dict(
        {k: v for k, v in model_f.state_dict().items() if k in model_p.state_dict()}
    )
    model_p.qconfig = quant.get_default_qconfig("fbgemm")
    model_p.eval()
    model_p = quant.prepare(model_p, inplace=False)
    for _ in range(5):
        model_p(X)
    model_p_int8 = quant.convert(model_p, inplace=False)
    with torch.no_grad():
        err_p = torch.nn.functional.mse_loss(model_p_int8(X), Y).item()
    print(
        f"  PTQ int8     MSE: {err_p:.6f}  (+{(err_p - err_f) / err_f * 100:.1f}% vs float)"
    )

    # QAT
    model_q = Regressor()
    model_q.load_state_dict(model_f.state_dict())
    model_q.qconfig = quant.get_default_qat_qconfig("fbgemm")
    model_q = quant.prepare_qat(model_q, inplace=False)
    opt_q = torch.optim.SGD(model_q.parameters(), lr=0.001)
    for _ in range(100):
        opt_q.zero_grad()
        loss = torch.nn.functional.mse_loss(model_q(X), Y)
        loss.backward()
        opt_q.step()
    model_q.eval()
    model_q_int8 = quant.convert(model_q, inplace=False)
    with torch.no_grad():
        err_q = torch.nn.functional.mse_loss(model_q_int8(X), Y).item()
    print(
        f"  QAT int8     MSE: {err_q:.6f}  (+{(err_q - err_f) / err_f * 100:.1f}% vs float)"
    )

    print(f"\n  Summary: float={err_f:.4f}  PTQ={err_p:.4f}  QAT={err_q:.4f}")
    print()


EXPERIMENTS = {
    "ptq": exp_ptq,
    "qat": exp_qat,
    "accuracy": exp_accuracy,
    "observe": exp_observe,
    "fuse": exp_fuse,
    "params": exp_params,
}


def main():
    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for name in exps:
        if name not in EXPERIMENTS:
            print(f"unknown exp '{name}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[name]()

    print("[ptq_qat demo] DONE")


if __name__ == "__main__":
    main()
