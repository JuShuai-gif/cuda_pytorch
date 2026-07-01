"""PyTorch mixed precision & Conv-BN fusion demo.

Covers three mechanisms from the companion notes:
  1. torch.autocast  - computation mixed precision (weights stay fp32)
  2. model.half()    - storage mixed precision (weights physically become fp16)
  3. Conv-BN fusion  - fold BN into Conv for inference (quantization pre-step)

Run:
    python test1.py              # full demo
    python test1.py autocast     # only autocast section
    python test1.py half         # only half() section
    python test1.py fusion       # only fusion section
"""

import sys

import torch
import torch.nn as nn


# ============ 1. torch.autocast: computation mixed precision ============
def exp_autocast():
    print("=" * 60)
    print("1. torch.autocast: computation mixed precision")
    print("=" * 60)

    m = nn.Linear(512, 256).cuda()
    x = torch.randn(32, 512, device="cuda")

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        y = m(x)

    print(f"  input dtype:      {x.dtype}")  # fp32
    print(f"  weight dtype:     {m.weight.dtype}")  # still fp32
    print(f"  output dtype:     {y.dtype}")  # fp16 (autocast default)
    print(f"  weight storage:   {m.weight.data_ptr():#x}")

    # Verifying: weights are NOT modified
    params_by_dtype = {}
    for name, p in m.named_parameters():
        dt = params_by_dtype.setdefault(p.dtype, [])
        dt.append(name)
    print("  params_by_dtype:  {")
    for dtype, names in params_by_dtype.items():
        print(f"    {dtype}: {names}")
    print("  }")
    print("  -> autocast doesn't change weight storage; only casts inputs on-the-fly")
    print()

    # Show the policy table in action: matmul is fp16, softmax is fp32
    print("  Operator dispatch examples:")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        a = torch.randn(16, 64, device="cuda")
        b = torch.randn(64, 32, device="cuda")
        mm_out = torch.mm(a, b)  # AT_FORALL_LOWER_PRECISION_FP -> fp16
        sm_out = torch.softmax(mm_out, dim=-1)  # AT_FORALL_FP32_SET_OPT_DTYPE -> fp32
        print(f"  torch.mm output dtype:       {mm_out.dtype}")  # fp16
        print(f"  torch.softmax output dtype:  {sm_out.dtype}")  # fp32
    print()


# ============ 2. model.half(): storage mixed precision ============
def exp_half():
    print("=" * 60)
    print("2. model.half(): storage mixed precision")
    print("=" * 60)

    m = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU()).cuda()

    print(f"  BEFORE half():")
    print(f"    Linear.weight dtype:  {m[0].weight.dtype}")
    print(f"    BN.running_mean dtype:{m[1].running_mean.dtype}")
    print(f"    BN.num_batches_tracked dtype: {m[1].num_batches_tracked.dtype}")

    before_mem = sum(p.element_size() * p.numel() for p in m.parameters())
    before_buf = sum(b.element_size() * b.numel() for b in m.buffers())

    m.half()  # _apply(lambda t: t.half() if t.is_floating_point() else t)

    print(f"  AFTER half():")
    print(f"    Linear.weight dtype:  {m[0].weight.dtype}")
    print(f"    BN.running_mean dtype:{m[1].running_mean.dtype}")
    print(f"    BN.num_batches_tracked dtype: {m[1].num_batches_tracked.dtype}")
    print(f"    (num_batches_tracked stays int64 -- guarded by is_floating_point)")

    after_mem = sum(p.element_size() * p.numel() for p in m.parameters())
    after_buf = sum(b.element_size() * b.numel() for b in m.buffers())
    print(
        f"    param memory: {before_mem} -> {after_mem} bytes ({after_mem / before_mem:.1%})"
    )
    print(
        f"    buffer memory: {before_buf} -> {after_buf} bytes ({after_buf / before_buf:.1%})"
    )
    print()

    # autocast vs half() distinction
    print("  autocast vs half() summary:")
    print(
        "    autocast: weights stay fp32, only inputs cast on-the-fly -> no VRAM savings"
    )
    print(
        "    half():   weights permanently become fp16 -> VRAM ~halved, precision lost"
    )
    print()


# ============ 3. Conv-BN fusion: fold BN into Conv for inference ============
def exp_fusion():
    print("=" * 60)
    print("3. Conv-BN fusion: fold BN into Conv for inference")
    print("=" * 60)

    conv = nn.Conv2d(3, 16, 3, bias=True).cuda().eval()
    bn = nn.BatchNorm2d(16).cuda().eval()

    # Set known weights for verification
    conv.weight.data.fill_(0.5)
    conv.bias.data.fill_(0.1)
    bn.weight.data.fill_(2.0)  # gamma
    bn.bias.data.fill_(0.3)  # beta
    bn.running_mean.fill_(1.2)
    bn.running_var.fill_(0.04)
    bn.eps = 1e-5

    x = torch.randn(1, 3, 8, 8, device="cuda")

    with torch.no_grad():
        y_original = bn(conv(x))

    # fuse: W' = W * gamma / sqrt(rv+eps), b' = (b-rm) * gamma / sqrt(rv+eps) + beta
    fused = torch.nn.utils.fuse_conv_bn_eval(conv, bn)

    with torch.no_grad():
        y_fused = fused(x)

    diff = (y_original - y_fused).abs().max().item()
    print(f"  max abs diff after fusion: {diff:.2e}")
    print(f"  output match: {'YES' if diff < 1e-5 else 'NO (floating rounding)'}")

    # Check that BN layer is gone
    print(
        f"  fused module has BN? {any(isinstance(m, nn.BatchNorm2d) for m in fused.modules())}"
    )

    # Mathematically verify weight fusion
    rv = bn.running_var
    rm = bn.running_mean
    eps = bn.eps
    gamma = bn.weight
    beta = bn.bias

    W_expected = conv.weight * (gamma / torch.sqrt(rv + eps)).reshape(-1, 1, 1, 1)
    b_expected = (conv.bias - rm) * gamma / torch.sqrt(rv + eps) + beta

    W_diff = (fused.weight - W_expected).abs().max().item()
    b_diff = (fused.bias - b_expected).abs().max().item()
    print(
        f"  weight fusion match: {'YES' if W_diff < 1e-5 else 'NO'} (max diff={W_diff:.2e})"
    )
    print(
        f"  bias   fusion match: {'YES' if b_diff < 1e-5 else 'NO'} (max diff={b_diff:.2e})"
    )
    print()

    # Show why fusion matters for quantization: fewer ops = fewer quantize/dequantize pairs
    print("  Why fusion matters for quantization:")
    print("    Before fusion: Conv -> quant -> dequant -> BN -> quant -> dequant")
    print("    After  fusion: Conv -> quant -> dequant")
    print("    (BN folded into Conv weights, BN layer disappears at inference)")
    print()


EXPERIMENTS = {
    "autocast": exp_autocast,
    "half": exp_half,
    "fusion": exp_fusion,
}


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, exiting.")
        return

    exps = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS)
    for exp in exps:
        if exp not in EXPERIMENTS:
            print(f"unknown exp '{exp}', choose from: {list(EXPERIMENTS)}")
            continue
        EXPERIMENTS[exp]()

    print("[amp demo] DONE")


if __name__ == "__main__":
    main()
