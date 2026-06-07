# Lecture 06: PTQ Pipeline + QAT with FakeQuantize & STE

## Overview

This code accompanies **MIT 6.5940 Lecture 06: Quantization (Part II)**.
It implements three quantization strategies and compares their accuracy:

1. **PTQ (Post-Training Quantization)** -- calibrate activation ranges on a
   representative dataset, then quantize weights and insert activation
   quantizers in one shot (no re-training).
2. **QAT (Quantization-Aware Training)** -- wrap every Conv2d and Linear
   layer with a `FakeQuantize` module that simulates quantization noise during
   forward passes; use the **Straight-Through Estimator (STE)** so gradients
   flow through the rounding operation unchanged.
3. **Mixed Precision** -- assign different bit widths to different layers
   (e.g., 8-bit for the first and last layers, 4-bit for middle layers) and
   train with QAT.

We use a small CNN (SimpleCNN, ~207K params) on synthetic 28x28 data so the
entire pipeline runs on CPU in under a minute.

## Prerequisites

```bash
pip install torch
```

## Usage

```bash
cd src/lecture-06
python main.py
```

The script runs entirely on CPU and produces:

1. **FP32 baseline** -- train SimpleCNN for 8 epochs, measure test accuracy.
2. **PTQ pipeline** -- calibrate activation ranges (scale+zp per layer),
   quantize all weights, insert activation quantizer hooks, and evaluate.
3. **QAT pipeline** -- wrap layers with `FakeQuantize`, calibrate observers,
   fine-tune for 5 epochs with STE, and evaluate.
4. **Mixed precision simulation** -- per-layer bit-width assignments
   (conv1=8b, conv2=4b, fc1=4b, fc2=8b), calibrate, fine-tune, evaluate.
5. **Comparison table** -- accuracy delta for each method vs the FP32 baseline.
6. **STE gradient sanity check** -- verify that the STE passes gradients
   through the rounding operation as identity.

## Key Functions

| Function | Description |
|----------|-------------|
| `compute_scale_zp(tensor, bits)` | Compute asymmetric affine quantization parameters (scale, zp, qmin, qmax) |
| `fake_quantize(x, scale, zp, qmin, qmax)` | Quantize then dequantize (simulate quantization noise) |
| `fake_quantize_ste(x, scale, zp, qmin, qmax)` | Fake-quantize with Straight-Through Estimator gradient |
| `FakeQuantize(bits)` | Trainable nn.Module with observer mode, scale/zp buffers, and STE |
| `calibrate_activation_ranges(model, loader, bits)` | Forward hooks collect per-layer activation min/max |
| `ptq_quantize_model(model, act_params, bits)` | PTQ: quantize weights in-place + attach activation quantizer hooks |
| `build_qat_model(fp32_model, bits)` | Wrap Conv2d/Linear with QATConv2d/QATLinear containing FakeQuantize |
| `calibrate_qat(model, loader)` | Calibrate QAT FakeQuantize observers then freeze them |
| `build_mixed_precision_model(fp32_model, bit_config)` | QAT model with per-layer custom bit widths |
| `train_one_epoch / evaluate_accuracy` | Standard training/evaluation loops |
| `remove_quant_hooks(model)` | Remove activation quantizer hooks from a PTQ model |

## Concepts

### Straight-Through Estimator (STE)

The quantization rounding operation \u230ar \u231d is non-differentiable (gradient
is zero almost everywhere). The STE replaces the true gradient with the
identity function:

```
Forward:  y = fake_quantize(x)          # quantization noise applied
Backward: dL/dx = dL/dy * 1            # gradient passes through unchanged
```

This is implemented via the `.detach()` trick:

```python
x_fq = fake_quantize(x)
output = x + (x_fq - x).detach()
# Forward: output = x_fq
# Backward: d(output)/dx = 1 (identity)
```

A `torch.autograd.Function` version (`_FakeQuantizeSTE`) is also provided for
educational comparison.

### PTQ vs QAT

| | PTQ | QAT |
|---|---|---|
| **When** | After training | Before/during training |
| **Weights** | Quantized in one shot | Fake-quantized every forward |
| **Activations** | Calibrated, then fixed quantizers | Learned scale/zp during training |
| **Training** | No re-training | Fine-tuning with quantization noise |
| **Accuracy** | Good at 8-bit, degrades at lower bits | Recovers accuracy via fine-tuning |
| **Cost** | Low (one calibration pass) | Higher (additional training epochs) |

### Asymmetric Affine Quantization

For a tensor with min value `x_min` and max value `x_max`, quantized to
`b` bits:

```
qmin = 0, qmax = 2^b - 1
scale = (x_max - x_min) / (qmax - qmin)
zp    = round(-x_min / scale), clamped to [qmin, qmax]
x_q   = round(x / scale + zp), clamped to [qmin, qmax]
x_dq  = (x_q - zp) * scale
```

The zero-point `zp` maps float 0.0 to an integer value, allowing asymmetric
ranges (e.g., after ReLU where all values are non-negative).

### Mixed Precision

Not all layers are equally sensitive to quantization. The first layer
(extracting low-level features) and the last layer (producing class logits)
are typically more sensitive. Mixed precision assigns higher bit widths
to sensitive layers and lower bit widths to robust middle layers:

```
conv1 (first layer):  8-bit  # extracts edges/textures
conv2 (middle):       4-bit  # tolerates more noise
fc1   (middle):       4-bit  # tolerates more noise
fc2   (classifier):   8-bit  # output sensitive
```

## References

- Jacob, B., et al. "Quantization and Training of Neural Networks for
  Efficient Integer-Arithmetic-Only Inference." CVPR 2018.
- Krishnamoorthi, R. "Quantizing deep convolutional networks for efficient
  inference: A whitepaper." arXiv:1806.08342, 2018.
- Bengio, Y., Leonard, N., & Courville, A. "Estimating or Propagating
  Gradients Through Stochastic Neurons for Conditional Computation."
  arXiv:1308.3432, 2013. (STE origin)
- PyTorch Quantization docs: https://pytorch.org/docs/stable/quantization.html
