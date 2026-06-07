# Lecture 05: Linear Quantization int8 / int4 / int2

## Overview

This code accompanies **MIT 6.5940 Lecture 05: Quantization (Part I)**.
It implements linear (affine) quantization at multiple bit widths
(int8, int4, int2), K-means-based non-uniform quantization, and visual
comparison of quantization levels against the original weight distribution.

We use synthetic weights generated from a mixture distribution (Gaussian
with outliers and skew) so the entire pipeline runs on CPU in seconds.

## Prerequisites

```bash
pip install torch matplotlib numpy scipy
```

## Usage

```bash
cd src/lecture-05
python main.py
```

The script runs entirely on CPU (no GPU required) and produces:

1. **Linear (affine) quantization** at int8, int4, and int2
2. **Error comparison** (MSE, MAE, max-abs-error, cosine similarity)
3. **K-means quantization** as a non-uniform alternative
4. **Linear vs K-means comparison** across bit widths
5. **quantization_histogram.png** -- weight distribution with quantization level overlay

Example output:

```
======================================================================
  LECTURE 05: Linear Quantization int8 / int4 / int2
======================================================================

[1] Generating 5000 synthetic weights ...
  Shape: (5000,)
  Min: -6.7731,  Max: 5.6280
  Mean: 0.0925,  Std: 0.8494

[2] Linear (affine) quantisation across bit widths ...
  int 8:  range=[-6.7731, 5.6280], scale=0.048632, zp=139, levels=163,
          MSE=0.000198, MAE=0.012153, cos_sim=0.999865
  int 4:  range=[-6.7731, 5.6280], scale=0.826738, zp=8, levels=15,
          MSE=0.057346, MAE=0.207938, cos_sim=0.962469
  int 2:  range=[-6.7731, 5.6280], scale=4.133691, zp=2, levels=4,
          MSE=0.455586, MAE=0.508281, cos_sim=0.680205

[3] Quantisation error comparison:
  Bit Width             MSE          MAE    Max Abs Err    Cos Sim
  ------------ ------------ ------------ -------------- ----------
  int8             0.000198     0.012153       0.024315   0.999865
  int4             0.057346     0.207938       0.413322   0.962469
  int2             0.455586     0.508281       2.065320   0.680205

[4] K-Means quantisation (non-uniform) ...
  int 8:  clusters=256, MSE=0.001429, MAE=0.008551
  int 4:  clusters=16,  MSE=0.012212, MAE=0.073902
  int 2:  clusters=4,   MSE=0.140943, MAE=0.278583

[5] Linear vs K-Means quantisation comparison:
  Bit Width      Linear MSE  K-Means MSE    Improvement
  ------------ ------------ ------------ --------------
  int8             0.000198     0.001429       -622.98%
  int4             0.057346     0.012212         78.71%
  int2             0.455586     0.140943         69.06%
```

## Key Functions

| Function | Purpose |
|---|---|
| `linear_quantize(tensor, bits)` | Asymmetric affine quantization to `b` bits |
| `dequantize(q, scale, zp)` | Reconstruct float values from quantized integers |
| `compute_quantization_error(orig, recon)` | Compute MSE, MAE, max-abs-error, cosine similarity |
| `kmeans_quantize(tensor, bits)` | Non-uniform quantization via K-means clustering |
| `generate_synthetic_weights(n)` | Generate synthetic weight distribution for testing |
| `plot_weight_histogram(w, info)` | Plot weight histogram with quantization levels |

## What You Learn

### 1. Affine Quantization

Uniform affine (asymmetric) quantization maps floating-point values
to integers using a scale factor and a zero point:

```
scale = (x_max - x_min) / (2^b - 1)
zp    = round(-x_min / scale)
q     = clamp(round(x / scale + zp), 0, 2^b - 1)
x_hat = (q - zp) * scale
```

The **scale** determines the step size between adjacent quantization levels.
The **zero point** maps `0.0` in the float domain to an integer in the
quantized domain, enabling asymmetric range coverage.

### 2. Bit Width and Precision

| Bit Width | Quantization Levels | Typical Use |
|---|---|---|
| **int8** (W8A8) | 256 | Standard for edge deployment (TensorRT, TFLite) |
| **int4** (W4A16/W4A4) | 16 | Aggressive compression (GPTQ, AWQ for LLMs) |
| **int2** | 4 | Extreme compression (binary coding, ternary nets) |

Each halving of bit width approximately doubles the maximum quantization
error (scale doubles, so max error = scale/2 doubles).

### 3. Error Metrics

We compute four metrics to evaluate quantization quality:

- **MSE** (Mean Squared Error): penalizes large deviations more heavily;
  primary metric for quantization.
- **MAE** (Mean Absolute Error): average magnitude of error.
- **Max Absolute Error**: the worst-case error for any single weight.
- **Cosine Similarity**: measures direction preservation; important
  because dot-product operations (the core of matrix multiplication)
  depend on vector directions.

### 4. K-Means (Non-Uniform) Quantization

Instead of uniformly spaced levels, K-means quantization clusters weights
into `k = 2^b` groups and stores each group's centroid:

```
centroids = kmeans(weights, k=2^b)
q[i]      = argmin_j |w[i] - centroids[j]|
w_hat[i]  = centroids[q[i]]
```

**Advantages over linear quantization:**
- Levels are placed where the data actually is (data-dependent)
- Better MSE for the same bit width, especially at low bits

**Disadvantages:**
- Requires the full weight tensor to train (or calibration data)
- Codebook must be stored alongside assignments
- Dequantization needs a lookup operation (centroids[q])

### 5. Linear vs K-Means at Different Bit Widths

Typical observations from the experiment:

- **int8**: Linear quantization performs very well (256 uniform levels
  are dense enough for most distributions). K-means may not improve
  (or even worsen) results due to cluster initialization noise with
  many clusters and relatively few data points.

- **int4**: K-means significantly outperforms linear quantization
  (~70-80% MSE reduction) by placing its 16 centroids at the most
  representative weight values rather than uniformly.

- **int2**: K-means provides a large improvement (~60-70% MSE
  reduction) but both methods are quite lossy at this extreme
  compression level.

### 6. The Quantization Histogram

The generated `quantization_histogram.png` shows:

- **Blue histogram bars**: distribution of original weights
- **Red dashed vertical lines**: quantization levels (dequantized values)
- The grid lines reveal how quantization levels are spaced relative to
  the weight distribution

Key observations:
- At int8, the levels are dense and closely follow the distribution
- At int4, you can see 15 distinct levels (fewer than 16 because
  not all values in the quantization range are used for this distribution)
- At int2, only 4 levels cover the entire range, causing significant
  rounding

## References

- MIT 6.5940 Lecture 05: [EfficientML.ai](https://efficientml.ai)
- HAN Lab: [https://hanlab.mit.edu](https://hanlab.mit.edu)
- Jacob et al., "Quantization and Training of Neural Networks for
  Efficient Integer-Arithmetic-Only Inference" (CVPR 2018)
- Krishnamoorthi, "Quantizing deep convolutional networks for efficient
  inference: A whitepaper" (2018)
- Han et al., "Deep Compression: Compressing Deep Neural Networks with
  Pruning, Trained Quantization and Huffman Coding" (ICLR 2016)
