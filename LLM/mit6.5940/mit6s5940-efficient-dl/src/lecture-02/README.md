# Lecture 02: Efficiency Metrics Deep Dive

## Overview

This code accompanies **MIT 6.5940 Lecture 02: Efficiency Metrics Deep Dive**.
It benchmarks **latency**, **throughput**, **parameter count**, and **FLOPs** for
three convolutional neural network architectures on CPU:

| Model | Type | Approx. Params | Description |
|---|---|---|---|
| **CustomCNN** | Hand-crafted | ~0.4M | 5-layer convnet with BatchNorm, designed for benchmarking |
| **ResNet18** | torchvision | ~11.7M | Residual network with skip connections (He et al., 2016) |
| **MobileNetV2** | torchvision | ~3.5M | Mobile-optimised with inverted residuals and depthwise convolutions |

The script produces a Markdown-formatted report table that connects each metric
to the hardware efficiency concepts discussed in the lecture.

## Prerequisites

```bash
pip install torch torchvision
```

## Usage

```bash
cd src/lecture-02
python main.py
```

The script runs entirely on CPU (no GPU required) and produces output similar to:

```
================================================================================
  LECTURE 02: Efficiency Metrics Deep Dive  --  Benchmark Report
================================================================================

## Model Overview

| Model          |      Params |   Trainable |          MACs |         FLOPs | Size (MiB) |
| -------------- | ----------- | ----------- | ------------- | ------------- | ----------- |
| CustomCNN      |    3,197,226 |   3,197,226 |    588,857,344 |  1,177,714,688 |      12.20  |
| ResNet18       |   11,689,512 |  11,689,512 |  1,819,162,112 |  3,638,324,224 |      44.59  |
| MobileNetV2    |    3,504,872 |   3,504,872 |    308,624,640 |    617,249,280 |      13.37  |

## Latency (batch_size=1, CPU)

| Model          |  Latency (ms) |
| -------------- | ------------- |
| CustomCNN      |         45.23 |
| ResNet18       |        123.45 |
| MobileNetV2    |         55.67 |

## Throughput (samples/sec) vs Batch Size (CPU)

| Model          | b= 1          | b= 4          | b=16          | b=32          |
| -------------- | ------------- | ------------- | ------------- | ------------- |
| CustomCNN      |          22.1 |          60.3 |         120.5 |         150.2 |
| ResNet18       |           8.1 |          18.4 |          25.1 |          28.3 |
| MobileNetV2    |          18.0 |          45.2 |          80.1 |          95.4 |
```

*(All numbers above are illustrative; actual values depend on your CPU.)*

## Key Functions

| Function | Purpose |
|---|---|
| `count_parameters(model)` | Returns (total, trainable) parameter counts |
| `estimate_total_flops(model, input_shape)` | Total Conv2d MACs via forward hooks |
| `estimate_flops_conv2d(in_c, out_c, k, h, w)` | MACs for one Conv2d layer |
| `model_size_mb(total_params)` | Converts param count to MiB (FP32) |
| `measure_latency(model, input_shape)` | Single-sample CPU forward-pass latency |
| `measure_throughput(model, input_shape, batch_size)` | Samples/sec at given batch size |
| `generate_markdown_report(results, batch_sizes)` | Formats results as a Markdown table |

## What You Learn

### 1. Latency vs. Throughput

**Latency** (time per sample when batch_size=1) tells you how fast a single
inference completes -- critical for interactive applications (real-time video,
voice assistants, autonomous driving). **Throughput** (samples per second) at
larger batch sizes reveals how well the model amortises overhead across
multiple inputs -- important for cloud serving and batch-processing pipelines.

Key observation: latency often does **not** decrease proportionally with
model size.  MobileNetV2 has ~1/3 the MACs of ResNet18 but achieves only
~2x lower latency on CPU because of differences in memory access patterns
and operator-level efficiency on general-purpose hardware.

### 2. Batch Size Effects

Increasing batch size improves throughput up to a point, after which you hit
the **memory bandwidth wall** on CPU:

| Batch Size | Effect |
|---|---|
| 1 | Minimum throughput; dominated by kernel-launch overhead |
| 4-16 | Sweet spot: better cache reuse, good throughput scaling |
| 32+ | Diminishing returns; L3 cache thrashing begins |

The script measures throughput at batch sizes [1, 4, 16, 32] so you can
identify the optimal batch size for your specific hardware.

### 3. MACs, FLOPs, and the Roofline Model

For a Conv2d layer: **MACs = C_out * H_out * W_out * (C_in / groups) * K^2**.
One MAC (multiply-accumulate) = 2 FLOPs.  But MACs alone do not predict
latency -- the **roofline model** (Lecture 02) separates operations into:

- **Compute-bound**: Time dominated by arithmetic (e.g., large-channel 3x3 conv)
- **Memory-bound**: Time dominated by data movement (e.g., depthwise conv, 1x1 conv)

MobileNetV2's depthwise convolutions have dramatically fewer MACs than ResNet18's
dense 3x3 convs, but the speedup is less than MAC-count alone would predict
because depthwise ops are memory-bandwidth-bound on most CPUs/GPUs.

### 4. Parameter Efficiency

| Model | Params | MACs | MACs/Param | Interpretation |
|---|---|---|---|---|
| CustomCNN | 0.4M | 1,467M | 3,707 | Simple convs, high MACs/param ratio |
| ResNet18 | 11.7M | 1,819M | 156 | Residual blocks, good gradient flow |
| MobileNetV2 | 3.5M | 309M | 88 | Depthwise separable, few params but lower MACs/param |

MobileNetV2 trades lower `MACs/Param` (less compute per stored weight) for
dramatically lower total compute -- a deliberate design choice for mobile
deployment where FLOPS budget is the binding constraint.

## Models in Detail

### CustomCNN

```
Conv2d(3, 16, k=3) -> BN -> ReLU
Conv2d(16, 32, k=3, s=2) -> BN -> ReLU
Conv2d(32, 64, k=3) -> BN -> ReLU
Conv2d(64, 128, k=3, s=2) -> BN -> ReLU
Conv2d(128, 256, k=3) -> BN -> ReLU
AdaptiveAvgPool2d(1) -> Flatten -> Linear(256, 10)
```

A plain 5-conv + 1-FC architecture with progressive channel expansion
(16 → 32 → 64 → 128 → 256) and two spatial downsampling stages.  No skip
connections, no depthwise convolutions -- a simple baseline against which
the optimised architectures are compared.

### ResNet18

Standard torchvision implementation with residual blocks and BatchNorm.
The skip connections improve gradient flow but add negligible compute overhead.
Approximately 1.82 GigMACs per 224x224 input.

### MobileNetV2

Inverted residual blocks with depthwise separable convolutions.  Achieves
~309 MMACs (6x fewer than ResNet18) with only ~3.5M parameters.  The linear
bottlenecks and inverted residuals minimise memory footprint while preserving
representational capacity.

## Connection to Lecture 02

Lecture 02 deepens the efficiency metrics introduced in Lecture 01 by asking:

> Given two models with similar accuracy, how do we choose which is *actually*
> more efficient?

This code provides the quantitative framework to answer that question:

- **Latency + Throughput** capture wall-clock performance on real hardware
- **MACs + FLOPs** estimate theoretical compute requirements
- **Parameter count + Model size** quantify storage and memory costs

Together, these six numbers tell the full efficiency story that any single
metric (e.g., "smaller model = faster") would miss.

## References

- MIT 6.5940 Lecture 02: [EfficientML.ai](https://efficientml.ai)
- HAN Lab: [https://hanlab.mit.edu](https://hanlab.mit.edu)
- ResNet: He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016)
- MobileNetV2: Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks" (CVPR 2018)
- Roofline Model: Williams, Waterman, Patterson, "Roofline: An Insightful Visual Performance Model" (CACM 2009)
