# Lecture 01: Introduction - Parameter Counting, FLOPs Estimation, Model Size

## Overview

This code accompanies **MIT 6.5940 Lecture 01: Introduction to Efficient Deep Learning**.
It implements the fundamental profiling primitives that every efficiency-aware ML
practitioner needs: counting parameters, estimating compute (FLOPs/MACs), measuring
model size, and benchmarking inference latency.

We use `torchvision.models.resnet18` as the canonical example and print a summary
table that connects each metric to the hardware constraints discussed in the lecture.

## Prerequisites

```bash
pip install torch torchvision
```

## Usage

```bash
cd src/lecture-01
python main.py
```

The script runs entirely on CPU (no GPU required) and produces output similar to:

```
Loading torchvision.models.resnet18 ...
Parameters: 11,689,512 total, 11,689,512 trainable
Estimating Conv2d MACs ...
Conv2d MACs: 1,819,162,112
Model size (FP32): 44.59 MiB
Measuring CPU inference latency ...
Avg. inference time: 123.45 ms

============================================================
  MODEL EFFICIENCY SUMMARY: ResNet-18
============================================================
  Metric                                      Value
  --------------------------------------------------------
  Total parameters                      11,689,512
  Trainable parameters                  11,689,512
  Total Conv2d MACs                  1,819,162,112
  Total FLOPs (MACs x 2)             3,638,324,224
  Model size (FP32, MiB)                    44.59
  CPU inference latency                   123.45 ms
============================================================
```

## Key Functions

| Function | Purpose |
|---|---|
| `count_parameters(model)` | Returns (total, trainable) parameter counts |
| `estimate_flops_conv2d(in_c, out_c, k, h, w)` | MACs for one Conv2d layer |
| `estimate_total_flops(model, input_shape)` | Total Conv2d MACs via forward hooks |
| `model_size_mb(total_params)` | Converts param count to MiB (FP32) |
| `measure_inference_time(model, input_shape)` | Average CPU forward-pass latency |

## What You Learn

### 1. Parameter Counting

```
total_params = sum(p.numel() for p in model.parameters())
```

ResNet-18 has ~11.7M parameters. Each FP32 parameter occupies **4 bytes**, so the
model weighs ~44.6 MiB in memory -- already approaching the limit of a typical
microcontroller's SRAM (64-512 KB). This is why **pruning** (Lecture 03-04) and
**quantization** (Lecture 05-06) are essential for edge deployment.

### 2. FLOPs/MACs Estimation

For a single Conv2d layer with input `(C_in, H, W)`, kernel `K x K`, and output
channels `C_out`:

```
MACs = C_out * H_out * W_out * (C_in / groups) * K * K
```

where `H_out = (H + 2*P - K) / S + 1`.

One **MAC** (Multiply-Accumulate) = one multiplication + one addition = **2 FLOPs**.
ResNet-18 performs ~1.82 GigMACs (~3.64 GFLOPs) per 224x224 image -- that is fine
for a desktop GPU but challenging for a smartphone running dozens of inferences
per second.

**Common pitfall**: MACs ≠ latency. A depthwise convolution has dramatically fewer
MACs but can be slower on GPU due to poor memory bandwidth utilisation (see Lecture 02).

### 3. Model Size (Memory Footprint)

```
model_size_MiB = total_params * bytes_per_param / (1024 * 1024)
```

| Precision | Bytes/Param | ResNet-18 Size |
|---|---|---|
| FP32 | 4 | ~44.6 MiB |
| FP16 | 2 | ~22.3 MiB |
| INT8 | 1 | ~11.1 MiB |

### 4. Inference Latency

Measured as the **average forward-pass time** on CPU. This single number hides
important details (memory-bound vs. compute-bound ops, GPU vs. CPU characteristics)
that the rest of the course explores in depth.

## Connection to Lecture 01

The lecture introduces the core tension of efficient deep learning:

> Model size grows 4x every 2 years, but hardware capabilities only grow 2x.

This code quantifies that tension by giving you the tools to measure exactly how
"big" a model really is -- in parameters, in operations, in bytes, and in
wall-clock time. These metrics are the foundation for every optimisation technique
covered in Lectures 02-23:

- **Pruning**: reduces `total_params` by removing unimportant weights
- **Quantization**: reduces `bytes_per_param` from 4 (FP32) to 2 (FP16) or 1 (INT8)
- **NAS + Distillation**: designs models with fewer `MACs` while preserving accuracy
- **TinyML (MCUNet/TinyEngine)**: co-designs model + inference engine so that both
  `model_size_MiB` and latency fit within MCU constraints (< 256 KB SRAM, < 10 ms)

## References

- MIT 6.5940 Lecture 01: [EfficientML.ai](https://efficientml.ai)
- HAN Lab: [https://hanlab.mit.edu](https://hanlab.mit.edu)
- ResNet Paper: He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016)
