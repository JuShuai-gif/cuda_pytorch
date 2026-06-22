# Lecture 11 - TinyEngine Optimization Simulation

> MIT 6.5940: Efficient Deep Learning Computing  
> Topic: TinyEngine — 端侧推理引擎优化

## Overview

TinyEngine applies multiple optimization techniques to accelerate neural network inference on resource-constrained devices. This simulation benchmarks different convolution implementations and operator fusion strategies.

## Key Concepts

- **im2col + GEMM**: Converts convolution to matrix multiplication for efficient computation
- **Winograd F(2,3)**: Reduces multiply operations by 2.25× for 3×3 convolutions with stride 1
- **Operator Fusion**: Merges Conv + BatchNorm + ReLU into a single fused operation
- **Memory Layout**: NCHW vs NHWC — channel-first vs channel-last access patterns affect cache efficiency

## Implementation

| Technique           | Description                                              |
| ------------------- | -------------------------------------------------------- |
| Naive Conv          | Baseline using `F.conv2d` with CPU-optimized backend     |
| im2col + GEMM       | Manual patch unfolding → column matrix → matmul          |
| Winograd F(2,3)     | Transform domain convolution with B^T/G/A^T matrices     |
| Conv+BN+ReLU Fusion | Mathematical folding of BatchNorm into Conv weights       |
| NCHW vs NHWC        | Channel-first vs channel-last memory layout benchmarking |
| Inference Benchmark  | Timing comparison across all methods                      |

## Usage

```bash
cd src/lecture-11
python main.py
```

## Expected Output

```
============================================================
Convolution Optimization Benchmark
============================================================
Method              Time (ms)    Speedup    Max Error
------------------------------------------------------------
Naive (F.conv2d)      12.34       1.00×      -
im2col + GEMM         8.56        1.44×      1.5e-04
Winograd F(2,3)       5.48        2.25×      1.4e-04
Fused Conv+BN+ReLU    6.72        1.84×      3.7e-04
============================================================
Memory Layout Comparison:
NCHW: 1.234 ms  (channel-first)
NHWC: 0.987 ms  (channel-last, better cache locality)
============================================================
```

## References

- Lavin & Gray, "Fast Algorithms for Convolutional Neural Networks" (CVPR 2016)
- MIT 6.5940 Lecture 11 Slides
