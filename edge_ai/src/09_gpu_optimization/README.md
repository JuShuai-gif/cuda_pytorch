# 09_gpu_optimization - CUDA Kernel Optimization Examples

## Overview

This directory contains CUDA kernel implementations demonstrating key GPU optimization techniques:

- **Matrix multiplication**: naive -> shared memory tiled -> further optimized
- **Memory coalescing**: coalesced vs strided access patterns
- **CUDA streams**: overlapping multiple kernel launches (pipeline)
- **Kernel fusion**: fuse bias + ReLU into a single kernel (TensorRT-style concept)
- **Occupancy tuning**: register pressure and block size analysis
- **NVTX annotations**: performance markers for Nsight profiling

## File Structure

```
09_gpu_optimization/
|-- timer.h              # GPUTimer class, CUDA_CHECK macro, NVTX annotations
|-- matmul_naive.cuh     # Naive matrix multiplication kernel declaration
|-- matmul_naive.cu      # Naive matrix multiplication kernel definition
|-- matmul_tiled.cuh     # Tiled + optimized matmul kernel declarations (shared memory)
|-- matmul_tiled.cu      # Tiled + optimized matmul kernel definitions
|-- coalesced_demo.cuh   # Coalesced/strided access kernel declarations + demo
|-- coalesced_demo.cu    # Coalesced/strided access kernel definitions + demo impl
|-- stream_pipeline.cuh  # CUDA streams compute kernel declaration + demo
|-- stream_pipeline.cu   # CUDA streams compute kernel definition + demo impl
|-- kernel_fusion.cuh    # Bias/ReLU/fused kernel declarations + demo
|-- kernel_fusion.cu     # Bias/ReLU/fused kernel definitions + demo impl
|-- main.cu              # Entry point: demo_matmul, demo_occupancy, device info
|-- CMakeLists.txt
|-- README.md
```

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./gpu_optimization_demos
```

## Prerequisites

- NVIDIA GPU with CUDA support (Pascal/Volta/Turing/Ampere or newer)
- CUDA Toolkit >= 11.0
- CMake >= 3.18
- NVTX headers for annotations (bundled with CUDA or install separately)

## Adjusting GPU Architecture

Edit `CMakeLists.txt` and change `CMAKE_CUDA_ARCHITECTURES` to match your GPU:

| GPU | Architecture |
|-----|-------------|
| GTX 10xx | 61 |
| Tesla V100 | 70 |
| RTX 20xx | 75 |
| A100 | 80 |
| RTX 30xx | 86 |
| RTX 40xx | 89 |

## Profiling

```bash
# Nsight Compute - analyze individual kernels
ncu --set full -o matmul_report ./gpu_optimization_demos

# Nsight Systems - system-level timeline with NVTX annotations
nsys profile --trace=cuda,nvtx,osrt -o timeline ./gpu_optimization_demos
nsys-ui timeline.nsys-rep
```
