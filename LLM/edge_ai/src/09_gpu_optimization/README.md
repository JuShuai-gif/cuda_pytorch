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

## 推荐阅读顺序

1. **`timer.h`** — GPU 计时基础设施、CUDA_CHECK 宏及 NVTX 注解，被所有内核演示使用
2. **`matmul_naive.cuh` + `matmul_naive.cu`** — 最基础的 CUDA 内核（仅全局内存），从基础理解 GPU 编程模型
3. **`matmul_tiled.cuh` + `matmul_tiled.cu`** — 共享内存分块优化，展示 tiling 为何能提升性能
4. **`coalesced_demo.cuh` + `coalesced_demo.cu`** — 内存合并 vs 跨步访问模式演示
5. **`stream_pipeline.cuh` + `stream_pipeline.cu`** — CUDA 流重叠计算与传输
6. **`kernel_fusion.cuh` + `kernel_fusion.cu`** — 内核融合（Bias + ReLU），消除内核启动开销
7. **`main.cu`** — 最后阅读，编排所有演示并汇总输出性能对比

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
