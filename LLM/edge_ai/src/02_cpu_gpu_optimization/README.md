# CPU/GPU Hybrid Robot Perception Pipeline

Realistic robot perception benchmark demonstrating CPU-GPU hybrid processing with real data and actual CUDA kernels.

## Pipeline

1. **CPU**: Generate synthetic 1920x1080x3 camera image (scene with sky, road, vehicles, pedestrians)
2. **CPU**: Bilinear resize to 640x480x3 float32, normalize to [0,1]
3. **GPU**: Conv2D (16 filters, 3x3 kernel) on image tensor
4. **GPU**: ReLU activation
5. **GPU**: 2x2 MaxPool downsampling
6. **GPU**: Detection head decodes feature maps to bounding box parameters
7. **CPU**: Non-Maximum Suppression on detected boxes

## Three Approaches Compared

| Approach | Strategy |
|----------|----------|
| Naive | Sequential: CPU→H2D→GPU→D2H→CPU per frame, default stream |
| Stream Overlap | 3 CUDA streams interleave transfers and compute across frames |
| Pinned Mapped | `cudaHostAllocMapped` for zero-copy; GPU reads/writes host memory directly |

## File Structure

```
02_cpu_gpu_optimization/
├── CMakeLists.txt
├── README.md
├── timer.h                   # CpuTimer and GpuTimer utilities
├── cpu_preprocess.h          # Image generation, resize, NMS, voxelize declarations
├── cpu_preprocess.cpp        # Actual implementations with real loops
├── gpu_inference.cuh         # CUDA kernel wrapper declarations and CUDA_CHECK
├── gpu_inference.cu          # Conv2D, ReLU, MaxPool, DetectionHead kernels
├── pipeline_runner.h         # Constants and runner function declarations
├── pipeline_runner.cpp       # Three pipeline approach implementations + JSON writer
└── main.cu                   # Entry point: device info, benchmarks, output
```

## 推荐阅读顺序

1. **`timer.h`** — CPU 与 GPU 高分辨率计时工具类，被 pipeline_runner 和 main 共同使用
2. **`gpu_inference.cuh` + `gpu_inference.cu`** — CUDA 内核（Conv2D、ReLU、MaxPool、DetectionHead），GPU 端核心计算
3. **`cpu_preprocess.h` + `cpu_preprocess.cpp`** — CPU 端预处理（图像生成、resize）和后处理（NMS）
4. **`pipeline_runner.h` + `pipeline_runner.cpp`** — 编排三种方案（naive、stream-overlapped、pinned-mapped）的实现
5. **`main.cu`** — 最后阅读，调用所有 run_*() 函数并汇总输出性能对比

## Prerequisites

- NVIDIA GPU with CUDA support
- CMake 3.18+
- CUDA Toolkit installed

## Build

```bash
mkdir build && cd build
cmake ..
make
```

## Run

```bash
./cpu_gpu_bench
```

Outputs `gpu_pipeline_metrics.json` with latency, throughput, and breakdown for all three approaches.
