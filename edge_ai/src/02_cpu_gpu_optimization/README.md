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
