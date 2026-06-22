# Attention Optimization: From Theory to Industrial Kernel

A systematic CUDA/PyTorch learning project for Transformer Attention optimization, from baseline attention to FlashAttention, KV cache, PagedAttention, and production inference kernels.

## Goals

- Understand Scaled Dot-Product Attention from math, memory, and kernel perspectives.
- Build reproducible CPU/CUDA implementations chapter by chapter.
- Learn how to profile latency, throughput, occupancy, memory traffic, and arithmetic intensity.
- Read industrial kernels such as FlashAttention, TensorRT-LLM attention, xFormers memory-efficient attention, and vLLM PagedAttention.

## Project Structure

```text
attention-optimization/
├── notes/                   # Chinese chapter notes with formulas and diagrams
│   ├── chapter_01.md
│   ├── chapter_02.md        # GPU Attention implementation, tiling, Roofline
│   └── chapter_16.md
├── src/
│   ├── chapter_01/          # CPU naive attention baseline
│   ├── chapter_02/          # CUDA naive + shared-memory attention
│   ├── chapter_03/          # Profiling workflow
│   ├── chapter_04/          # FlashAttention V1
│   ├── chapter_05/          # FlashAttention V2
│   ├── chapter_06/          # FlashAttention V3 concepts
│   ├── chapter_07/          # KV Cache
│   ├── chapter_08/          # PagedAttention
│   ├── chapter_09/          # MQA / GQA
│   ├── chapter_10/          # Sliding Window Attention
│   ├── chapter_11/          # Sparse Attention
│   ├── chapter_12/          # Linear Attention
│   ├── chapter_13/          # Quantized Attention
│   ├── chapter_14/          # TensorRT-LLM Attention
│   ├── chapter_15/          # xFormers source analysis
│   └── chapter_16/          # vLLM source analysis
├── benchmark/               # Shared benchmark helpers
├── profiler/                # Profiling scripts and CLI helpers
├── docs/                    # Architecture and supporting documentation
├── final_project/           # Mini TensorRT-LLM-style attention engine
└── README.md
```

## Learning Roadmap

| Phase | Chapters | Focus |
|---|---|---|
| Foundations | 01-03 | attention math, CPU/GPU baseline, profiling |
| FlashAttention | 04-06 | online softmax, tiling, fusion, FA2/FA3 ideas |
| Inference | 07-09 | KV cache, PagedAttention, MQA/GQA |
| Advanced Patterns | 10-13 | sliding window, sparse, linear, quantized attention |
| Production Systems | 14-16 | TensorRT-LLM, xFormers, vLLM source reading |

## Dependencies

Recommended environment:

- Linux with NVIDIA GPU
- CUDA Toolkit 11.8+ or 12.x
- CMake 3.18+
- C++17 compiler
- Python 3.9+
- PyTorch for Python benchmarks/profiling
- Nsight Systems / Nsight Compute for GPU profiling

Ampere or newer GPUs are recommended. The default CMake examples use `sm_80`; change `CMAKE_CUDA_ARCHITECTURES` for your hardware.

## Build

```bash
cd /home/hpc/ghr_code/cuda_pytorch/attention-optimization
mkdir -p build
cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build . -j
```

Build a specific chapter target:

```bash
cmake --build . --target naive_attention_gpu chapter_02_unit_test -j
```

Run registered CTest checks:

```bash
ctest --test-dir build --output-on-failure
```

## Run

Chapter01 CPU baseline:

```bash
./chapters/naive_attention
python ../src/chapter_01/benchmark.py
```

Chapter02 CUDA tests and benchmark:

```bash
./chapters/chapter_02_unit_test
./chapters/naive_attention_gpu
```

Python benchmark/profiling examples:

```bash
python ../src/chapter_02/benchmark.py
python ../src/chapter_03/attention_profile.py
```

Nsight Compute example:

```bash
ncu --set full ./chapters/naive_attention_gpu
```

## Chapter02 Content Example

Chapter02 now demonstrates the expected quality bar for future chapters:

- notes explain GPU mapping, memory hierarchy, formulas, FLOPs, memory footprint, and Roofline analysis;
- CUDA code contains global-memory and shared-memory kernels with explicit comments on tiling and redundant work;
- `unit_test.cu` validates GPU outputs against a CPU reference;
- benchmark reports latency, estimated bandwidth, and estimated TFLOPS;
- README documents build, test, benchmark, and profiling commands.

## Coding Conventions

- Notes: Chinese explanation with English technical terms and formulas.
- C++/CUDA comments: English, focused on kernel mapping and memory behavior.
- C++ standard: C++17.
- CUDA code: explicit error checking, deterministic small tests, benchmark warmup before timing.
- Python: PEP 8 style.

## Industrial Optimization Checklist

For each chapter implementation, prefer this progression:

1. CPU or PyTorch reference for correctness.
2. Small deterministic unit tests.
3. Reproducible benchmark shapes and warmup.
4. Nsight profile for latency, occupancy, memory, and stalls.
5. Clear notes linking math, code, and measured bottlenecks.
