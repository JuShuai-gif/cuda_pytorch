# Chapter 02: GPU Attention Implementation

This chapter implements two educational CUDA kernels for Scaled Dot-Product Attention:

- `naive_attention_kernel`: one CUDA thread computes one `O[row, col]` output element.
- `naive_attention_smem_kernel`: same algorithm with K/V tiles staged through shared memory.
- `unit_test.cu`: compares both GPU kernels with a CPU reference on small tensors.

The code is intentionally simple and explicit. It is not a production FlashAttention replacement; it exists to make memory traffic, redundant softmax work, and tiling tradeoffs visible.

## Build

From the project root:

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build . --target naive_attention_gpu chapter_02_unit_test -j
```

If your GPU is not Ampere/A100-class, replace `80` with your compute capability, for example `75`, `86`, or `90`.

## Run Unit Tests

```bash
./chapters/chapter_02_unit_test
```

The tests validate:

- global-memory CUDA kernel vs CPU reference;
- shared-memory CUDA kernel vs CPU reference;
- non power-of-two shapes and small boundary cases.

## Run Benchmark

```bash
./chapters/naive_attention_gpu
```

Output columns:

| Column | Meaning |
|---|---|
| `N` | sequence length |
| `Time(ms)` | average kernel latency after warmup |
| `BW(GB/s)` | estimated effective memory bandwidth |
| `TFLOPS` | estimated effective attention throughput |
| `Kernel` | `Global` or `SharedMem` implementation |

## Profile

```bash
ncu --set full ./chapters/naive_attention_gpu
```

Recommended Nsight Compute checks:

- achieved occupancy;
- DRAM throughput;
- SM throughput;
- warp stall reasons;
- shared-memory usage and occupancy interaction.

## Engineering Notes

The naive global kernel recomputes `Q[row] @ K^T` three times per output element: max, exp-sum, and weighted-sum phases. Because each output column repeats the same softmax work, the implementation is dominated by redundant memory traffic and redundant compute. The shared-memory version demonstrates tiling, but it does not remove the algorithmic redundancy. Chapter04 addresses this with FlashAttention-style online softmax and kernel fusion.
