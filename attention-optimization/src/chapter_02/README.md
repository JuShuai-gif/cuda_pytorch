# Chapter 02: GPU Attention Implementation

## Implementation

- `cuda_naive_attention.cu` - Direct GPU port of naive attention
- Maps the O(N^2) attention pattern onto CUDA thread hierarchy
- Compares performance vs CPU baseline from Chapter 01

## Key Concepts

- Matrix multiplication as the core primitive
- GEMM mapping to GPU thread blocks
- Tensor Core fundamentals
- Shared Memory (SMEM) role
- Warp-level execution
- Thread block decomposition

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
make chapter_02_naive_attention
```

## Run

```bash
./chapters/chapter_02/naive_attention
```

## Expected Outcome

Understand:
1. Why a naive GPU implementation is still slow
2. Where the memory bottleneck really is
3. How to decompose attention computation across thread blocks
