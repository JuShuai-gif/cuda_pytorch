# Chapter 03: Attention Profiling

Tools and scripts for profiling attention kernels.

## Files

- `attention_profile.py` - PyTorch profiler + nvtx annotated attention benchmark
- `benchmark.cu` - CUDA micro-benchmarks with NVTX ranges
- `CMakeLists.txt` - Build configuration
- `benchmark.py` - Python benchmarking with torch.profiler integration

## Usage

```bash
# Python profiling
python attention_profile.py

# Nsight Systems profiling
nsys profile -o profile_output python attention_profile.py

# Nsight Compute kernel profiling
ncu --kernel-name naive_attention_kernel --set full ./build/chapter_03/bench
```
