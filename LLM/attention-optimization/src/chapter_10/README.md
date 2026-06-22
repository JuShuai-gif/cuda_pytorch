# Chapter 10: Sliding Window Attention

## Scope

This chapter contains the runnable artifacts for `Sliding Window Attention`. The implementation is intentionally educational: it favors explicit data movement, small reproducible examples, and clear metrics over maximum performance. Production caveats are documented in `notes/chapter_10.md`.

## Files

- `sliding_window_attention.cpp` - main implementation or reading guide.
- `CMakeLists.txt` - build configuration for this chapter.
- `README.md` - build, run, and profiling instructions.

## Build

From project root:

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build . --target sliding_window_attention -j
```

For source-reading-only chapters, there is no binary target; read the chapter notes and use the listed upstream files as a checklist.

## Run

```bash
./chapters/sliding_window_attention
```

If the target is CUDA-based, profile it with:

```bash
ncu --set full ./chapters/sliding_window_attention
nsys profile -o chapter_10_timeline ./chapters/sliding_window_attention
```

## Metrics To Report

| Metric | Why it matters |
|---|---|
| Latency | End-to-end user-visible cost. |
| Throughput | Tokens/s, requests/s, or effective TFLOPS. |
| Memory traffic | Determines whether the kernel is bandwidth-bound. |
| Occupancy | Helps explain latency hiding, register pressure, and SMEM pressure. |
| Correctness check | Prevents benchmarking a broken optimization. |

## Engineering Notes

Keep every experiment reproducible: record shape, dtype, device, warmup, iteration count, and whether the path is prefill or decode. For attention kernels, always distinguish mathematical complexity from real HBM traffic.
