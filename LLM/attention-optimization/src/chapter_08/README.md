# Chapter 08: PagedAttention

## Scope

This chapter contains the runnable artifacts for `PagedAttention`. The implementation is intentionally educational: it favors explicit data movement, small reproducible examples, and clear metrics over maximum performance. Production caveats are documented in `notes/chapter_08.md`.

## Files

- `mini_paged_attention.cpp` - main implementation or reading guide.
- `CMakeLists.txt` - build configuration for this chapter.
- `README.md` - build, run, and profiling instructions.

## Build

From project root:

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build . --target mini_paged_attention -j
```

For source-reading-only chapters, there is no binary target; read the chapter notes and use the listed upstream files as a checklist.

## Run

```bash
./chapters/mini_paged_attention
```

If the target is CUDA-based, profile it with:

```bash
ncu --set full ./chapters/mini_paged_attention
nsys profile -o chapter_08_timeline ./chapters/mini_paged_attention
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
