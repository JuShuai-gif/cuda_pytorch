# 08_computer_architecture - Cache & Memory Hierarchy Benchmarks

## Overview

This directory contains benchmarks demonstrating key computer architecture concepts.

## File Structure

```
08_computer_architecture/
  timer.h          - Timer utility, print_header(), g_sink anti-optimization
  cache_bench.h    - Cache benchmark declarations
  cache_bench.cpp  - Cache line detection, cache hit vs miss, false sharing, row vs column
  numa_bench.h     - NUMA benchmark declaration
  numa_bench.cpp   - NUMA node awareness benchmark (cross-node access latency)
  simd_bench.h     - SIMD benchmark declaration
  simd_bench.cpp   - SIMD optimization demo (AVX2: FMA, dot product)
  sys_info.h       - System info query declaration
  sys_info.cpp     - System CPU/cache info query via sysconf
  main.cpp         - Entry point calling all benchmarks
  CMakeLists.txt
  README.md
```

## 推荐阅读顺序

1. **`timer.h`** — 共享计时基础设施、`print_header()` 工具函数及 `g_sink` 防优化变量，被所有 benchmark 依赖
2. **`sys_info.h` + `sys_info.cpp`** — 系统信息查询，逻辑最简单，最先被执行，为后续 benchmark 提供硬件上下文
3. **`cache_bench.h` + `cache_bench.cpp`** — 核心缓存概念：缓存行探测、命中 vs 未命中、伪共享、行优先 vs 列优先遍历
4. **`numa_bench.h` + `numa_bench.cpp`** — NUMA 拓扑感知及跨节点访问延迟，是对"内存访问成本"主题的延伸
5. **`simd_bench.h` + `simd_bench.cpp`** — AVX2 SIMD 向量化优化（FMA、点积），独立于缓存 benchmark 但在性能分析主题内
6. **`main.cpp`** — 最后阅读，作为入口点串联所有 benchmark 调用

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./arch_benchmarks
```

## NUMA Awareness

For NUMA benchmarks:
- If `libnuma` is available, the program measures cross-node access latency
- If running on a single NUMA node system, cross-node measurement is skipped
- To see NUMA topology: `numactl --hardware`
- To run bound to a specific node: `numactl --cpunodebind=0 --membind=0 ./arch_benchmarks`

## Requirements

- C++17 compiler (GCC 9+, Clang 10+)
- CMake >= 3.14
- Linux
- Optional: libnuma-dev for NUMA benchmarks
- Optional: AVX2-capable CPU for SIMD demo

## Notes

- Run with `sudo` to use `mlockall` for stable cache measurements
- For NUMA benchmarks, a multi-socket machine gives the most interesting results
- Cache line size is auto-detected; x86 should show 64 bytes
