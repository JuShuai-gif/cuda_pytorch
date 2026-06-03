# Chapter 13 — 内存模型与缓存优化进阶

深入 CPU 缓存、内存布局、NUMA 架构的优化实战。

## 内容概览

| 文件 | 主题 | 关键知识点 |
|------|------|-----------|
| `01_cache_line_alignment.cpp` | 缓存行对齐 | cache line 检测、alignas、跨行访问开销 |
| `02_false_sharing_bench.cpp` | 伪共享基准测试 | 同 cache line vs padding 分隔的吞吐量对比 |
| `03_memory_fence.cpp` | 内存栅栏 | acquire/release fence、seq_cst fence、双检锁 |
| `04_numa_affinity.cpp` | NUMA与CPU绑核 | NUMA 拓扑检测、pthread_setaffinity、线程迁移 |
| `05_data_layout_optimization.cpp` | 数据布局优化 | AoS vs SoA、冷热分离、padding 最佳实践 |

## 编译运行

```bash
mkdir build && cd build
cmake ..
cmake --build . -j$(nproc)
./ch13_01_cache_line_alignment
./ch13_02_false_sharing_bench
./ch13_03_memory_fence
./ch13_04_numa_affinity
./ch13_05_data_layout_optimization
```

## 学习建议

1. 先理解 CPU 缓存层级结构（L1/L2/L3/RAM 的延迟差异）
2. 重点掌握伪共享的识别和修复（padding + alignas）
3. NUMA 优化需要特定硬件（多路服务器），可先理解原理
4. 数据布局（AoS vs SoA）对 SIMD 和缓存效率影响巨大
