# src 实验源码

本目录包含 28 组 C++17 内存性能实验，由 `CMakeLists.txt` 统一管理。

## 编译

```bash
cd src
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

选项：

| CMake 选项 | 默认 | 说明 |
|---|---|---|
| `CMAKE_BUILD_TYPE` | — | Debug / Release / RelWithDebInfo |
| `ENABLE_NATIVE_OPTIMIZATION` | OFF | 附加 `-march=native` |
| `ENABLE_NUMA_EXAMPLES` | OFF | 编译 NUMA 实验（需 libnuma） |
| `ENABLE_AVX2_EXAMPLES` | OFF | AVX2 示例（运行时检测） |
| `ENABLE_AVX512_EXAMPLES` | OFF | AVX-512 示例（运行时检测） |

通用编译参数：`-Wall -Wextra -Wpedantic`，Release 为 `-O3 -g -fno-omit-frame-pointer`。

## 实验索引

| 可执行文件 | 目录 | 主题 |
|---|---|---|
| memory_latency | 01_memory_latency | 内存层级延迟 + pointer chasing |
| sequential_random_access | 02_sequential_random_access | 顺序/逆序/步长/随机访问 |
| cache_line_size | 03_cache_line_size | Cache Line 大小与利用率 |
| stride_access | 04_stride_access | Stride 访问与 TLB/行效应 |
| cache_capacity | 05_cache_capacity | 工作集 vs 缓存容量 |
| cache_associativity | 06_cache_associativity | 关联度与 Set 冲突 |
| cache_conflict | 07_cache_conflict | 冲突未命中演示 |
| write_back_behavior | 08_write_back_behavior | 写策略行为 |
| matrix_traversal | 09_matrix_traversal | 行/列遍历与转置 |
| cache_blocking | 10_cache_blocking | 分块大小扫描 |
| aos_soa | 11_aos_soa | AoS vs SoA |
| pointer_chasing | 12_pointer_chasing | 链表 vs 连续数组 |
| false_sharing | 13_false_sharing | 同缓存行 vs padding |
| atomic_contention | 14_atomic_contention | 局部/atomic/CAS/mutex/reduction |
| thread_affinity | 15_thread_affinity | 绑核 vs 不绑核 |
| prefetch | 16_prefetch | 无/软件预取/距离 |
| non_temporal_store | 17_non_temporal_store | 流式写 vs 普通写 |
| tlb_capacity | 18_tlb_capacity | TLB 容量 |
| page_size | 19_page_size | 页面大小影响 |
| huge_pages | 20_huge_pages | THP / hugetlbfs（环境检查） |
| page_fault | 21_page_fault | minor/major/MAP_POPULATE |
| memory_mapping | 22_memory_mapping | mmap vs read |
| memory_bandwidth | 23_memory_bandwidth | STREAM 类带宽 |
| numa_local_remote | 24_numa_local_remote | NUMA 本地/远程（需 libnuma） |
| numa_first_touch | 25_numa_first_touch | NUMA first-touch（需 libnuma） |
| numa_replication | 26_numa_replication | NUMA 复制（需 libnuma） |
| instruction_cache | 27_instruction_cache | 指令缓存与代码布局 |
| integrated_project | 28_integrated_project | 综合：矩阵乘法优化链 |
| p1_debug_vs_release | 29_engineering_pitfalls | 坑：Debug 构建测速 |
| p2_dead_code_elimination | 29_engineering_pitfalls | 坑：基准被优化掉 |
| p3_vector_bool | 29_engineering_pitfalls | 坑：vector\<bool\> |
| p4_shared_ptr_contention | 29_engineering_pitfalls | 坑：shared_ptr 原子计数竞争 |
| p5_alignment | 29_engineering_pitfalls | 坑：堆对象不对齐 |
| p6_benchmark_noise | 29_engineering_pitfalls | 坑：单次测速噪声 |
| p7_atomic_memory_order | 29_engineering_pitfalls | 坑：原子序与原子成本 |
| p8_volatile_not_atomic | 29_engineering_pitfalls | 坑：volatile ≠ 线程安全 |
| p9_hugepage_verify | 29_engineering_pitfalls | 坑：madvise 大页不生效 |

## 说明

- 所有实验默认 Release、预热、多轮，输出 mean/median/min/max/stddev 与 checksum。
- 使用固定随机种子，保证可复现。
- 涉及特定指令集/大页/NUMA 时，程序先检测环境，不支持则输出提示并跳过，不会制造 Illegal Instruction。
- NUMA 实验仅在 `-DENABLE_NUMA_EXAMPLES=ON` 且检测到 libnuma 时编译。
