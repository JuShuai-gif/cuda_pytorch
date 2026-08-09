# CPU微架构与计算瓶颈

本章合并CPU监控、perf、FlameGraph、VTune、热点、Cache、Branch、TLB、Bandwidth、AoS/SoA和SIMD。

## CPU Hotspot

Demo：`20_cpu_hotspot_bad_good`。hot/medium/cold被设计成明显采样层级。

```bash
/usr/bin/time -v ./src/build/20_cpu_hotspot_bad_good
perf stat ./src/build/20_cpu_hotspot_bad_good
perf record -F 99 -g -- ./src/build/20_cpu_hotspot_bad_good bad
perf report
perf annotate
```

perf report中“70% hot_function”表示约70%采样被归因到该函数/调用栈，不代表每次调用耗时占70%。FlameGraph横轴是聚合栈排列，不是时间轴；宽度是样本占比，纵轴是调用深度。

## Cache Locality

Demo：`21_cache_locality_bad_good`，同时比较Sequential/Random与Row/Column-major。

```bash
perf stat -e cycles,instructions,cache-references,cache-misses ./src/build/21_cache_locality_bad_good
valgrind --tool=cachegrind ./src/build/21_cache_locality_bad_good
```

Cache Miss Rate = misses / references。IPC低 + LLC Miss高 + DRAM带宽高，才较强支持Memory Bound。随机访问破坏空间局部性；重复复用体现时间局部性。

## Branch Miss

随机条件应预先生成，不能把RNG放进核心计时区。

```bash
perf stat -e cycles,instructions,branches,branch-misses ./src/build/03_branch_miss
```

Branch miss导致错误推测路径被清空。IPC低 + branch miss高 + DRAM不高，支持branch/frontend候选。CPU branch prediction与CUDA warp divergence不是同一机制。

## TLB与Page Fault

Demo：`22_tlb_pagefault`。

- Cache miss：数据不在目标cache层。
- TLB miss：虚拟地址翻译缓存未命中，可能page walk。
- Page fault：页表映射/驻留处理；minor通常不需磁盘，major通常涉及IO。

```bash
perf list | grep -i tlb
perf stat -e page-faults,minor-faults,major-faults ./src/build/22_tlb_pagefault
```

只能使用`perf list`实际存在的TLB事件。Large Tensor、KV Cache、模型权重和长期buffer都可能放大TLB压力。

## STREAM与Roofline

`23_stream_bad_good`实现Copy、Scale、Add、Triad并输出GB/s。Vector Add/Triad低算术强度，通常偏Memory Bound。

```bash
OMP_NUM_THREADS=1 ./src/build/23_stream_bad_good
OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores ./src/build/23_stream_bad_good
```

Roofline：`Performance ≤ min(Peak FLOPS, AI × Memory Bandwidth)`，其中`AI=FLOPs/Bytes`。大GEMM通常AI高；Vector Add、RMSNorm、KV cache常偏带宽；具体结论依shape、dtype、实现和硬件。

## AoS、SoA与SIMD

只处理Particle的x/y时，AoS会流入未使用字段，SoA更利于cache和vectorization。

```bash
./src/build/24_aos_soa
g++ -O3 -march=native -fopt-info-vec -fopt-info-vec-missed src/cpp/13_simd.cpp
objdump -d -C ./src/build/13_simd
```

x86关注SSE/AVX/AVX2/AVX-512，Arm/Jetson关注NEON/SVE。`-march=native`只适合本机实验，不应作为可分发二进制默认假设。

## CPU流水线诊断视角

现代CPU通过取指、解码、重命名、乱序调度、执行和退休形成流水线。常见归因是Front-End Bound、Back-End Core Bound、Back-End Memory Bound、Bad Speculation和Retiring。IPC低只说明每周期退休指令少，不能区分根因。

## perf stat的科学用法

```bash
perf stat -r 5 -d -d -d ./src/build/21_cache_locality_bad_good
perf stat -e cycles,instructions,branches,branch-misses ./src/build/03_branch_miss
```

采集事件过多会multiplex。混合核心CPU可能分别报告cpu_core和cpu_atom，不同PMU计数不能直接混合计算IPC；可用taskset固定同类核心。

## Inclusive与Self热点

Inclusive包含子调用树，Self只归因函数本体。pipeline inclusive高但self低，说明工作在子函数。

```bash
cmake -S src -B build-fp -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS=-fno-omit-frame-pointer
perf record -F 199 -g -- ./build-fp/20_cpu_hotspot_bad_good bad
```

## Cache层级与工作集

L1小而快，L2更大，LLC通常跨核心共享，最后是DRAM。工作集超过容量不等于每次miss；复用距离、映射冲突、prefetch和内存级并行同样重要。

高miss rate可能只是流式访问；低miss rate也可能因少量高延迟miss受限。联合观察LLC miss绝对量、DRAM带宽、backend stall、IPC和布局A/B。

## TLB与Large Tensor

4KiB页下1GiB内存涉及262144页，远超TLB容量。Huge Page扩大TLB reach，但增加内存浪费和部署复杂度。模型权重顺序扫描通常比随机KV索引易预取；大型ring buffer随机跨页也会产生page-walk压力。

## Branch优化不是简单去掉if

策略包括数据排序、查表、算术选择、拆罕见路径、SIMD mask或conditional move。Branchless可能执行原本不需要的昂贵工作，必须A/B。

## STREAM字节数与AI

| Kernel | 数据流 | FLOPs/元素 | 近似AI(double) |
|---|---:|---:|---:|
| Copy | 读1写1 | 0 | 0 |
| Scale | 读1写1 | 1 | 1/16 |
| Add | 读2写1 | 1 | 1/24 |
| Triad | 读2写1 | 2 | 2/24 |

真实流量还受write allocate、non-temporal store和cache命中影响。

## OpenMP扩展

```bash
for t in 1 2 4 8 16; do
  OMP_NUM_THREADS=$t OMP_PROC_BIND=close OMP_PLACES=cores ./src/build/23_stream_bad_good
done
```

带宽程序常在少量核心后饱和；继续加线程只增加争用。Compute程序还受串行比例、负载不均衡、NUMA与turbo影响。

## AoS、SoA与AoSoA

PointCloud只筛选x/y/z时SoA更友好；总是使用完整机器人状态时AoS可能自然；AoSoA按SIMD宽度分块，可兼顾局部性和组织。

## SIMD验证

```bash
g++ -O3 -march=native -fopt-info-vec-optimized -fopt-info-vec-missed src/cpp/13_simd.cpp -o /tmp/simd
objdump -d -C /tmp/simd | less
```

常见阻碍包括指针别名、循环依赖、非连续访问、复杂分支、对齐和浮点语义。不能仅凭O3断言已向量化。

## 完整CPU诊断案例

```text
症状：单核100%，吞吐低
perf stat：IPC 0.6，branch miss低，LLC miss高
带宽：接近STREAM实测上限
perf record：随机索引聚合是热点
假设：Memory Bound
优化：数据重排为连续块
复验：runtime下降、IPC上升、LLC miss下降、checksum一致
```
