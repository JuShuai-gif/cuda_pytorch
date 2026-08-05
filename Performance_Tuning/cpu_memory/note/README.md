# 项目介绍

本项目把 Ulrich Drepper 的经典论文《What Every Programmer Should Know About Memory》
（Version 1.0，2007 年 11 月）整理为一套系统化的中文学习项目：

- **note/**：34 篇中文笔记，从 CPU 内存体系基础到 Linux 性能分析，区分 PDF 原文观点与现代补充；
- **src/**：28 组可编译可运行的 C++17 实验，覆盖缓存、TLB、NUMA、多线程、内存带宽等主题；
- **scripts/**：构建、运行、perf、Cachegrind、NUMA、系统信息探测脚本；
- **benchmark_results/**：本机实际运行产生的 Benchmark 数据与图（不编造）。

本项目不是 PDF 的翻译，而是"结构化教程 + 可复现实验 + 工具实践"。

# PDF 信息

| 项 | 值 |
|---|---|
| 标题 | What Every Programmer Should Know About Memory |
| 作者 | Ulrich Drepper (Red Hat, Inc.) |
| 版本 | Version 1.0 |
| 撰写时间 | 2007 年 |
| PDF 页数 | 114 页 |
| 文件位置 | cuda_pytorch/Performance_Tuning/cpumemory.pdf |

> 历史背景：论文面向 2007 年前后的 DDR/DDR2、FSB、南北桥、早期多核/超线程与 NUMA 起步期硬件。
> 笔记中所有此类描述均标注为历史背景，并补充现代实现，不冒充 PDF 原文。

# 适合人群

- 想理解 CPU Cache、DRAM、TLB、NUMA 如何影响程序性能的开发者；
- 写 C/C++（或任何系统级语言）并关心性能的程序员；
- Linux 系统性能工程师、运维/SRE 中对内存子系统感兴趣者；
- 准备面试（系统设计、性能优化、底层原理方向）的学习者；
- 希望掌握 perf、Cachegrind、numactl 等工具实操的人。

前置要求：会读 C/C++ 代码（理解指针、结构体、循环即可），能在 Linux 终端执行命令。

# 前置知识

- C/C++ 基础：指针、结构体、数组、编译与链接；
- Linux 基础命令：编译（g++/cmake）、运行、查看系统信息；
- 基本操作系统概念：进程、线程、虚拟地址空间（了解概念即可，深入内容笔记中会讲）；
- 一点计算机组成常识：什么是 CPU、寄存器、内存、总线。

不需要预先熟悉汇编或微架构细节，相关内容会在笔记中逐步建立。

# 项目目录

```
cpu_memory/
├── note/                         # 中文学习笔记（35 篇）
│   ├── README.md                 # 本文件
│   ├── 00_全书导读.md             # 章节/实验映射与学习路线总纲
│   ├── 01_CPU与内存系统总览.md
│   ├── ...（02~33）
│   ├── 34_项目完成报告.md         # 阶段四产出
│   └── 35_工程实战中的坑.md       # 工作中的高频坑（对应实验 29）
├── src/                          # C++17 实验源码
│   ├── CMakeLists.txt
│   ├── README.md
│   ├── common/                   # 公共组件：benchmark / cpu_info / statistics
│   ├── 01_memory_latency/ ... 28_integrated_project/
│   └── 29_engineering_pitfalls/  # 工程实战坑（p1~p9）
├── scripts/                      # 构建与性能分析脚本
│   ├── build.sh
│   ├── run_all.sh
│   ├── benchmark_all.sh
│   ├── perf_stat.sh
│   ├── perf_record.sh
│   ├── cachegrind.sh
│   ├── numa_test.sh
│   └── system_info.sh
└── benchmark_results/            # 本机 Benchmark 数据与图
```

# 建议学习顺序

1. 先读 note/00_全书导读.md，了解整体结构；
2. 沿"路线一"（01→15）建立 CPU 内存体系基础；
3. 沿"路线二"（18→25）学习 C++ 内存优化；
4. 沿"路线三"（26→29）学习 Linux 性能分析工具；
5. 做 31 检查清单、32 综合实践项目，复盘 33 术语表；
6. 最后读 35 工程实战中的坑，配合实验 29 验证"原理被误用时会发生什么"。

# 笔记索引

| 编号 | 主题 | 对应的 PDF 章节 |
|---|---|---|
| 00 | 全书导读 | 全文 |
| 01 | CPU 与内存系统总览 | 1, 2, 2.3 |
| 02 | SRAM 与 DRAM | 2.1, 2.1.1, 2.1.2, C |
| 03 | DRAM 访问原理 | 2.1.3, 2.2~2.2.3, 2.2.5 |
| 04 | 内存控制器与内存带宽 | 2.2.4, 2.3 |
| 05 | CPU Cache 基础 | 3, 3.1, 3.3.2 |
| 06 | Cache Line 与局部性 | 3.2 |
| 07 | Cache 映射与关联度 | 3.3, 3.3.1, 3.3.5 |
| 08 | Cache 写策略与一致性 | 3.3.3 |
| 09 | 多核缓存一致性 | 3.3.4 |
| 10 | Instruction Cache | 3.4, 3.4.1, 6.2.2, 7.4 |
| 11 | 缓存未命中与内存墙 | 3.5~3.5.4 |
| 12 | 虚拟内存与 MMU | 4, 4.4 |
| 13 | 多级页表与地址翻译 | 4.1, 4.2 |
| 14 | TLB 原理与优化 | 4.3, 4.3.1, 4.3.2 |
| 15 | 大页与 Huge Page | 6.2.4, 7.5 |
| 16 | NUMA 架构 | 5.1, 5.4 |
| 17 | Linux NUMA 支持 | 5.2, 5.3 |
| 18 | 缓存访问优化 | 6.2, 6.2.1, 6.2.3 |
| 19 | 数据布局优化 AoS 与 SoA | 6.2.1 部分 |
| 20 | 矩阵分块与 Cache Blocking | 6.2.1 部分, A.1 |
| 21 | 硬件与软件预取 | 6.3 |
| 22 | Non-Temporal Store | 6.1 |
| 23 | 多线程与 False Sharing | 6.4, 6.4.1, 6.4.3, A.3 |
| 24 | 原子操作与缓存竞争 | 6.4.2, 8.1, 8.2.1 |
| 25 | NUMA 编程实践 | 6.5, D |
| 26 | 内存性能分析工具 | 7 |
| 27 | perf 内存事件分析 | 7.1, B |
| 28 | Cachegrind 与缓存模拟 | 7.2, 7.3 |
| 29 | Page Fault 分析 | 7.5 |
| 30 | 现代 CPU 内存系统补充 | 8 + 现代知识 |
| 31 | 性能优化检查清单 | 综合 |
| 32 | 综合实践项目 | 综合, A |
| 33 | 术语表 | 综合 |
| 34 | 项目完成报告 | 阶段四产出 |
| 35 | 工程实战中的坑 | 综合 + 实验 29 |

# 实验索引

| 实验目录 | 主题 | 关联笔记 |
|---|---|---|
| src/01_memory_latency | 内存层级延迟（L1/L2/LLC/DRAM + pointer chasing） | 05, 11 |
| src/02_sequential_random_access | 顺序/逆序/步长/随机访问 | 06, 11 |
| src/03_cache_line_size | Cache Line 大小与利用率 | 06 |
| src/04_stride_access | Stride 访问与 TLB 效应 | 06, 14 |
| src/05_cache_capacity | 工作集大小与缓存容量 | 05, 11 |
| src/06_cache_associativity | 关联度与 Set 冲突 | 07 |
| src/07_cache_conflict | Cache 冲突未命中 | 07 |
| src/08_write_back_behavior | 写回/写通行为 | 08 |
| src/09_matrix_traversal | 矩阵行/列遍历与转置 | 19 |
| src/10_cache_blocking | Cache Blocking 分块 | 20 |
| src/11_aos_soa | AoS vs SoA | 19 |
| src/12_pointer_chasing | 链表与指针追逐 | 06, 12 |
| src/13_false_sharing | False Sharing | 23 |
| src/14_atomic_contention | 原子操作竞争 | 24 |
| src/15_thread_affinity | 线程亲和性 | 16, 25 |
| src/16_prefetch | 硬件/软件预取 | 21 |
| src/17_non_temporal_store | Non-Temporal Store | 22 |
| src/18_tlb_capacity | TLB 容量 | 14 |
| src/19_page_size | 页面大小 | 15 |
| src/20_huge_pages | Huge Page / THP | 15 |
| src/21_page_fault | Page Fault 分析 | 29 |
| src/22_memory_mapping | mmap / read | 13, 29 |
| src/23_memory_bandwidth | STREAM 类带宽 | 11, 23 |
| src/24_numa_local_remote | NUMA 本地/远程访问 | 16, 25 |
| src/25_numa_first_touch | NUMA First-Touch | 17, 25 |
| src/26_numa_replication | NUMA 数据复制 | 25 |
| src/27_instruction_cache | 指令缓存与分支 | 10 |
| src/28_integrated_project | 综合实践（矩阵乘法优化链） | 20, 32 |
| src/29_engineering_pitfalls | 工程实战坑（p1~p9） | 35 |

# 编译方法

```bash
# 默认 Release 构建（使用脚本）
./scripts/build.sh

# 手动构建
cd src
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 常用构建类型：Debug / Release / RelWithDebInfo
# 可选开关（默认关闭）：
#   -DENABLE_NATIVE_OPTIMIZATION=ON    # -march=native
#   -DENABLE_AVX2_EXAMPLES=ON          # AVX2 示例
#   -DENABLE_AVX512_EXAMPLES=ON        # AVX-512 示例
#   -DENABLE_NUMA_EXAMPLES=ON          # 链接 libnuma 的 NUMA 示例
```

要求：C++17 编译器（GCC/Clang 均可），CMake >= 3.12，Linux + pthread。
推荐编译参数：`-Wall -Wextra -Wpedantic -O3 -g -fno-omit-frame-pointer`。

# 运行方法

```bash
# 运行单个实验
./build/01_memory_latency/memory_latency

# 运行全部实验（跳过硬件不支持项）
./scripts/run_all.sh

# 一键跑基准
./scripts/benchmark_all.sh
```

所有实验：默认使用 Release、预热、多轮、输出平均值/中位数/最小值/标准差、固定随机种子、
输出 checksum、防止编译器优化掉测试代码。基准数据写入 benchmark_results/。

# perf 使用方法

```bash
# 系统信息探测（先了解你的 CPU/缓存/NUMA）
./scripts/system_info.sh

# 统计关键事件（自动容错，不因事件不支持而失败）
./scripts/perf_stat.sh ./build/01_memory_latency/memory_latency

# 记录并反汇编热点
./scripts/perf_record.sh ./build/09_matrix_traversal/matrix_traversal
```

需要 root（或 `perf_event_paranoid` 允许）才能读取部分硬件事件。
事件名因 CPU 而异（Intel/AMD 不同），脚本会自动尝试并在不支持时跳过，不硬编码假设。

# Cachegrind 使用方法

```bash
./scripts/cachegrind.sh ./build/01_memory_latency/memory_latency
```

若未安装 Valgrind，脚本会输出安装提示：
`sudo apt install valgrind`（Debian/Ubuntu）或 `sudo dnf install valgrind`（Fedora）。

# NUMA 实验方法

```bash
./scripts/numa_test.sh          # 检测并测试 NUMA
numactl --hardware              # 查看 NUMA 拓扑
```

本机为单 NUMA 节点时，脚本自动跳过远程访问测试并明确提示，
绝不声称"NUMA 测试成功"。多节点机器上可使用：

```bash
numactl --cpunodebind=0 --membind=0 ./build/24_numa_local_remote/numa_local_remote
numactl --interleave=all ./build/24_numa_local_remote/numa_local_remote
```

# 常见权限问题

| 现象 | 原因 | 解决 |
|---|---|---|
| perf 报 "Operation not permitted" | perf_event_paranoid 限制 | 提高 `kernel.perf_event_paranoid` 或用 root |
| dmesg 不可读 | kernel.dmesg_restrict | 需要 root |
| Huge Page 无法分配 | 未预留 HugePage 或权限不足 | 参照 note/15，root 下 `sysctl vm.nr_hugepages` |
| libnuma 链接失败 | 未安装 | 不启用 ENABLE_NUMA_EXAMPLES 即可，不影响整体编译 |
| NUMA 实验无远程数据 | 单节点机器 | 属预期行为，脚本会提示，不编造结果 |

# 学习路线

### 路线一：CPU 内存体系基础

```
RAM → Cache → Cache Line → 局部性 → 虚拟内存 → TLB
```
对应笔记：01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → 11 → 12 → 13 → 14 → 15
对应实验：01, 02, 03, 04, 05, 06, 07, 08, 12, 18, 19, 20

### 路线二：C++ 内存性能优化

```
顺序访问 → 数据布局 → Cache Blocking → Prefetch → False Sharing
```
对应笔记：18 → 19 → 20 → 21 → 22 → 23 → 24 → 25
对应实验：09, 10, 11, 13, 14, 15, 16, 17, 24, 25, 26, 28

### 路线三：Linux 系统性能分析

```
perf → Cachegrind → Page Fault → Huge Page → NUMA → CPU Affinity
```
对应笔记：26 → 27 → 28 → 29 → 15 → 16 → 17 → 25
对应脚本：system_info.sh, perf_stat.sh, perf_record.sh, cachegrind.sh, numa_test.sh

# 质量约定

- 笔记中 PDF 观点均标注页码与原始章节，不大量复制原文；
- 历史背景与现代补充显式区分；
- 每个实验可编译、可运行、带 checksum、多轮统计、不被编译器优化掉；
- Benchmark 数据只来自本机实际运行，不编造性能提升百分比；
- 无法验证的信息标注"当前环境或资料未验证"。
