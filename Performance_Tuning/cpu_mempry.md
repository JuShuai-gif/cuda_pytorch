你现在是一名精通以下领域的高级系统性能工程师：

- C/C++ 性能优化
- CPU 微架构
- CPU Cache
- DRAM 与内存控制器
- 虚拟内存、页表和 TLB
- Cache Coherence
- NUMA
- Linux 内存管理
- Linux perf
- Valgrind/Cachegrind
- 多线程与伪共享
- SIMD、预取和内存带宽优化

当前项目目录中有一个 PDF：

/home/hpc/ghr_code/cuda_pytorch/Performance_Tuning/cpumemory.pdf

PDF 名称为：

What Every Programmer Should Know About Memory

作者：Ulrich Drepper。

请完整分析这份 PDF，并把它整理为一套系统化的中文学习项目。

要求：

- Markdown 学习笔记放在 note/ 目录；
- C++ 实验代码放在 src/ 目录；
- 构建和性能分析脚本放在 scripts/ 目录；
- Benchmark 结果放在 benchmark_results/ 目录。

不要简单翻译 PDF，而是把它整理成：

1. 系统化中文教程；
2. CPU 内存体系结构说明；
3. 可编译、可运行的 C++ 实验；
4. 可复现的性能测试；
5. perf、Cachegrind、numactl 等工具实践；
6. 从初学者到系统性能工程师的学习路线。

==================================================
一、项目目录
==================================================

请在 /home/hpc/ghr_code/cuda_pytorch/Performance_Tuning/cpu_memory 建立以下目录：

├── note/
│   ├── README.md
│   ├── 00_全书导读.md
│   ├── 01_CPU与内存系统总览.md
│   ├── 02_SRAM与DRAM.md
│   ├── 03_DRAM访问原理.md
│   ├── 04_内存控制器与内存带宽.md
│   ├── 05_CPU_Cache基础.md
│   ├── 06_Cache_Line与局部性.md
│   ├── 07_Cache映射与关联度.md
│   ├── 08_Cache写策略与一致性.md
│   ├── 09_多核缓存一致性.md
│   ├── 10_Instruction_Cache.md
│   ├── 11_缓存未命中与内存墙.md
│   ├── 12_虚拟内存与MMU.md
│   ├── 13_多级页表与地址翻译.md
│   ├── 14_TLB原理与优化.md
│   ├── 15_大页与Huge_Page.md
│   ├── 16_NUMA架构.md
│   ├── 17_Linux_NUMA支持.md
│   ├── 18_缓存访问优化.md
│   ├── 19_数据布局优化_AoS与SoA.md
│   ├── 20_矩阵分块与Cache_Blocking.md
│   ├── 21_硬件与软件预取.md
│   ├── 22_Non_Temporal_Store.md
│   ├── 23_多线程与False_Sharing.md
│   ├── 24_原子操作与缓存竞争.md
│   ├── 25_NUMA编程实践.md
│   ├── 26_内存性能分析工具.md
│   ├── 27_perf内存事件分析.md
│   ├── 28_Cachegrind与缓存模拟.md
│   ├── 29_Page_Fault分析.md
│   ├── 30_现代CPU内存系统补充.md
│   ├── 31_性能优化检查清单.md
│   ├── 32_综合实践项目.md
│   └── 33_术语表.md
│
├── src/
│   ├── CMakeLists.txt
│   ├── README.md
│   │
│   ├── common/
│   │   ├── benchmark.h
│   │   ├── benchmark.cpp
│   │   ├── cpu_info.h
│   │   ├── cpu_info.cpp
│   │   ├── compiler_barrier.h
│   │   ├── statistics.h
│   │   └── statistics.cpp
│   │
│   ├── 01_memory_latency/
│   ├── 02_sequential_random_access/
│   ├── 03_cache_line_size/
│   ├── 04_stride_access/
│   ├── 05_cache_capacity/
│   ├── 06_cache_associativity/
│   ├── 07_cache_conflict/
│   ├── 08_write_back_behavior/
│   ├── 09_matrix_traversal/
│   ├── 10_cache_blocking/
│   ├── 11_aos_soa/
│   ├── 12_pointer_chasing/
│   ├── 13_false_sharing/
│   ├── 14_atomic_contention/
│   ├── 15_thread_affinity/
│   ├── 16_prefetch/
│   ├── 17_non_temporal_store/
│   ├── 18_tlb_capacity/
│   ├── 19_page_size/
│   ├── 20_huge_pages/
│   ├── 21_page_fault/
│   ├── 22_memory_mapping/
│   ├── 23_memory_bandwidth/
│   ├── 24_numa_local_remote/
│   ├── 25_numa_first_touch/
│   ├── 26_numa_replication/
│   ├── 27_instruction_cache/
│   └── 28_integrated_project/
│
├── scripts/
│   ├── build.sh
│   ├── run_all.sh
│   ├── benchmark_all.sh
│   ├── perf_stat.sh
│   ├── perf_record.sh
│   ├── cachegrind.sh
│   ├── numa_test.sh
│   └── system_info.sh
│
└── benchmark_results/

==================================================
二、执行原则
==================================================

不要一次性生成所有文件。

必须分阶段执行。

第一阶段只完成：

1. 完整读取 PDF 目录和章节结构；
2. 确定每个原始章节的起止页；
3. 建立“PDF 章节 → note 文件”的映射；
4. 建立“知识点 → src 实验”的映射；
5. 创建目录结构；
6. 编写 note/00_全书导读.md；
7. 编写 note/README.md；
8. 创建详细任务清单；
9. 暂时不要生成所有代码。

把章节映射整理成表格：

| PDF章节 | PDF页码 | 核心主题 | 对应笔记 | 对应实验 |
|---|---:|---|---|---|

在没有完成全书结构分析之前，不要批量生成笔记。

==================================================
三、内容来源规则
==================================================

所有内容分为三类。

### 1. PDF 原始内容

来自 PDF 的观点必须标注：

> 来源：PDF 第 XX～XX 页  
> 对应章节：6.2.1 Optimizing Level 1 Data Cache Access

不得大段复制 PDF 原文。

必须用自己的语言总结，避免逐句翻译。

### 2. 原理解释

可以补充必要的计算机体系结构知识，但必须围绕 PDF 内容展开。

### 3. 现代补充

这份 PDF 编写时间较早，其中可能出现以下历史背景：

- Northbridge；
- Southbridge；
- Front Side Bus；
- DDR、DDR2；
- 早期多核处理器；
- OProfile；
- 旧版 Linux 内核接口；
- 早期 NUMA 系统；
- 早期 Cache 层级结构。

涉及这些内容时必须说明：

> 历史背景：以下描述对应 PDF 编写时期的硬件。

然后补充：

> 现代补充：现代 CPU 通常使用集成内存控制器，并具有不同的 Cache、互连和 NUMA 架构。

不得直接把现代补充冒充成 PDF 原文。

无法确认现代信息时，不得编造，应标记：

> 当前环境或资料未验证。

==================================================
四、笔记编写规范
==================================================

每个 note/*.md 至少包含以下内容：

# 标题

## 1. 本章要解决的问题

说明这一章解决什么性能问题。

## 2. 前置知识

列出理解本章需要的基础概念。

## 3. 核心概念

解释术语，例如：

- Cache Line
- Spatial Locality
- Temporal Locality
- Associativity
- Cache Set
- Cache Tag
- Dirty Cache Line
- Write Back
- Write Through
- TLB
- Page Table
- Page Fault
- NUMA Node
- Memory Policy
- False Sharing

## 4. 硬件工作流程

使用 Mermaid、ASCII 图或文字流程图描述数据流。

例如：

CPU Core
   ↓ Load
L1 Data Cache
   ↓ Miss
L2 Cache
   ↓ Miss
Last Level Cache
   ↓ Miss
Memory Controller
   ↓
DRAM

## 5. PDF 核心观点

按照 PDF 内容总结，并标注页码和原始章节。

## 6. 通俗解释

用适合刚开始学习 CPU 内存体系的读者能够理解的语言解释。

## 7. 示例分析

给出简单数据或地址示例。

例如：

- 为什么读取 int 会加载整个 Cache Line；
- 为什么连续访问比随机访问快；
- 为什么按行遍历矩阵通常比按列遍历快；
- 为什么链表遍历容易发生 Cache Miss；
- 为什么两个线程修改不同变量也可能互相影响。

## 8. 未优化代码

展示常见低性能实现。

## 9. 优化后代码

展示对应优化实现。

## 10. 为什么会更快

必须从以下角度分析：

- Cache Line 利用率；
- L1/L2/LLC 命中率；
- DRAM 访问次数；
- 硬件预取；
- TLB Miss；
- Page Fault；
- Cache Coherence；
- False Sharing；
- 内存带宽；
- NUMA Local/Remote Access；
- 指令级并行；
- 编译器优化。

## 11. 如何验证

给出：

- 编译命令；
- 运行命令；
- perf stat；
- perf record；
- Cachegrind；
- numactl；
- taskset；
- 系统信息查看命令。

## 12. 实验结果应该怎么看

解释指标含义，不得只展示数字。

## 13. 常见误区

说明该优化在哪些场景可能无效或变慢。

## 14. 实践练习

提供 3～5 个练习。

## 15. 本章总结

总结最重要的结论。

## 16. 对应代码

列出 src 中对应代码路径。

==================================================
五、必须实现的实验
==================================================

所有实验使用 C++17，在 Linux 环境下运行。

每个实验目录建议包含：

experiment_name/
├── baseline.cpp
├── optimized.cpp
├── benchmark.cpp
├── README.md
└── CMakeLists.txt

--------------------------------
实验 1：内存层级延迟
--------------------------------

测试：

- 寄存器或局部计算；
- L1 Cache；
- L2 Cache；
- Last Level Cache；
- DRAM。

使用不同大小的工作集观察延迟台阶。

注意：

普通数组遍历容易被硬件预取影响，因此应同时实现 pointer chasing 实验。

--------------------------------
实验 2：顺序访问与随机访问
--------------------------------

比较：

- 顺序读取；
- 逆序读取；
- 固定步长读取；
- 随机读取。

输出：

- 总耗时；
- ns/element；
- GB/s；
- checksum。

--------------------------------
实验 3：Cache Line
--------------------------------

研究：

- Cache Line 大小；
- 每次访问 4、8、16、32、64、128 字节；
- Cache Line 利用率；
- 相邻数据预取。

读取系统 Cache 信息：

/sys/devices/system/cpu/cpu0/cache/

不得假定所有 CPU 的 Cache Line 都一定是 64 字节。

--------------------------------
实验 4：Stride Access
--------------------------------

测试步长：

1、2、4、8、16、32、64、128、256 个元素。

观察：

- Cache Miss；
- TLB Miss；
- 有效内存带宽。

--------------------------------
实验 5：Cache Capacity
--------------------------------

工作集大小从几 KB 增长到几百 MB。

例如：

4KB
8KB
16KB
32KB
64KB
128KB
256KB
512KB
1MB
2MB
4MB
8MB
16MB
32MB
64MB
128MB

绘制或输出：

- 工作集大小；
- 平均访问延迟；
- cycles/access；
- cache-misses。

--------------------------------
实验 6：Cache Associativity
--------------------------------

构造多个映射到相同 Cache Set 的地址。

观察：

- Associativity；
- Conflict Miss；
- 容量未满但仍发生 Cache Miss 的情况。

如果无法可靠控制物理地址映射，必须明确说明实验限制。

--------------------------------
实验 7：矩阵访问
--------------------------------

比较：

- 按行遍历；
- 按列遍历；
- 矩阵转置；
- Cache Blocking；
- 不同 Block Size。

Block Size 至少测试：

8、16、32、64、128。

--------------------------------
实验 8：AoS 与 SoA
--------------------------------

实现粒子数据：

struct Particle {
    float x;
    float y;
    float z;
    float velocity;
    float mass;
};

比较：

- Array of Structures；
- Structure of Arrays；
- 只访问部分字段；
- 访问全部字段。

分析：

- Cache Line 利用率；
- SIMD 友好程度；
- 内存带宽。

--------------------------------
实验 9：链表与连续数组
--------------------------------

比较：

- std::vector；
- std::list；
- 随机排列的节点；
- 连续节点池；
- pointer chasing。

说明链表慢的原因不只是指针操作，还包括：

- 空间局部性差；
- Cache Miss；
- TLB Miss；
- 无法有效预取；
- 节点分配开销。

--------------------------------
实验 10：False Sharing
--------------------------------

两个线程分别更新两个不同计数器。

实现：

1. 两个计数器位于同一 Cache Line；
2. 使用 alignas 或 padding 分离；
3. 比较运行时间；
4. 使用 perf 观察相关事件。

不得直接假定 padding 为 64 就一定适合所有 CPU。

应读取或说明目标平台的 Cache Line 大小。

--------------------------------
实验 11：Atomic Contention
--------------------------------

比较：

- 普通局部变量；
- std::atomic；
- 多线程共享 atomic；
- 每线程局部计数后 reduction；
- mutex。

解释：

- Cache Line 所有权转移；
- 原子 RMW；
- Cache Coherence Traffic；
- 可扩展性。

--------------------------------
实验 12：Thread Affinity
--------------------------------

使用：

- pthread_setaffinity_np；
- sched_setaffinity；
- taskset。

比较：

- 线程固定核心；
- 不固定核心；
- 同一物理核心的 SMT 线程；
- 不同物理核心。

如果无法识别物理核心拓扑，必须输出提示，不得伪造结论。

--------------------------------
实验 13：Prefetch
--------------------------------

比较：

- 无预取；
- 硬件预取可识别的顺序访问；
- __builtin_prefetch；
- _mm_prefetch；
- 不同预取距离；
- 随机访问。

说明：

软件预取不一定更快，预取距离错误可能导致性能下降或 Cache Pollution。

--------------------------------
实验 14：Non-Temporal Store
--------------------------------

实现：

- 普通 Store；
- Streaming Store；
- 大数组连续写入；
- 写入后立即读取；
- 写入后很久才读取。

在支持的 x86 平台使用适当 Intrinsics。

必须：

- 进行 CPU 指令集检查；
- 对内存进行正确对齐；
- 使用必要的内存屏障；
- 不支持时自动跳过。

--------------------------------
实验 15：TLB Capacity
--------------------------------

通过每页访问一个元素测试：

- 不同页面数量；
- 4KB 页面；
- 工作集超过 TLB 容量后的变化；
- dTLB-load-misses。

避免顺序访问被硬件预取完全掩盖。

--------------------------------
实验 16：Huge Page
--------------------------------

至少介绍并在环境允许时测试：

- Transparent Huge Pages；
- mmap；
- madvise；
- MADV_HUGEPAGE；
- 显式 HugeTLB。

程序不能假设系统一定配置了 Huge Page。

运行前检查：

/proc/meminfo
/sys/kernel/mm/transparent_hugepage/

无法启用时输出清晰提示。

--------------------------------
实验 17：Page Fault
--------------------------------

比较：

- malloc 后不访问；
- 首次访问；
- 第二次访问；
- Minor Page Fault；
- Major Page Fault；
- MAP_POPULATE；
- madvise。

使用：

perf stat -e page-faults,minor-faults,major-faults

--------------------------------
实验 18：Memory Mapping
--------------------------------

比较：

- read；
- mmap；
- 顺序读取；
- 随机读取；
- 小文件；
- 大文件；
- 冷缓存与热缓存。

不得擅自清理系统 Page Cache。

需要 root 权限的操作只能给出提示，不能假定已执行。

--------------------------------
实验 19：内存带宽
--------------------------------

实现类似 STREAM 的简单实验：

- Copy；
- Scale；
- Add；
- Triad。

输出：

- 数据量；
- 运行时间；
- 有效带宽 GB/s。

必须保证数组大于 Last Level Cache，避免只测到 Cache 带宽。

--------------------------------
实验 20：NUMA Local 与 Remote
--------------------------------

在 NUMA 机器上比较：

- 本地节点分配；
- 远程节点分配；
- interleave；
- first-touch；
- CPU affinity；
- memory binding。

使用：

numactl --hardware
numactl --cpunodebind
numactl --membind
numactl --interleave

代码可选使用 libnuma。

CMake 必须检测 libnuma：

- 找到时编译 NUMA 实验；
- 未找到时跳过并显示提示；
- 不得导致整个项目编译失败。

在非 NUMA 机器上不得编造结果。

==================================================
六、Benchmark 规范
==================================================

所有性能实验必须：

1. 使用 Release 构建；
2. 运行预热；
3. 执行多轮；
4. 输出平均值；
5. 输出中位数；
6. 输出最小值；
7. 输出标准差；
8. 使用固定随机种子；
9. 输出 checksum；
10. 验证 baseline 与 optimized 结果一致；
11. 防止编译器删除测试代码；
12. 区分初始化时间和核心计算时间；
13. 不使用单次结果下结论；
14. 不编造性能提升百分比。

Benchmark 数据结构至少包含：

struct BenchmarkResult {
    double mean_ms;
    double median_ms;
    double min_ms;
    double max_ms;
    double stddev_ms;
};

可以使用：

- volatile；
- compiler barrier；
- asm volatile；
- checksum；
- 返回值；
- std::atomic_signal_fence。

但必须解释为什么使用这些机制。

==================================================
七、CMake 构建要求
==================================================

统一使用 src/CMakeLists.txt 管理所有实验。

要求：

- C++17；
- Debug；
- Release；
- RelWithDebInfo；
- GCC 和 Clang 均可；
- 开启警告；
- 保留调试符号；
- 保留 Frame Pointer；
- pthread；
- 可选 libnuma。

推荐参数：

-Wall
-Wextra
-Wpedantic
-O3
-g
-fno-omit-frame-pointer

不要默认强制使用：

-march=native
-mavx2
-mavx512f

应提供 CMake 选项：

ENABLE_NATIVE_OPTIMIZATION
ENABLE_NUMA_EXAMPLES
ENABLE_AVX2_EXAMPLES
ENABLE_AVX512_EXAMPLES

涉及特定指令集时：

1. 单独编译对应源码；
2. 运行时检测 CPU 能力；
3. 不支持时输出提示并跳过；
4. 不得造成 Illegal Instruction。

==================================================
八、性能分析工具
==================================================

scripts/system_info.sh 输出：

- uname -a；
- lscpu；
- numactl --hardware；
- CPU Cache 信息；
- 页面大小；
- Huge Page 信息；
- 编译器版本；
- 内核版本；
- 内存容量。

scripts/perf_stat.sh 支持：

./scripts/perf_stat.sh ./build/example

至少统计：

cycles
instructions
branches
branch-misses
cache-references
cache-misses
page-faults
minor-faults
major-faults
context-switches
cpu-migrations

根据平台支持情况尝试：

L1-dcache-loads
L1-dcache-load-misses
LLC-loads
LLC-load-misses
dTLB-loads
dTLB-load-misses

某些事件不受 CPU 或内核支持时，不得让脚本整体失败。

scripts/cachegrind.sh 使用：

valgrind --tool=cachegrind

如果没有安装 Valgrind，输出安装提示。

scripts/numa_test.sh：

1. 检查 numactl；
2. 检查 NUMA Node 数量；
3. 单节点机器自动跳过远程访问测试；
4. 不得声称 NUMA 测试成功。

==================================================
九、图表和示意图
==================================================

PDF 中包含大量结构图和性能曲线。

分析这些图时：

1. 说明图号；
2. 说明图所在页码；
3. 说明横轴和纵轴；
4. 说明实验变量；
5. 说明曲线转折点；
6. 解释转折点与 Cache、TLB、DRAM 的关系；
7. 不得只说“数值变大了”；
8. 不要直接复制整张原图；
9. 可以用 Mermaid 或 ASCII 重新绘制概念图；
10. 可以根据自己实际实验生成新的数据图。

自己生成的实验图必须注明：

> 本图由当前环境 Benchmark 数据生成，不是 PDF 原图。

==================================================
十、现代知识补充
==================================================

创建：

note/30_现代CPU内存系统补充.md

区分 PDF 的历史内容和现代实现。

可以补充：

- 集成内存控制器；
- DDR4 和 DDR5；
- 多通道内存；
- LLC；
- Inclusive、Exclusive、Non-inclusive Cache；
- MESI/MOESI；
- Intel UPI；
- AMD Infinity Fabric；
- Chiplet；
- 多 CCD/CCX；
- ARM 多核缓存；
- NUMA；
- Transparent Huge Pages；
- Linux perf；
- eBPF 内存分析；
- Intel VTune；
- AMD uProf。

所有现代补充必须明确标记为补充资料。

如果没有可靠资料或当前环境无法验证，不得编造具体延迟、带宽和 Cache 参数。

==================================================
十一、README 要求
==================================================

note/README.md 必须包含：

# 项目介绍

# PDF 信息

# 适合人群

# 前置知识

# 项目目录

# 建议学习顺序

# 笔记索引

# 实验索引

# 编译方法

# 运行方法

# perf 使用方法

# Cachegrind 使用方法

# NUMA 实验方法

# 常见权限问题

# 学习路线

建立三条学习路线：

### 路线一：CPU 内存体系基础

RAM → Cache → Cache Line → 局部性 → 虚拟内存 → TLB。

### 路线二：C++ 内存性能优化

顺序访问 → 数据布局 → Cache Blocking → Prefetch → False Sharing。

### 路线三：Linux 系统性能分析

perf → Cachegrind → Page Fault → Huge Page → NUMA → CPU Affinity。

==================================================
十二、质量检查
==================================================

每完成一篇笔记，检查：

- 是否标注 PDF 页码；
- 是否区分 PDF 原文观点和现代补充；
- 是否解释核心术语；
- 是否有对应实验；
- Markdown 链接是否有效；
- 是否存在大段原文复制。

每完成一个实验，检查：

- 是否可以编译；
- 是否可以运行；
- 是否有 checksum；
- baseline 和 optimized 是否结果一致；
- 是否执行了多轮；
- 是否避免被编译器优化掉；
- 是否记录了 CPU、内核和编译器环境；
- 是否没有编造 Benchmark 数据。

==================================================
十三、分阶段执行
==================================================

阶段一：

1. 阅读完整 PDF 目录；
2. 提取章节和页码；
3. 创建目录结构；
4. 创建 note/00_全书导读.md；
5. 创建 note/README.md；
6. 建立笔记和代码映射；
7. 输出后续任务清单。

阶段二：

按照 PDF 顺序编写笔记。

每次只处理一个主要章节，不要一次性处理整本 PDF。

阶段三：

按照依赖顺序实现实验：

1. Benchmark 公共组件；
2. 内存延迟；
3. 顺序与随机访问；
4. Cache Line；
5. Stride；
6. Cache Capacity；
7. Matrix；
8. AoS/SoA；
9. False Sharing；
10. TLB；
11. Page Fault；
12. NUMA。

阶段四：

执行最终验收：

1. 删除旧 build；
2. 从头配置 CMake；
3. 编译全部代码；
4. 运行全部可运行实验；
5. 自动跳过当前硬件不支持的实验；
6. 检查 Markdown 链接；
7. 检查空文件；
8. 检查 TODO；
9. 检查伪代码；
10. 检查错误的文件路径；
11. 检查未验证却声称成功的内容；
12. 生成：

note/34_项目完成报告.md

==================================================
十四、禁止事项
==================================================

禁止：

- 逐句翻译整本 PDF；
- 大段复制原文；
- 编造 PDF 内容；
- 编造硬件信息；
- 编造 Benchmark 结果；
- 编造性能提升比例；
- 把 2007 年的硬件结构当成现代 CPU 的唯一结构；
- 假设 Cache Line 一定是 64 字节；
- 假设系统一定支持 NUMA；
- 假设系统一定支持 Huge Page；
- 假设 perf 的所有事件都存在；
- 假设当前 CPU 支持 AVX2 或 AVX-512；
- 未执行编译就声称编译通过；
- 未执行程序就声称测试通过；
- 使用 Debug 数据得出性能结论；
- 仅凭一次运行得出性能结论。

现在开始执行阶段一。

首先完整读取 cpumemory.pdf 的目录和章节结构，然后创建：

1. note/00_全书导读.md
2. note/README.md
3. 完整的章节映射表
4. 完整的代码实验规划

本轮不要开始批量生成后续笔记和代码。





继续执行阶段二。

本轮只处理 PDF 第 3 章 CPU Caches。

要求：

1. 完整阅读第 3 章；
2. 分析其中所有重要图表；
3. 编写 note/05 至 note/11；
4. 标注 PDF 页码和原始小节；
5. 重点解释：
   - Cache Line
   - Cache Set
   - Tag
   - Associativity
   - Cache Miss
   - Write Back
   - Cache Coherence
   - Instruction Cache
6. 规划对应实验，但暂时不要处理虚拟内存和 NUMA。









继续执行阶段三。

本轮只实现：

src/common
src/01_memory_latency
src/02_sequential_random_access
src/03_cache_line_size
src/04_stride_access
src/05_cache_capacity
src/09_matrix_traversal
src/10_cache_blocking
src/12_pointer_chasing

要求：

1. 使用 C++17；
2. 接入总 CMakeLists.txt；
3. 实际执行 CMake 配置和编译；
4. 修复所有编译错误；
5. 实际运行；
6. 输出 checksum；
7. 每个 Benchmark 至少运行 10 轮；
8. 输出 mean、median、min、max、stddev；
9. 不得编造性能数据；
10. 将使用方法和结果解释写入对应 README.md。





现在进行整个项目的最终验收。

要求：

1. 删除 build 目录；
2. 重新执行 CMake 配置；
3. 编译全部实验；
4. 运行当前硬件支持的全部实验；
5. 检查 NUMA、Huge Page、AVX 和 perf 的兼容性；
6. 检查所有 Markdown 链接；
7. 搜索 TODO、FIXME、占位符和伪代码；
8. 检查每篇笔记的 PDF 页码来源；
9. 检查代码与笔记的对应关系；
10. 检查 Benchmark 是否可能被编译器消除；
11. 检查 baseline 与 optimized 结果是否一致；
12. 将实际验收结果写入 note/34_项目完成报告.md。

没有执行成功的内容必须明确标记为未验证，不得声称通过。