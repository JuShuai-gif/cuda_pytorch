# README —— C++ 性能优化教程（基于《Optimizing software in C++》）

## 1. 项目介绍

本教程基于 Agner Fog 的《Optimizing software in C++》整理而成，目标是让已掌握基础 C++、但对 CPU Cache、分支预测、流水线、乱序执行、指令延迟/吞吐、SIMD、编译器优化、perf 和 Benchmark 设计不熟悉的读者，能够系统地学习 C++ 性能优化。

教程由三部分组成：

- **`note/`**：19 篇中文技术笔记（含导读与最终验收报告），所有关键观点标注 PDF 页码；
- **`src/`**：18 组可编译运行的 C++17 实验，每组包含 baseline、optimized、benchmark；
- **`scripts/`**：构建、运行、批量 Benchmark、perf 分析的自动化脚本。

核心方法论（贯穿全书）：

> 先测量，再优化。不要对非热点代码进行无意义的微优化。

## 2. PDF 基本信息

| 项目 | 内容 |
|------|------|
| 书名 | Optimizing software in C++（An optimization guide for Windows, Linux, and Mac platforms） |
| 作者 | Agner Fog |
| 版权 | © 2004-2026，最后更新 2026-07-12 |
| 页数 | 179 页（PDF 页码 = 书中印刷页码） |
| 系列 | Agner Fog 五卷优化手册的第一卷 |
| 原文 | `../optimizing_cpp.pdf`（PDF 物理页码与印刷页码一致） |

## 3. 建议学习顺序

先读 `note/00_全书导读.md` 了解全书结构，然后按主题学习。推荐按下列顺序：

```
01 性能优化基本原则 → 02 平台与编译器选择 → 03 性能热点分析
→ 04 C++语言结构性能分析 → 05 编译器优化原理 → 06 内存与缓存优化
→ 07 多线程优化 → 08 乱序执行与指令级并行 → 09 SIMD与自动向量化
→ 10 Intrinsics编程 → 11 CPU指令集分发 → 12 专项优化技巧
→ 13 模板与编译期优化 → 14 性能测试与Benchmark → 15 嵌入式系统优化
→ 16 编译参数速查表 → 17 性能优化检查清单 → 18 学习路线与实践项目
```

每篇笔记对应一组可动手实验，建议"读完一篇 → 编译运行该篇实验 → 用 perf 验证 → 再读下一篇"。

## 4. 每篇笔记的链接

| 笔记 | 链接 |
|------|------|
| 00 全书导读（章节与实验映射） | [00_全书导读.md](00_全书导读.md) |
| 01 性能优化基本原则（PDF 第 1 章） | [01_性能优化基本原则.md](01_性能优化基本原则.md) ✅ |
| 02 平台与编译器选择（PDF 第 2 章） | [02_平台与编译器选择.md](02_平台与编译器选择.md) ✅ |
| 03 性能热点分析（PDF 第 3 章） | [03_性能热点分析.md](03_性能热点分析.md) ✅ |
| 04 C++语言结构性能分析（PDF 第 7 章） | [04_C++语言结构性能分析.md](04_C++语言结构性能分析.md) ✅ |
| 05 编译器优化原理（PDF 第 8 章） | [05_编译器优化原理.md](05_编译器优化原理.md) ✅ |
| 06 内存与缓存优化（PDF 第 9 章） | [06_内存与缓存优化.md](06_内存与缓存优化.md) ✅ |
| 07 多线程优化（PDF 第 10 章） | [07_多线程优化.md](07_多线程优化.md) ✅ |
| 08 乱序执行与指令级并行（PDF 第 11 章） | [08_乱序执行与指令级并行.md](08_乱序执行与指令级并行.md) ✅ |
| 09 SIMD与自动向量化（PDF 第 12 章） | [09_SIMD与自动向量化.md](09_SIMD与自动向量化.md) ✅ |
| 10 Intrinsics编程（PDF 第 12.4/12.5/12.8） | [10_Intrinsics编程.md](10_Intrinsics编程.md) ✅ |
| 11 CPU指令集分发（PDF 第 13 章） | [11_CPU指令集分发.md](11_CPU指令集分发.md) ✅ |
| 12 专项优化技巧（PDF 第 14 章） | [12_专项优化技巧.md](12_专项优化技巧.md) ✅ |
| 13 模板与编译期优化（PDF 第 15 章） | [13_模板与编译期优化.md](13_模板与编译期优化.md) ✅ |
| 14 性能测试与Benchmark（PDF 第 16 章） | [14_性能测试与Benchmark.md](14_性能测试与Benchmark.md) ✅ |
| 15 嵌入式系统优化（PDF 第 17 章） | [15_嵌入式系统优化.md](15_嵌入式系统优化.md) ✅ |
| 16 编译参数速查表（PDF 第 18 章） | [16_编译参数速查表.md](16_编译参数速查表.md) ✅ |
| 17 性能优化检查清单（全书综合） | [17_性能优化检查清单.md](17_性能优化检查清单.md) ✅ |
| 18 学习路线与实践项目（全书综合） | [18_学习路线与实践项目.md](18_学习路线与实践项目.md) ✅ |
| 19 项目完成报告（阶段四生成） | [19_项目完成报告.md](19_项目完成报告.md) ✅ |
| 20 工程实践与常见工作中的坑（工程深度补充） | [20_工程实践与常见坑.md](20_工程实践与常见坑.md) ✅ |

## 5. 每个实验的链接

所有实验位于 `src/`，统一由 `src/CMakeLists.txt` 管理。`src/common/` 提供共用的计时与 CPU 检测模块。

| 实验目录 | 实验内容 | 对应笔记 |
|----------|----------|----------|
| [01_profiling](../src/01_profiling) | 手动计时 / perf stat / perf record / perf report | 03、14 |
| [02_integer_float](../src/02_integer_float) | 整数/浮点类型、除法、类型转换 | 04 |
| [03_branch](../src/03_branch) | 分支预测、branchless、branch-misses | 04、08 |
| [04_loop](../src/04_loop) | 循环展开、不变量外提、多累加器、循环融合/拆分 | 04、05、08 |
| [05_function](../src/05_function) | 函数调用方式对比 | 04 |
| [06_class_virtual](../src/06_class_virtual) | 类、虚函数、继承、RTTI、构造析构 | 04、05 |
| [07_container](../src/07_container) | 标准容器性能、reserve、内存池 | 06 |
| [08_memory_cache](../src/08_memory_cache) | 访问模式、Cache 规模扫描、AoS/SoA、Cache 竞争 | 06、08 |
| [09_alignment](../src/09_alignment) | 内存对齐与 SIMD 对齐 | 06、10 |
| [10_multithreading](../src/10_multithreading) | 多线程、atomic、mutex、reduction、扩展性 | 07 |
| [11_false_sharing](../src/11_false_sharing) | False Sharing 与 alignas(64) | 07 |
| [12_auto_vectorization](../src/12_auto_vectorization) | 自动向量化、restrict、向量化报告 | 05、09 |
| [13_intrinsics](../src/13_intrinsics) | SSE/AVX2/AVX-512 intrinsics | 09、10 |
| [14_cpu_dispatch](../src/14_cpu_dispatch) | CPUID、运行时选择最佳实现 | 11 |
| [15_lookup_table](../src/15_lookup_table) | 直接计算 vs 查表 | 12 |
| [16_division_optimization](../src/16_division_optimization) | 整数/浮点除法优化 | 04、12 |
| [17_template_metaprogramming](../src/17_template_metaprogramming) | 编译期计算、if constexpr、constexpr | 13 |
| [18_benchmark](../src/18_benchmark) | 性能测试陷阱 | 14 |

## 6. 编译方法

```bash
# 一键构建（创建 build、配置、编译全部示例，出错即停）
./scripts/build.sh

# 或手动执行
cmake -S src -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

支持的构建类型：

- `Debug`：`-O0 -g`，用于调试，**不要用它测性能**；
- `Release`：`-O3 -g -fno-omit-frame-pointer`，默认推荐；
- `RelWithDebInfo`：`-O2 -g`。

相关 CMake 开关（详见 `src/CMakeLists.txt`）：

- `-DUSE_MARCH_NATIVE=ON`：启用 `-march=native`（默认 OFF，保证可移植）；
- AVX2 / AVX-512 示例在各自 `CMakeLists.txt` 中单独设置编译参数，不会全局开启。

## 7. 运行方法

```bash
# 顺序运行所有示例（打印实验名，失败记录错误）
./scripts/run_all.sh

# 运行单个实验（以分支预测为例）
./build/03_branch/branch_prediction

# 运行全部 Benchmark 并保存带时间戳的结果
./scripts/benchmark_all.sh

# A/B 性能回归对比（CI 门禁；第二个可执行比第一个慢超过阈值则退出码 1）
./scripts/compare_versions.sh ./build/13_intrinsics/13_benchmark \
    --other /tmp/13_benchmark_old --threshold 0.05
```

结果保存于 `benchmark_results/`，每次运行生成带时间戳的目录。

## 8. perf 使用方法

```bash
# 一键 profile（执行 perf stat → perf record → perf report）
./scripts/perf_profile.sh ./build/08_memory_cache/cache_stride

# 手动使用
perf stat ./build/03_branch/branch_prediction          # 汇总硬件计数器
perf record -g ./build/03_branch/branch_prediction      # 采样记录
perf report                                             # 查看热点
```

注意：本机 `perf_event_paranoid=4`，普通用户无法读取硬件计数器。需要：

```bash
# 方案一：临时放开
sudo sysctl kernel.perf_event_paranoid=1
# 方案二：用 sudo 运行 perf
sudo perf stat ./build/03_branch/branch_prediction
```

常用事件示例：

```bash
perf stat -e cycles,instructions,cache-misses,branch-misses,context-switches <程序>
perf stat -e cycles ./build/03_branch/branch_prediction
```

## 9. 查看汇编的方法

```bash
# 单文件查看汇编（保留符号）
objdump -d -S --no-show-raw-insn build/03_branch/branch_prediction | less

# 直接让编译器输出汇编（不生成目标文件）
g++ -O3 -std=c++17 -S -masm=intel src/03_branch/baseline.cpp -o /tmp/base.s
g++ -O3 -std=c++17 -S -masm=intel src/03_branch/optimized.cpp -o /tmp/opt.s

# 对比两份汇编
diff /tmp/base.s /tmp/opt.s

# 查看自动向量化报告（GCC）
g++ -O3 -std=c++17 -mavx2 -fopt-info-vec -fopt-info-vec-missed -c src/12_auto_vectorization/vectorizable.cpp
```

## 10. 推荐学习路线

### 路线一：C++ 性能优化基础

适合只掌握基础 C++ 的读者，重点是"怎么写 C++ 更快"。

```
01 性能优化基本原则 → 02 平台与编译器选择 → 04 C++语言结构性能分析
→ 05 编译器优化原理 → 12 专项优化技巧 → 14 性能测试与Benchmark → 17 检查清单
配套实验：02_integer_float、03_branch、04_loop、05_function、06_class_virtual
```

### 路线二：Linux 性能分析

重点学习 perf、Benchmark 和热点分析，目标是"会测量、会定位瓶颈"。

```
03 性能热点分析 → 14 性能测试与Benchmark → 01/08 相关实验
配套实验：01_profiling、18_benchmark、08_memory_cache（配合 perf）
脚本：scripts/perf_profile.sh
```

### 路线三：CPU 与 SIMD 优化

重点学习 Cache、流水线、乱序执行、自动向量化和 Intrinsics，目标"吃透 CPU"。

```
06 内存与缓存优化 → 08 乱序执行与指令级并行 → 09 SIMD与自动向量化
→ 10 Intrinsics编程 → 11 CPU指令集分发
配套实验：08_memory_cache、09_alignment、11_false_sharing、12_auto_vectorization、
         13_intrinsics、14_cpu_dispatch
```

### 收尾：工程落地（所有路线）

三条路线完成后，用 `20_工程实践与常见坑.md` 收尾，把技术转成工程能力：

```
20 工程实践与常见坑 → 建立基准仓库 + CI 性能回归门禁
工具：scripts/compare_versions.sh（A/B 回归对比）、scripts/benchmark_all.sh（结果落盘）
```

三条路线最终都会汇聚到同一个原则：**先测量，再优化**。

---

> 版权说明：笔记与实验代码均为本人（整理者）依据 PDF 编写。PDF 版权归 Agner Fog 所有（见 PDF 第 179 页），本教程仅供非公开教育用途。
