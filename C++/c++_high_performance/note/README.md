# note/README.md - 现代 C++ 高性能编程中文学习项目

## 1. 项目介绍

本项目以《C++ High Performance - Boost and optimize the performance of your C++ 17 code》(Packt, 2018) 为蓝本，
将其转化为一套系统化、可实践的中文学习项目：

- `note/`：系统化中文笔记（19 篇 + 索引），涵盖全书 11 章；
- `src/`：可独立编译运行的 C++17 教学示例，含"未优化 / 优化"对照实验；
- `scripts/`：构建、运行、Benchmark、perf、Sanitizer、汇编分析脚本；
- `benchmark_results/`：可重复执行的性能测试结果（含时间戳）。

核心原则：**不逐句翻译、不大段复制原书代码**；所有代码根据书中思想重新实现，
性能结论必须由本项目实际 Benchmark 验证，且标注依赖条件（数据规模、编译器、标准库、硬件）。

## 2. PDF 信息

| 项目 | 内容 |
|---|---|
| 书名 | C++ High Performance - Boost and optimize the performance of your C++ 17 code |
| 作者 | Viktor Sehr, Bjorn Andrist |
| 出版社 | Packt Publishing |
| 成书时间 | 2018 年 1 月 |
| PDF 页数 | 362 页，正文 11 章 |
| 文件位置 | `../C++ High Performance - Boost and optimize the performance of your C++ 17 code.pdf` |
| 页码约定 | 笔记统一引用 **PDF 页码**；印刷页码 = PDF 页码 - 17 |

## 3. 适合人群

- 已掌握 C++ 基础语法，希望理解 C++ 高性能编程的开发者；
- 需要了解 move 语义、lambda、智能指针、模板元编程等现代 C++ 特性的工程师；
- 从事性能敏感开发（游戏、图形、金融、嵌入式、后端）的工程师；
- 想建立"测量 → 定位热点 → 优化 → 再测量"工程方法论的开发者。

## 4. 前置知识

- C++ 基础语法（类、模板、指针、函数重载）；
- 基本数据结构概念（数组、链表、哈希表、二叉树）；
- 基本操作系统概念（进程、虚拟内存、线程）；
- Linux 命令行基础（`cmake`、`g++`、`perf` 的使用）。

## 5. 目录结构

```
c++_high_performance/
├── note/                        # 中文学习笔记（本目录）
│   ├── README.md                # 本文件：项目索引
│   ├── 00_全书导读与学习路线.md  # 全书分析、章节映射、学习路线
│   ├── 01 ~ 11                  # 对应原书 Chapter 1-11
│   ├── 12_Benchmark设计指南.md
│   ├── 13_编译器优化与汇编分析.md
│   ├── 14_C++17到C++20_C++23现代化补充.md
│   ├── 15_高性能C++常见误区.md
│   ├── 16_综合实践项目.md
│   ├── 17_高性能C++检查清单.md
│   ├── 18_术语表.md
│   └── 19_项目完成报告.md
├── src/                         # 可编译运行的 C++17 代码
│   ├── CMakeLists.txt
│   ├── common/                  # benchmark / statistics / barrier / test 基础设施
│   ├── chapter01_zero_cost ~ chapter11_parallel_stl
│   └── projects/                # 综合实践项目
├── scripts/                     # 构建与性能分析脚本
├── benchmark_results/           # 性能测试结果
└── progress.md                  # 分阶段进度跟踪
```

## 6. 阅读顺序

参见 [00_全书导读与学习路线.md](00_全书导读与学习路线.md) 中"四条学习路线"。

建议先通读 00，再按"路线一（基础）→ 路线二（性能工程）"推进，视兴趣进入路线三/四。

## 7. 笔记索引

| 编号 | 笔记 | 对应 PDF 章节 | 状态 |
|---|---|---|---|
| 00 | [全书导读与学习路线](00_全书导读与学习路线.md) | 全书 | ✅ |
| 01 | [C++与零成本抽象](01_C++与零成本抽象.md) | Chapter 1 | ✅ |
| 02 | [现代C++核心特性](02_现代C++核心特性.md) | Chapter 2 | ✅ |
| 03 | [性能测量与优化方法论](03_性能测量与优化方法论.md) | Chapter 3 | ✅ |
| 04 | [数据结构与内存布局](04_数据结构与内存布局.md) | Chapter 4 | ✅ |
| 05 | [迭代器原理与自定义迭代器](05_迭代器原理与自定义迭代器.md) | Chapter 5 | ✅ |
| 06 | [STL算法与Ranges](06_STL算法与Ranges.md) | Chapter 6 | ✅ |
| 07 | [内存管理与自定义分配器](07_内存管理与自定义分配器.md) | Chapter 7 | ✅ |
| 08 | [模板元编程与编译期计算](08_模板元编程与编译期计算.md) | Chapter 8 | ✅ |
| 09 | [代理对象与惰性求值](09_代理对象与惰性求值.md) | Chapter 9 | ⬜ |
| 10 | [并发与C++内存模型](10_并发与C++内存模型.md) | Chapter 10 | ⬜ |
| 11 | [Parallel_STL与GPU计算](11_Parallel_STL与GPU计算.md) | Chapter 11 | ⬜ |
| 12 | [Benchmark设计指南](12_Benchmark设计指南.md) | 方法论 | ⬜ |
| 13 | [编译器优化与汇编分析](13_编译器优化与汇编分析.md) | 方法论 | ⬜ |
| 14 | [C++17到C++20_C++23现代化补充](14_C++17到C++20_C++23现代化补充.md) | 现代化 | ⬜ |
| 15 | [高性能C++常见误区](15_高性能C++常见误区.md) | 总结 | ⬜ |
| 16 | [综合实践项目](16_综合实践项目.md) | 实践 | ⬜ |
| 17 | [高性能C++检查清单](17_高性能C++检查清单.md) | 总结 | ⬜ |
| 18 | [术语表](18_术语表.md) | 索引 | ⬜ |
| 19 | [项目完成报告](19_项目完成报告.md) | 验收 | ⬜ |

## 8. 实验索引

见 [00_全书导读与学习路线.md §4](00_全书导读与学习路线.md#4-知识点与-src-实验对应关系) 的逐章实验映射表。
每个实验目录结构：

```
experiment_name/
├── baseline.cpp      # 直观但可能低效的实现
├── optimized.cpp     # 优化实现
├── benchmark.cpp     # 性能测试
├── tests.cpp         # 正确性测试
├── README.md         # 原理、运行方法与结果解释
└── CMakeLists.txt
```

## 9. 构建方法

```bash
./scripts/build.sh                # Release 构建
./scripts/clean_build.sh          # 清理并重新构建
cmake -B build -DCMAKE_BUILD_TYPE=Debug   # Debug 构建
cmake -B build -DENABLE_TESTS=ON         # 启用正确性测试
```

可选 CMake 选项：`ENABLE_TESTS`、`ENABLE_BENCHMARKS`、`ENABLE_SANITIZERS`、
`ENABLE_THREAD_SANITIZER`、`ENABLE_CPP20_EXAMPLES`、`ENABLE_PARALLEL_STL`、
`ENABLE_BOOST`、`ENABLE_BOOST_COMPUTE`、`ENABLE_OPENCL`、`ENABLE_NATIVE_OPTIMIZATION`。

## 10. 运行方法

```bash
./scripts/run_all.sh              # 运行全部普通示例
./build/<target_name>             # 运行单个示例
```

## 11. Benchmark 方法

```bash
./scripts/benchmark_all.sh        # 运行全部 Benchmark，结果保存到 benchmark_results/
```

Benchmark 均使用 Release/RelWithDebInfo、预热、多轮，输出 mean/median/min/max/stddev/迭代次数/checksum。
禁止依据单次结果下结论，详见 [12_Benchmark设计指南.md](12_Benchmark设计指南.md)。

## 12. perf 方法

```bash
./scripts/perf_stat.sh ./build/<target_name>    # cycles/instructions/branches/cache 等
./scripts/perf_record.sh ./build/<target_name>  # 采样 + 生成报告
```

## 13. 汇编查看方法

```bash
./scripts/assembly.sh <source.cpp>              # 输出 GCC/Clang 优化与未优化汇编
```

## 14. Sanitizer 使用方法

```bash
./scripts/sanitizer_test.sh        # ASan + UBSan + LeakSanitizer
./scripts/thread_sanitizer_test.sh # ThreadSanitizer（并发示例）
```

## 15. 可选依赖

| 依赖 | 用途 | 本项目当前环境 |
|---|---|---|
| Google Test | 可选测试框架（无它也可构建核心示例） | 未安装，默认关闭 |
| Boost | Chapter 11 Boost Compute、容器测试 | 1.83 ✅ |
| OpenCL / GPU | Chapter 11 GPU 实验 | 头文件 ✅，需运行时探测 |
| Clang | 双编译器汇编对照 | 未安装，脚本自动跳过 |
| perf | 性能采样 | ✅ |

缺少依赖时：自动检测、跳过对应示例并给出原因，**不导致整个项目构建失败**。

## 16. 当前进度

见 `../progress.md`。

## 17. 已验证环境

| 项目 | 版本 |
|---|---|
| 操作系统 | Ubuntu 24.04.4 LTS，Linux 6.17.0-40-generic |
| CPU | Intel Core i9-14900HX（32 线程） |
| GPU | NVIDIA GN21-X11 |
| GCC | 13.3.0 |
| Clang | 未安装 |
| CMake | 4.1.3 |
| Boost | 1.83 |
| OpenCL | 头文件存在 |
| perf / gprof | 可用 |
| 标准库 | libstdc++（GCC 13） |

## 18. 未验证内容

- Clang 汇编对照（clang 未安装）；
- C++20 Ranges / Parallel STL 的 GCC 支持情况（待阶段实现时验证）；
- Boost Compute / OpenCL 运行时可用性（NVIDIA 驱动与 OpenCL 运行时待探测）；
- 全部性能结论（必须由本项目实际 Benchmark 验证后才写入笔记）。
