# C++ 并发编程实战 - 学习知识库

基于 **《C++ Concurrency in Action 第二版》**（Anthony Williams 著，陈晓伟 译）构建的系统化 C++ 并发编程学习项目，集**学习笔记 + 工业级代码 + 实战项目**三位一体。

---

## 目录结构

```text
c++/templates/
├── note/                          # 学习笔记（对应全书 11 章）
├── src/                           # 示例代码（每章独立可编译运行）
│   ├── chapter-01/                # 你好，C++ 并发世界
│   ├── chapter-02/                # 线程管理
│   ├── chapter-03/                # 线程间共享数据
│   ├── chapter-04/                # 同步并发操作
│   ├── chapter-05/                # C++ 内存模型和原子操作
│   ├── chapter-06/                # 基于锁的并发数据结构
│   ├── chapter-07/                # 无锁并发数据结构
│   ├── chapter-08/                # 并发代码设计
│   ├── chapter-09/                # 高级线程管理
│   ├── chapter-10/                # 并行算法（C++17）
│   ├── chapter-11/                # 测试和调试多线程应用
│   ├── chapter-12/                # 现代 C++20 并发新特性 ★
│   ├── chapter-13/                # 内存模型与缓存优化进阶 ★
│   ├── chapter-14/                # 并行算法进阶 ★
│   ├── chapter-15/                # 性能分析与优化 ★
│   ├── chapter-16/                # 高性能计算 HPC ★
│   └── chapter-17/                # 工程化实践 ★
├── project/                       # 工业级并发项目
│   └── industrial-concurrency-project/
├── README.md                      # 本文档
└── CMakeLists.txt                 # 顶层构建文件
```

---

## 环境要求

| 组件 | 最低版本 | 推荐版本 |
|------|----------|----------|
| C++ 标准 | C++17 | C++20 |
| 编译器 | GCC 8+ / Clang 7+ / MSVC 2019+ | GCC 11+ / Clang 14+ / MSVC 2022+ |
| CMake | 3.14+ | 3.20+ |
| OS | Linux / macOS / Windows | — |
| TBB（可选）| 2019+ | oneTBB 2021+（第10章并行算法需要）|

## 编译方法

```bash
# 进入项目目录
cd c++/templates

# 创建构建目录
mkdir build && cd build

# 配置（启用所有章节）
cmake -DBUILD_ALL=ON ..

# 编译（使用所有可用核心）
cmake --build . -j$(nproc)
```

> **注意**：顶层 `CMakeLists.txt` 中部分章节默认为注释状态。如需编译特定章节，可在 CMake 配置时通过 `-DBUILD_ALL=ON` 启用，或手动取消对应 `add_subdirectory` 的注释。

---

## 运行示例

```bash
# === 章节示例 ===

# 第1章：初识并发
./build/src/chapter-01/ch01_01_hello_concurrent

# 第3章：层级互斥锁
./build/src/chapter-03/ch03_05_hierarchical_mutex

# 第9章：线程池（带 future 支持）
./build/src/chapter-09/ch09_02_thread_pool_with_future

# === 工业项目 ===

./build/project/industrial-concurrency-project/example_basic
```

---

## 学习路线

| 阶段 | 内容 | 建议时间 |
|------|------|----------|
| **入门** | 第1~2章：并发概念 + 线程管理 | 1 周 |
| **基础** | 第3~4章：共享数据 + 同步操作 | 2 周 |
| **进阶** | 第5~7章：内存模型 + 锁/无锁数据结构 | 2~3 周 |
| **高级** | 第8~9章：代码设计 + 线程池 | 2 周 |
| **实战** | 第10~11章：并行算法 + 测试调试 | 1~2 周 |
| **拓展** | 第12章：现代 C++20 并发新特性 | 1 周 ★ |
| **进阶** | 第13章：内存与缓存优化、第14章：并行算法进阶 | 2 周 ★ |
| **优化** | 第15章：性能分析与优化 | 1 周 ★ |
| **HPC** | 第16章：高性能计算 (OpenMP / GPU) | 1 周 ★ |
| **工程** | 第17章：工程化实践 (Sanitizer / CI) | 1 周 ★ |
| **项目** | 工业项目：AI 推理任务调度 | 1~2 周 |

---

## 章节目录索引

| 章节 | 章节名称 | 核心知识点 | 源码目录 | 笔记文件 |
|------|----------|------------|----------|----------|
| 第1章 | 你好，C++ 并发世界 | `std::thread` 基础、并发 vs 并行、`hardware_concurrency()`、RAII 计时器 | `src/chapter-01/` | `note/chapter-01.md` |
| 第2章 | 线程管理 | 线程创建/传参/join/detach、RAII 线程守卫、线程所有权转移、并行累加 | `src/chapter-02/` | `note/chapter-02.md` |
| 第3章 | 线程间共享数据 | `std::mutex`、`std::lock_guard`、`std::unique_lock`、`std::scoped_lock`、层级锁、`call_once`、`shared_mutex`、线程安全栈 | `src/chapter-03/` | `note/chapter-03.md` |
| 第4章 | 同步并发操作 | `condition_variable`、生产者-消费者、`future/promise`、`async`、`packaged_task`、`shared_future`、超时等待 | `src/chapter-04/` | `note/chapter-04.md` |
| 第5章 | C++ 内存模型和原子操作 | `std::atomic`、内存序（relaxed / acquire-release / seq_cst）、CAS、自旋锁、无锁计数器 | `src/chapter-05/` | `note/chapter-05.md` |
| 第6章 | 基于锁的并发数据结构 | 线程安全栈/队列、细粒度锁队列、线程安全查找表、线程安全链表 | `src/chapter-06/` | `note/chapter-06.md` |
| 第7章 | 无锁并发数据结构 | 无锁栈、SPMC/MPMC 无锁队列、ABA 问题、Hazard Pointer | `src/chapter-07/` | `note/chapter-07.md` |
| 第8章 | 并发代码设计 | 并行 `for_each`/`find`、并行部分和、伪共享、异常安全的并行代码 | `src/chapter-08/` | `note/chapter-08.md` |
| 第9章 | 高级线程管理 | 线程池、带 future 的线程池、工作窃取、可中断线程 | `src/chapter-09/` | `note/chapter-09.md` |
| 第10章 | 并行算法（C++17） | `std::execution::par`、并行排序/for_each/transform_reduce、执行策略 | `src/chapter-10/` | `note/chapter-10.md` |
| 第11章 | 测试和调试多线程应用 | 死锁复现与检测、竞态检测器、压力测试、并发日志 | `src/chapter-11/` | `note/chapter-11.md` |
| 第12章 ★ | 现代 C++20 并发特性 | jthread、stop_token、semaphore、latch、barrier、协程 | `src/chapter-12/` | `note/chapter-12.md` |
| 第13章 ★ | 内存模型与缓存优化 | Cache Line、伪共享基准、Memory Fence、NUMA、CPU 绑核 | `src/chapter-13/` | `note/chapter-13.md` |
| 第14章 ★ | 并行算法进阶 | Pipeline、Parallel Reduce、Prefix Sum、Batch Processing | `src/chapter-14/` | `note/chapter-14.md` |
| 第15章 ★ | 性能分析与优化 | Benchmark、Latency/Throughput、Contention、Lock vs Lock-free | `src/chapter-15/` | `note/chapter-15.md` |
| 第16章 ★ | 高性能计算 HPC | OpenMP 基础、Schedule、Reduction、CUDA 概念、异构计算 | `src/chapter-16/` | `note/chapter-16.md` |
| 第17章 ★ | 工程化实践 | Sanitizer、Stress Test、Unit Test、CMake 最佳实践 | `src/chapter-17/` | `note/chapter-17.md` |

---

## 推荐学习顺序

| 目标 | 推荐路径 | 说明 |
|------|----------|------|
| **初学者** | 第1章 → 第11章（顺序阅读） | 从零开始，建立完整体系 |
| **有经验者** | 跳过第1章，从第2章开始 | 已掌握并发基础概念 |
| **面试准备** | 重点第3章（锁）、第5章（原子）、第9章（线程池）、第12章（C++20特性） | 面试高频考点 |
| **HPC方向** | 第14章（并行算法）+ 第16章（OpenMP/CUDA）+ 第13章（NUMA） | 高性能计算方向 ★ |
| **性能优化** | 第13章（Cache）+ 第15章（Benchmark）+ 第5章（Atomics） | 性能工程师方向 ★ |
| **项目实战** | 第9章 + 工业项目 + 第17章（工程化） | 直接上手线程池与任务调度 |

---

## 工业项目架构

### 项目名称

**AI 算子推理任务调度系统**（Industrial Concurrency Project）

### 核心模块

| 模块 | 功能 | 对应章节知识 |
|------|------|-------------|
| **线程池** | 工作窃取线程池，动态负载均衡 | 第9章：高级线程管理 |
| **任务队列** | 多优先级无锁任务队列（MPMC） | 第7章：无锁数据结构 |
| **优先级调度** | 基于优先级的抢占式任务调度器 | 第6章：锁数据结构设计 |
| **并发缓存** | 线程安全的 LRU 缓存，支持高并发读写 | 第3章：`shared_mutex` |
| **日志系统** | 异步无阻塞日志记录器 | 第11章：并发日志调试 |
| **性能监控** | RAII 计时器 + 吞吐量统计 | 第1章：RAII 模式 |

### 架构概览

```text
┌─────────────────────────────────────────────────────┐
│                    任务提交层                         │
│   用户提交推理任务（优先级、超时、回调）               │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│                  优先级调度器                         │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│   │ 高优先级  │  │ 中优先级  │  │ 低优先级  │            │
│   │ 队列     │  │ 队列     │  │ 队列     │            │
│   └────┬────┘  └────┬────┘  └────┬────┘            │
│        │            │            │                  │
│        └────────────┼────────────┘                  │
│                     │                               │
│              优先级合并队列                          │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│               工作窃取线程池                          │
│   ┌──────┐ ┌──────┐ ┌──────┐     ┌──────┐          │
│   │Woker0│ │Woker1│ │Woker2│ ... │WokerN│          │
│   │local │ │local │ │local │     │local │          │
│   │queue │◄┼──────┼─┼──────┼─────┤queue │          │
│   └──┬───┘ └──┬───┘ └──┬───┘     └──┬───┘          │
│      │        │        │            │               │
│      ▼        ▼        ▼            ▼               │
│   ┌──────────────────────────────────┐              │
│   │       并发 LRU 缓存              │              │
│   └──────────────────────────────────┘              │
│      │        │        │            │               │
│      ▼        ▼        ▼            ▼               │
│   ┌──────────────────────────────────┐              │
│   │       AI 推理引擎                │              │
│   └──────────────────────────────────┘              │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│                  异步日志系统                         │
│   无锁环形缓冲区 → 后台落盘线程                       │
└─────────────────────────────────────────────────────┘
```

### 知识点覆盖矩阵

| 章节 | 调度器 | 任务队列 | 缓存 | 日志 | 监控 |
|------|:------:|:--------:|:----:|:----:|:----:|
| Ch1: 并发基础 | | | | | ✓ |
| Ch2: 线程管理 | ✓ | | | ✓ | |
| Ch3: 共享数据 | ✓ | | ✓ | | |
| Ch4: 同步操作 | ✓ | | | ✓ | |
| Ch5: 原子操作 | | ✓ | | ✓ | |
| Ch6: 锁数据结构 | ✓ | | ✓ | | |
| Ch7: 无锁结构 | | ✓ | | | |
| Ch8: 代码设计 | | | | | |
| Ch9: 线程池 | ✓ | | | | |
| Ch10: 并行算法 | ✓ | | | | |
| Ch11: 测试调试 | | | ✓ | ✓ | ✓ |

---

## 解析说明

- **原书来源**：《C++ Concurrency in Action, 2nd Edition》（Anthony Williams 著，Manning Publications 出版）
- **中文版本**：基于陈晓伟非官方翻译版本整理
- **PDF 页数**：668 页
- **解析方式**：通过 `pdftotext` 提取文本后人工整理归纳
- **笔记性质**：笔记内容为学习理解后重新组织编写，非原文逐字照搬，包含大量原书外补充知识
- **代码声明**：所有示例代码为**原创工业级实现**，参考书中概念但未照抄书中代码片段

---

## 项目统计

| 类别 | 数量 | 说明 |
|------|------|------|
| 笔记文件 | 17 个 `.md` | 每章一篇，12-17章为新增拓展 |
| 代码文件 | 89 个 `.cpp` | 每章 3~8 个独立示例 |
| 代码行数 | ~17,000 行 | 含注释与空行 |
| 工业项目模块 | 8 个头文件 + 4 个测试 + 4 个示例 | planning 阶段 |
| 构建目标 | 90+ 个可执行文件 | 每示例独立 target |

---

## 参考资料

| 资源 | 链接 |
|------|------|
| **原书英文版** | C++ Concurrency in Action, 2nd Edition, Manning Publications |
| **中文翻译 GitHub** | [xiaoweiChen/CPP-Concurrency-In-Action-2ed-2019](https://github.com/xiaoweiChen/CPP-Concurrency-In-Action-2ed-2019) |
| **原书官方源码** | [anthonywilliams/ccia_code_samples](https://github.com/anthonywilliams/ccia_code_samples) |
| **C++ 并发参考** | [cppreference.com - Concurrency](https://en.cppreference.com/w/cpp/thread) |
| **C++17 并行算法** | [cppreference.com - Execution Policies](https://en.cppreference.com/w/cpp/algorithm/execution_policy_tag) |

---

## 贡献与许可

本项目为个人学习整理性质的知识库，仅供学习参考。代码部分遵循 MIT License，笔记内容转载请注明出处。
