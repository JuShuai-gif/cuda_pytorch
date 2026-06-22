# Chapter 01: Hello, C++ Concurrency World

本章介绍 C++ 并发编程的基础概念，包括线程创建、并发与并行的区别、硬件线程数检测、以及工业级计时器的实现。

## 示例文件

| 文件 | 知识点 | 说明 |
|------|--------|------|
| `01_hello_concurrent.cpp` | std::thread 基础 | 创建2个线程，join 等待完成 |
| `02_concurrent_vs_parallel.cpp` | 并发 vs 并行 | CPU密集型与IO密集型任务的调度差异 |
| `03_num_threads.cpp` | hardware_concurrency() | 检测硬件线程数，分配并行任务 |
| `04_industrial_timer.cpp` | RAII 计时器 | ScopedTimer 封装，性能基准测量 |

## 构建与运行

```bash
cd c++/templates/build
cmake ..
cmake --build . --target ch01_01_hello_concurrent
./src/chapter-01/ch01_01_hello_concurrent
```

## 核心概念

- **并发 (Concurrency)**: 多个任务可以独立推进，可能在单核上通过时间片轮转实现
- **并行 (Parallelism)**: 多个任务在不同 CPU 核心上真正同时执行
- **std::thread::hardware_concurrency()**: 获取硬件支持的并发线程数
- **RAII 管理资源**: 利用 C++ 对象生命周期自动管理线程、计时器等资源
