# Chapter 02: Thread Management

本章涵盖 std::thread 的生命周期管理、参数传递、所有权转移等核心话题。

## 示例文件

| 文件 | 知识点 | 说明 |
|------|--------|------|
| `01_basic_thread.cpp` | std::thread 基础 | join/detach 的区别和 joinable 状态 |
| `02_thread_guard.cpp` | RAII 线程保护 | thread_guard 类确保异常安全 |
| `03_joining_thread.cpp` | 自动 join 的线程 | 手动实现 std::jthread 的 RAII 版本 |
| `04_param_pass.cpp` | 参数传递 | 传值、std::ref、成员函数、lambda |
| `05_thread_ownership.cpp` | 所有权转移 | 移动语义、线程容器 |
| `06_parallel_accumulate.cpp` | 分治并行累加 | 工业级 parallel_accumulate 实现 |

## 核心概念

- **thread_guard**: RAII 包装，确保线程在异常时也能被 join
- **JoiningThread**: 拥有线程所有权，析构时自动 join (类似 C++20 std::jthread)
- **参数传递**: 默认传值（拷贝），使用 std::ref 传引用，注意生命周期
- **移动语义**: std::thread 不可拷贝，只能移动
- **分治并行**: 将数据分块，每个线程处理一块，最后汇总

## 构建与运行

```bash
cd build
cmake .. && cmake --build . --target ch02_01_basic_thread
./src/chapter-02/ch02_01_basic_thread
```
