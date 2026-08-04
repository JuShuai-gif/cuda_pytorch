# 工业级 C++ 并发项目

## AI/ML 算子推理任务调度系统

一个面向 AI/ML 推理工作负载的综合 C++20 并发任务调度系统。
本项目将 Anthony Williams 所著 **《C++ Concurrency in Action（第二版）》** 中的每个核心并发概念
以实用、工业级代码库的形式实现。

### 书籍章节覆盖

| 章节 | 主题 | 模块 |
|---------|-------|--------|
| Ch2 | 线程管理 | `main.cpp`、`thread_pool.cpp` |
| Ch3 | 线程间数据共享 | `task_queue.hpp`、`concurrent_cache.hpp`、`logger.hpp` |
| Ch4 | 同步并发操作 | `thread_pool.hpp`、`task_scheduler.hpp` |
| Ch5 | C++ 内存模型与原子操作 | `spinlock.hpp`、`stop_token.hpp` |
| Ch6 | 基于锁的并发数据结构 | `task_queue.hpp`、`priority_task_queue.hpp` |
| Ch7 | 无锁数据结构 | （仅设计说明） |
| Ch8 | 设计并发代码 | `task_scheduler.hpp`、示例代码 |
| Ch9 | 高级线程管理 | `thread_pool.hpp`、`stop_token.hpp` |
| Ch10 | 并发代码的测试与调试 | `tests/` 目录 |
| Ch11 | 多线程最佳实践 | `logger.hpp`、项目模式 |

### 项目结构

```
industrial-concurrency-project/
├── include/task_scheduler/    # 仅头文件及模板库
│   ├── task_scheduler.hpp      # 核心调度器（Ch8.5）
│   ├── thread_pool.hpp         # 带工作窃取的固定大小线程池（Ch9.1）
│   ├── task_queue.hpp          # 基于锁的多生产者多消费者队列（Ch6.2）
│   ├── priority_task_queue.hpp # 基于优先级的多生产者多消费者队列（Ch6.3）
│   ├── spinlock.hpp            # TTAS 自旋锁（Ch5.3）
│   ├── concurrent_cache.hpp    # 带 shared_mutex 的 LRU 缓存（Ch3.3）
│   ├── stop_token.hpp          # 简化的停止机制（Ch9.2）
│   └── logger.hpp              # 线程安全日志器（Ch11）
├── src/                        # 非模板实现
│   ├── main.cpp
│   ├── thread_pool.cpp
│   ├── task_scheduler.cpp
│   └── logger.cpp
├── tests/                      # 单元测试和压力测试
│   ├── test_thread_pool.cpp
│   ├── test_task_queue.cpp
│   ├── test_task_scheduler.cpp
│   └── test_stress.cpp
├── examples/                   # 使用示例
│   ├── example_basic.cpp
│   ├── example_pipeline.cpp
│   ├── example_inference.cpp
│   └── example_producer_consumer.cpp
└── docs/                       # 架构和设计文档
    ├── architecture.md
    └── design_notes.md
```

### 快速开始

```bash
# 构建
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 运行测试
ctest --output-on-failure

# 运行示例
./example_basic
./example_pipeline
./example_inference
./example_producer_consumer

# 运行主程序演示
./main
```

### ThreadSanitizer 构建

```bash
mkdir build-tsan && cd build-tsan
cmake .. -DCMAKE_BUILD_TYPE=Tsan
make -j$(nproc)
./test_stress
```

### 核心特性

- **工作窃取线程池**：带每线程本地队列的固定大小线程池（Ch9.1）
- **优先级调度**：面向延迟敏感任务的多级优先级队列（Ch6.3）
- **流水线执行**：带 future 链式调用的多阶段任务流水线（Ch8.3）
- **并发 LRU 缓存**：利用 `std::shared_mutex` 实现读优化的缓存（Ch3.3）
- **TTAS 自旋锁**：带指数退避的"测试-测试并设置"自旋锁（Ch5.3）
- **优雅关闭**：用于协作式中断的停止令牌机制（Ch9.2）
- **线程安全日志**：带原子快速路径的时间戳分级日志（Ch11）
- **全程 RAII**：无裸 `new`/`delete`，异常安全的资源管理
- **TSan 就绪**：为 ThreadSanitizer 验证而设计（Ch10）

### 环境要求

- C++20 编译器（GCC 12+、Clang 16+）
- CMake 3.14+
- pthread（Linux/macOS）

### 许可证

MIT - 详见 LICENSE 文件。
