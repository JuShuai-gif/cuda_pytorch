# Chapter 12 — 现代 C++20 并发新特性

C++20 引入的并发特性实战代码。

## 内容概览

| 文件 | 主题 | 关键知识点 |
|------|------|-----------|
| `01_jthread_basic.cpp` | jthread 基础 | 自动 join、内置 stop_token、与 thread 对比 |
| `02_stop_token.cpp` | 停止机制 | stop_source/stop_token/stop_callback、cv 可中断等待 |
| `03_semaphore.cpp` | 信号量 | counting_semaphore、binary_semaphore、连接池、生产者-消费者 |
| `04_latch.cpp` | 闩锁 | 一次性门控、初始化同步、try_wait |
| `05_barrier.cpp` | 屏障 | 多阶段同步、完成回调、arrive_and_drop |
| `06_coroutine_demo.cpp` | 协程基础 | Generator、Task、co_await/co_yield/co_return |

## 编译运行

```bash
mkdir build && cd build
cmake ..
cmake --build . -j$(nproc)
./ch12_01_jthread_basic
./ch12_02_stop_token
./ch12_03_semaphore
./ch12_04_latch
./ch12_05_barrier
./ch12_06_coroutine_demo
```

## 学习建议

1. `jthread` 应在所有新项目中替代 `std::thread`
2. `stop_token` 是协作式取消的标准方案
3. `semaphore/latch/barrier` 填补了 C++ 同步原语的长期空白
4. 协程的学习曲线较陡，建议先掌握 Generator 模式，再深入 async Task
