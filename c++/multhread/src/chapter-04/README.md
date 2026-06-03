# Chapter 04 - 线程同步原语

C++ Concurrency in Action 第4章示例代码。

## 内容概览

| 文件 | 主题 | 关键知识点 |
|------|------|-----------|
| `01_condition_variable.cpp` | 生产者-消费者 | `condition_variable::wait` 的 while 循环预防虚假唤醒 |
| `02_producer_consumer_queue.cpp` | 有界阻塞队列 | 两把条件变量 (not_full / not_empty) 实现满/空等待 |
| `03_future_promise.cpp` | future + promise | 单次结果传递、异常传播 |
| `04_async_task.cpp` | std::async | launch::async vs launch::deferred、异常传播 |
| `05_packaged_task.cpp` | packaged_task | 可调用对象包装、与线程池配合 |
| `06_shared_future.cpp` | shared_future | 多线程等待同一结果（一次性广播） |
| `07_timed_wait.cpp` | 限时等待 | wait_for / wait_until、条件变量超时、chrono 工具 |

## 编译运行

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
./01_condition_variable
./02_producer_consumer_queue
# ...
```
