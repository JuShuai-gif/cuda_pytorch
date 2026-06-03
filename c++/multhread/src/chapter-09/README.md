# Chapter 09: Advanced Thread Management (高级线程管理)

## 文件说明

| 文件 | 内容 |
|------|------|
| `01_simple_thread_pool.cpp` | 简单固定线程池: mutex + condition_variable 任务队列 |
| `02_thread_pool_with_future.cpp` | 工业级线程池: submit 返回 future, packaged_task 异常传播 |
| `03_work_stealing_pool.cpp` | Work Stealing 线程池: 每线程本地 deque + 全局队列 + 任务窃取 |
| `04_interruptible_thread.cpp` | 可中断线程: interrupt_flag, 中断点, 禁用中断 RAII |

## 关键技术点

- **线程池基础**: 生产者-消费者模式, `std::function<void()>` 类型擦除
- **future 支持**: `std::packaged_task` 封装可调用对象, future 获取结果
- **Work Stealing**: 本地队列 LIFO pop, 远程队列 FIFO steal, 减少竞争
- **可中断线程**: 原子标志 + condition_variable 唤醒等待线程
- **中断点**: `interruption_point()` 显式检查, RAII 禁用中断
