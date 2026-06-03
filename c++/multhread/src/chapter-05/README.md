# Chapter 05 - 原子操作与 C++ 内存模型

C++ Concurrency in Action 第5章示例代码。

## 内容概览

| 文件 | 主题 | 关键知识点 |
|------|------|-----------|
| `01_atomic_basic.cpp` | 原子基础 | atomic_flag 自旋锁, atomic\<bool\>, atomic\<int\> |
| `02_compare_exchange.cpp` | CAS 操作 | compare_exchange_weak vs strong, 无锁更新模式 |
| `03_memory_order_relaxed.cpp` | 松散序 | 仅保证原子性，无顺序保证，适用场景 |
| `04_memory_order_acq_rel.cpp` | 获取-释放序 | release/acquire 配对, happens-before, 生产者-消费者 |
| `05_memory_order_seq_cst.cpp` | 顺序一致性 | 全局全序, seq_cst vs relaxed 对比 |
| `06_spinlock.cpp` | 自旋锁实现 | Plain/Yield/TTAS/ExponentialBackoff, 与 mutex 性能对比 |
| `07_lockfree_counter.cpp` | 无锁计数器 | CAS 版本、拆分计数器（减少竞争）、缓存行对齐 |

## 编译运行

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
./01_atomic_basic
./02_compare_exchange
# ...
```
