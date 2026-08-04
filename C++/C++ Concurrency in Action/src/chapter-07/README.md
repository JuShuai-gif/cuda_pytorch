# Chapter 07: Lock-Free Data Structures (无锁数据结构)

## 文件说明

| 文件 | 内容 |
|------|------|
| `01_lockfree_stack.cpp` | Treiber Stack 无锁栈, 基于 CAS + shared_ptr 内存管理 |
| `02_lockfree_queue_spmc.cpp` | SPSC/SPMC 环形缓冲区无锁队列, 缓存行对齐消除伪共享 |
| `03_lockfree_queue_mpmc.cpp` | Michael-Scott MPMC 无锁队列, shared_ptr 自动节点管理 |
| `04_aba_demo.cpp` | ABA 问题演示, Tagged Pointer (版本号+指针) 解决方案 |
| `05_hazard_pointer.cpp` | Hazard Pointer 安全内存回收机制的简化实现 |
| `06_backoff.cpp` ★ | 5种退避策略对比 (No/Pause/Yield/Exponential/Randomized) |
| `07_lockfree_ringbuffer.cpp` ★ | MPMC 无锁环形缓冲区, cache line 对齐优化 |
| `08_epoch_reclamation.cpp` ★ | Epoch-Based 内存回收 (RCU简化), 与 Hazard Pointer 对比 |

## 编译

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

单个文件编译:

```bash
g++ -std=c++20 -O2 -pthread 01_lockfree_stack.cpp -o lockfree_stack
```

## 关键技术点

- **CAS (compare_exchange)**: 无锁编程的核心原语
- **内存序**: acquire/release/relaxed 的选择影响正确性和性能
- **ABA 问题**: 使用 Tagged Pointer 或 Double-width CAS 解决
- **内存回收**: shared_ptr 或 Hazard Pointer 延迟回收机制
- **伪共享**: 缓存行对齐 (`alignas(64)`)
