# Chapter 08: Parallel Algorithm Design (并行算法设计)

## 文件说明

| 文件 | 内容 |
|------|------|
| `01_parallel_for_each.cpp` | 并行 for_each: 分块策略 + jthread 并行处理, 索引版/迭代器版 |
| `02_parallel_find.cpp` | 并行 find: 原子标志提前退出 + 递归分治/分块两种实现 |
| `03_parallel_partial_sum.cpp` | 并行前缀和: 分块局部和 -> 跨块偏移 -> 并行加偏移 三阶段 |
| `04_false_sharing.cpp` | 伪共享演示: alignas(64) 对齐, padding 填充, 线程局部归约 |
| `05_exception_safe_parallel.cpp` | 异常安全: future 传播异常, jthread RAII, packaged_task |

## 关键技术点

- **分块策略**: 均匀分块, 每线程连续处理, 缓存友好
- **提前退出**: atomic<bool> 通知其他线程停止搜索
- **并行前缀和**: 两阶段法 (局部scan + 跨块传播)
- **伪共享**: 缓存行 64 字节对齐, 线程局部累加归约
- **异常传播**: future.get() 重新抛出, exception_ptr 跨线程
- **RAII**: jthread 自动 join, lock_guard 自动释放
