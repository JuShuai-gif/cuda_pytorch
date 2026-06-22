# Chapter 03: Sharing Data Between Threads

本章涵盖线程间共享数据与同步的核心技术：互斥量、死锁避免、线程安全数据结构等。

## 示例文件

| 文件 | 知识点 | 说明 |
|------|--------|------|
| `01_race_condition.cpp` | 数据竞争 | 演示无同步递增导致的错误 |
| `02_mutex_basic.cpp` | mutex + lock_guard | 互斥量保护临界区，RAII 锁 |
| `03_scoped_lock.cpp` | scoped_lock (C++17) | 同时锁多个互斥量，避免死锁 |
| `04_unique_lock.cpp` | unique_lock 灵活性 | defer/try/adopt/提前解锁/移动 |
| `05_hierarchical_mutex.cpp` | 分层互斥量 | 运行时死锁检测实现 |
| `06_call_once.cpp` | call_once + once_flag | 线程安全单次初始化 |
| `07_shared_mutex.cpp` | shared_mutex (C++17) | 读写锁，读多写少场景 |
| `08_threadsafe_stack.cpp` | 线程安全栈 | 接口设计，消除竞争条件 |
| `09_recursive_mutex.cpp` ★ | 递归锁 | recursive_mutex 原理/场景/坑/替代方案 |
| `10_timed_mutex.cpp` ★ | 超时锁 | timed_mutex/try_lock_for/until, 优雅降级 |

## 核心概念

- **互斥量 (Mutex)**: 保护临界区，保证互斥访问
- **RAII 锁管理**: lock_guard / scoped_lock / unique_lock / shared_lock
- **死锁避免**: scoped_lock 同时锁定，hierarchical_mutex 强制顺序
- **接口设计**: 不返回受保护数据的指针/引用
- **读写锁**: shared_mutex 适用于读多写少

## 构建与运行

```bash
cd build
cmake .. && cmake --build . --target ch03_01_race_condition
./src/chapter-03/ch03_01_race_condition
```
