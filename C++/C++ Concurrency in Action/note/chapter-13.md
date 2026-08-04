# 第13章：内存模型与缓存优化进阶

> 写出正确的并发代码只是第一步，写出**高性能**的并发代码需要深入理解 CPU 缓存、内存对齐、NUMA 架构等底层机制。

---

## 13.1 CPU 缓存层级与 Cache Line

### 原理

现代 CPU 的存储体系是一个金字塔：

```
   越小越快
   ┌──────┐
   │ Reg  │ < 1ns, ~1KB
   ├──────┤
   │ L1   │ ~1ns, 32KB per core
   ├──────┤
   │ L2   │ ~5ns, 256KB-1MB per core
   ├──────┤
   │ L3   │ ~15ns, 8-32MB shared
   ├──────┤
   │ RAM  │ ~100ns, GBs
   └──────┘
   越大越慢
```

**Cache Line（缓存行）** 是 CPU 缓存与内存之间数据传输的最小单位，通常为 **64 字节**（x86-64）。每次从内存读取数据，CPU 实际上会加载整个 64 字节的缓存行。

**生活类比**：图书馆借书。你想借《C++ 并发编程》，图书馆员不会只递给你这一本书——而是把这一排书架（64 字节）上的书全抱过来，因为"你大概率还会看旁边的书"（空间局部性）。

### 关键性能数据

| 操作 | 延迟（约） | 相对速度 |
|------|-----------|---------|
| L1 Cache 命中 | ~1 ns | 1x |
| L2 Cache 命中 | ~5 ns | 5x |
| L3 Cache 命中 | ~15 ns | 15x |
| 主存访问 | ~100 ns | 100x |
| 跨 NUMA 节点 | ~200+ ns | 200x+ |

---

## 13.2 伪共享（False Sharing）

### 原理

当两个线程分别修改**不同但位于同一缓存行的变量**时，CPU 的**缓存一致性协议（MESI/MOESI）** 会不断在两个核心之间同步整个缓存行，导致严重性能退化。

```cpp
// 伪共享的典型场景
struct SharedData {
    std::atomic<int> counter1;  // 线程 A 频繁修改
    std::atomic<int> counter2;  // 线程 B 频繁修改
    // counter1 和 counter2 很可能在同一缓存行（64字节内）
};
```

**生活类比**：两人合住一间宿舍，分别在自己的抽屉里放东西。但"校规"规定每次整理抽屉必须把整间宿舍清空重新布置。A 放个笔、B 放本书，两人的东西被反复搬来搬去——这就是伪共享。

### 解决方案：Cache Line Padding

```cpp
struct alignas(64) PaddedCounter {
    std::atomic<int> value;
    char padding[60]; // 填充到 64 字节，确保独占缓存行
};
```

C++17 提供了 `std::hardware_destructive_interference_size` 和 `std::hardware_constructive_interference_size`，虽然两者目前在主流编译器中为建议值而非强制值。

---

## 13.3 内存对齐优化

### alignas 说明符

```cpp
// 确保对象对齐到 64 字节边界
struct alignas(64) CacheAligned {
    int data;
    // 编译器自动填充到 64 字节
};
```

### 对齐原则

1. **热点数据对齐到 cache line 边界**：避免跨行访问
2. **写频繁的数据分散到不同 cache line**：避免伪共享
3. **读频繁的数据尽量紧凑**：提高缓存局部性
4. **使用 `alignas` + padding 而非仅靠编译器**

---

## 13.4 内存屏障（Memory Fence）

### 原理

`std::atomic_thread_fence` 提供比原子变量操作更粗粒度的内存顺序控制：

- `std::atomic_thread_fence(std::memory_order_acquire)`：获取栅栏——栅栏后的操作不能被重排到栅栏前
- `std::atomic_thread_fence(std::memory_order_release)`：释放栅栏——栅栏前的操作不能被重排到栅栏后
- `std::atomic_thread_fence(std::memory_order_seq_cst)`：全序栅栏

```cpp
// 释放-获取栅栏实现生产者-消费者
std::atomic<bool> flag{false};
int data = 0;

// 生产者
void producer() {
    data = 42;
    std::atomic_thread_fence(std::memory_order_release); // 保证 data=42 在 flag=true 之前完成
    flag.store(true, std::memory_order_relaxed);
}

// 消费者
void consumer() {
    while (!flag.load(std::memory_order_relaxed));
    std::atomic_thread_fence(std::memory_order_acquire); // 保证后续读取 data 能看到 producer 的写入
    assert(data == 42); // 安全
}
```

### Fence vs 原子操作内存序

| | 原子操作内存序 | 独立 fence |
|---|---|---|
| 粒度 | 单次原子操作 | 代码区域 |
| 可读性 | 高 | 低 |
| 灵活性 | 低 | 高（可保护多个非原子访问） |
| 使用频率 | 常用 | 较少，特殊场景 |

---

## 13.5 NUMA 架构

### 原理

**NUMA（Non-Uniform Memory Access）** 是多路服务器的内存架构：

- 每颗 CPU 拥有本地内存（访问快）和远端内存（访问慢）
- 操作系统倾向于在"最近"的内存上分配页面
- 不合理的线程-内存分布会导致频繁跨节点访问

```
 ┌──────────────┐     ┌──────────────┐
 │  Node 0      │     │  Node 1      │
 │ ┌──────────┐ │     │ ┌──────────┐ │
 │ │ CPU 0-7  │ │◄───►│ │ CPU 8-15 │ │
 │ └──────────┘ │ QPI │ └──────────┘ │
 │ ┌──────────┐ │     │ ┌──────────┐ │
 │ │ RAM 32GB │ │     │ │ RAM 32GB │ │
 │ └──────────┘ │     │ └──────────┘ │
 └──────────────┘     └──────────────┘
```

**生活类比**：公司的两个办公室。员工 A 在 1 号楼，文件柜也在 1 号楼——拿文件很快。员工 B 在 2 号楼，但被迫用 1 号楼的柜子——每次要穿过天桥，多花一倍时间。这就是 NUMA 的"本地 vs 远端"访问差异。

### NUMA 优化策略

1. **CPU 绑核（Thread Affinity）**：将线程绑定到特定 CPU 核心
2. **内存绑定**：通过 `numactl` 或 `mbind()` 将内存分配在目标节点
3. **First-touch 策略**：让初始化数据的线程就是后续使用该数据的线程
4. **避免跨节点共享**：减少跨 NUMA 节点的数据共享

### 相关工具

```bash
# 查看 NUMA 拓扑
numactl --hardware
lscpu | grep NUMA

# 绑定到特定 NUMA 节点运行
numactl --cpunodebind=0 --membind=0 ./my_program

# 查看线程的 CPU 亲和性
taskset -cp <pid>
```

---

## 13.6 CPU 亲和性（Thread Affinity）

### 原理

通过 `pthread_setaffinity_np`（Linux）或 `std::thread` 的原生句柄，将线程绑定到特定 CPU 核心：

```cpp
#include <pthread.h>

void pin_to_core(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}
```

### 适用场景

- 减少线程迁移带来的缓存失效
- NUMA 感知的任务分配
- 实时系统（RTOS）的延迟保证
- Benchmark 测试的可复现性

### 最佳实践

- 将工作线程绑定到独立核心，避免与系统线程竞争
- 同一 NUMA 节点内的核心共享 L3 缓存和本地内存
- 使用 `sched_setaffinity` 系统调用或 `taskset` 命令

---

## 13.7 缓存优化的综合策略

### 读密集型场景

- 数据紧凑排列（小的结构体）
- 使用 `alignas` 对齐到 cache line
- 预取（prefetch）技术
- 利用 L1/L2/L3 局部性

### 写密集型场景

- 数据分离到不同 cache line（防止伪共享）
- 减少跨核心共享（每核心私有数据）
- 批量写入（减少缓存一致性流量）

### 混合场景

- 冷热数据分离（hot/cold splitting）
- 使用 `__builtin_prefetch` 进行软件预取
- 避免锁保护的数据结构与无锁数据结构在同一缓存行

---

## 13.8 知识体系交叉引用

| 本章主题 | 相关章节 |
|----------|----------|
| Cache Line / False Sharing | 第8章 并发代码设计 |
| Memory Fence | 第5章 原子操作与内存序 |
| NUMA / 线程绑核 | 第9章 高级线程管理 |
| 缓存对齐 | 第7章 无锁数据结构 |

---

## 13.9 本章小结

高效的并发代码需要"向下兼容"硬件的思维方式：

1. **缓存行是性能的基本单位**——关注数据布局胜过关注单个操作
2. **伪共享是隐形的性能杀手**——用 padding 和 alignas 防御
3. **NUMA 是扩展性的瓶颈**——需要显式的亲和性管理
4. **内存栅栏提供粗粒度控制**——在需要保护非原子访问时使用

记住：编写正确的并发代码靠的是锁和原子操作，编写高效的并发代码靠的是缓存和内存布局的理解。
