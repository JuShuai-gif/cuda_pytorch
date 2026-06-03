# 第15章：性能分析与优化

> "过早优化是万恶之源"——但不知道如何优化的程序员无法进行任何优化。本章教你系统化地测量、分析和优化并发代码性能。

---

## 15.1 性能基准测试（Benchmark）

### 方法论

好的基准测试需要：

1. **可复现性**：相同的硬件/软件条件应给出相近结果
2. **统计意义**：多次运行取平均值/中位数
3. **消除干扰**：固定 CPU 频率、关闭省电模式、隔离测试核心
4. **渐进测试**：从单线程到多线程，观察扩展性

### 关键指标

| 指标 | 定义 | 关注点 |
|------|------|--------|
| **延迟（Latency）** | 单个操作的耗时 | 实时系统、用户响应 |
| **吞吐量（Throughput）** | 单位时间完成的操作数 | 批处理、服务器 |
| **扩展性（Scalability）** | 增加线程数时的性能提升 | 并行效率 |
| **P99 延迟** | 99% 请求的响应时间 | 尾延迟、服务质量 |

### 扩展性公式

```
加速比 = T(1) / T(N)
效率   = 加速比 / N

其中:  T(1) = 单线程耗时
       T(N) = N 线程耗时
       N    = 线程数
```

**阿姆达尔定律**：
```
加速比_max = 1 / (S + (1-S)/N)

其中 S = 串行部分占比
当 N → ∞ 时，加速比 → 1/S
```

---

## 15.2 Latency vs Throughput

### 经典权衡

```
高吞吐量、高延迟:  批处理系统（如 MapReduce）
低延迟、低吞吐量:  实时控制系统
高吞吐量、低延迟:  理想目标（需要大量优化）
```

**生活类比**：快递分拣中心。每一件快递都立刻送出去 → 低延迟、低吞吐量（一辆车送一件）。等满一车再一起送 → 高延迟（第一件等很久）、高吞吐量（一辆车送很多件）。

### 如何测量

```cpp
// 延迟测量: 记录每个操作的开始-结束时间
auto start = now();
operation();
auto latency = now() - start;

// 吞吐量测量: 固定时间内完成的操作数
auto start = now();
int count = 0;
while (now() - start < 1s) {
    operation();
    ++count;
}
// throughput = count / second
```

---

## 15.3 竞争分析（Contention Analysis）

### 什么是竞争

多个线程同时试图访问**同一共享资源**（锁、原子变量、内存位置）时发生。

### 竞争等级

1. **无竞争（Uncontended）**：只有一个线程访问 — 最快
2. **低竞争（Low Contention）**：偶尔有线程重叠 — 可接受
3. **高竞争（High Contention）**：线程频繁等待 — 性能灾难

### 分析方法

- 用 `std::atomic` 的计数器记录锁等待次数
- `perf lock` 分析内核锁争用
- 观察 CPU 利用率：高竞争时 CPU 利用率反而低（都在等待）

### 减少竞争的常用手段

1. **减少临界区**：锁外预处理，锁内最小化
2. **分片锁**：将一个大锁分成多个小锁
3. **无锁数据结构**：用 CAS 替代锁
4. **读写锁**：读多写少用 `shared_mutex`
5. **RCU**：读完全无锁

---

## 15.4 性能分析工具

### perf (Linux)

```bash
# 采样 CPU 热点
perf record -g ./my_program
perf report

# 统计硬件事件
perf stat -e cycles,instructions,cache-misses,branch-misses ./my_program

# 分析锁竞争
perf lock record ./my_program
perf lock report

# 分析 cache 性能
perf stat -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses ./my_program
```

### Flamegraph

```bash
# 生成火焰图
perf record -F 99 -g ./my_program
perf script | stackcollapse-perf.pl | flamegraph.pl > flamegraph.svg
```

### Google Benchmark

```cpp
#include <benchmark/benchmark.h>

static void BM_MutexContention(benchmark::State& state) {
    static std::mutex mtx;
    static int counter = 0;
    for (auto _ : state) {
        std::lock_guard lock(mtx);
        counter++;
        benchmark::DoNotOptimize(counter);
    }
}
BENCHMARK(BM_MutexContention)->Threads(1)->Threads(2)->Threads(4);
```

### ThreadSanitizer (TSan)

```bash
# 编译时加 -fsanitize=thread
g++ -fsanitize=thread -g -O1 my_program.cpp -o my_program
./my_program  # 自动检测数据竞争
```

---

## 15.5 锁 vs 无锁性能对比指南

| 场景 | 推荐方案 | 原因 |
|------|----------|------|
| 极低竞争 | mutex | 内核 fast path 很快 |
| 低竞争、短临界区 | spinlock | 避免上下文切换 |
| 中高竞争 | lock-free | 减少等待 |
| 读多写少 | shared_mutex / RCU | 读不阻塞 |
| 极高竞争 | 重新设计（减少共享） | 架构问题非实现问题 |

**关键认知**：无锁并非总是比有锁更快。当竞争很低时，mutex 的 fast path（futex）可能比 CAS 循环更高效。

---

## 15.6 优化清单

- [ ] 确认瓶颈是 CPU 而非 IO
- [ ] 减少共享状态的范围和频率
- [ ] 用批量操作减少同步次数
- [ ] cache line 对齐避免伪共享
- [ ] 选择正确的锁粒度（not too coarse, not too fine）
- [ ] 线程数 ≤ 物理核心数（CPU 密集型）
- [ ] 用无锁替换高竞争的锁区域
- [ ] 使用 per-thread 数据减少共享
- [ ] 打开编译器优化（-O2/-O3）
- [ ] 使用 LTO（Link-Time Optimization）

---

## 15.7 知识体系交叉引用

| 本章主题 | 相关章节 |
|----------|----------|
| Benchmark 方法 | 第11章 测试调试 |
| 竞争分析 | 第5章 原子操作 |
| perf/flamegraph | 第13章 缓存优化 |
| 锁 vs 无锁 | 第6章/第7章 锁与无锁结构 |
| TSan | 第11章 数据竞争检测 |

---

## 15.8 本章小结

性能优化的黄金法则：

1. **测量，不要猜测** — 用 profiler 找到真正的瓶颈
2. **优化 20% 的代码获得 80% 的收益** — 不要到处微优化
3. **并发优化首先是架构优化** — 减少共享比优化锁更重要
4. **理解你的硬件** — 缓存行、NUMA、流水线都会影响性能
5. **测试多核扩展性** — 2 核快不代表 32 核也快
