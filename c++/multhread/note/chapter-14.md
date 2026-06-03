# 第14章：并行算法进阶

> C++17 的并行 STL 是起点而非终点。本章深入探索手动实现的并行算法：Pipeline、Parallel Reduce、Parallel Scan、Batch Processing，以及 Work Stealing 的深层机制。

---

## 14.1 任务并行 vs 数据并行

### 任务并行（Task Parallelism）

将**不同任务**分配到不同线程：

```
任务A (解析) ──→ 任务B (验证) ──→ 任务C (存储)
       ↑              ↑              ↑
    线程1          线程2          线程3
```

### 数据并行（Data Parallelism）

将**数据**切分，相同操作在不同数据块上并行：

```
数据块1 ──→ 操作 ──→ 结果1    (线程1)
数据块2 ──→ 操作 ──→ 结果2    (线程2)
数据块3 ──→ 操作 ──→ 结果3    (线程3)
```

### 选择指南

| 场景 | 推荐 |
|------|------|
| 独立的异构任务 | 任务并行 |
| 大规模同构数据 | 数据并行 |
| 多阶段流水线 | 混合（Pipeline） |
| GPU 计算 | 数据并行（SIMT） |

---

## 14.2 Pipeline 模式

### 原理

Pipeline 将处理过程分解为多个**阶段**，每个阶段由独立线程执行，数据在阶段间通过**有界队列**传递：

```
[输入] → [Stage 1] → [Stage 2] → [Stage 3] → [输出]
          线程1        线程2        线程3
```

**关键**：每个阶段可以同时处理不同的数据项，实现**时间并行**。

**生活类比**：汽车装配线。工位 A 装发动机（3 分钟），工位 B 装车轮（2 分钟），工位 C 装座椅（2 分钟）。第 1 辆车在工位 B 时，第 2 辆车可以进入工位 A——虽然每辆车要 7 分钟才能下线，但流水线每 3 分钟就能产出一辆（瓶颈决定了吞吐量）。

### 性能分析

- **延迟**：所有阶段耗时之和
- **吞吐量**：由最慢的阶段决定（瓶颈）
- **加速比**：理想情况下等于阶段数（受瓶颈限制）

---

## 14.3 Parallel Reduce（并行归约）

### 原理

归约操作（sum、max、min、product）可以通过**分治**并行化：

```
数据: [a b c d e f g h]
        ↓        ↓
    局部和1   局部和2    ← 并行计算
        ↓        ↓
        └── 总和 ──┘    ← 合并
```

### 实现模式

```cpp
template <typename It, typename T, typename BinaryOp>
T parallel_reduce(It first, It last, T init, BinaryOp op) {
    const size_t n = std::distance(first, last);
    const size_t num_threads = std::thread::hardware_concurrency();
    const size_t chunk_size = n / num_threads;

    std::vector<std::jthread> threads;
    std::vector<T> partial_results(num_threads, init);

    for (size_t t = 0; t < num_threads; ++t) {
        auto chunk_begin = first + t * chunk_size;
        auto chunk_end = (t == num_threads - 1) ? last : chunk_begin + chunk_size;

        threads.emplace_back([&partial_results, t, chunk_begin, chunk_end, op]() {
            T local = partial_results[t];
            for (auto it = chunk_begin; it != chunk_end; ++it) {
                local = op(local, *it);
            }
            partial_results[t] = local;
        });
    }
    threads.clear();

    // 合并阶段结果
    T result = init;
    for (const auto& pr : partial_results) {
        result = op(result, pr);
    }
    return result;
}
```

### 关键考量

- 操作必须满足**结合律**（associative）才能正确并行化
- 浮点数加法不满足结合律（`a+b+c` 可能不等于 `a+c+b`）
- 线程数应与数据规模匹配（小数据用单线程）

---

## 14.4 Parallel Scan（并行前缀和）

### 原理

前缀和（Prefix Sum / Inclusive Scan）是许多并行算法的基础（如基数排序、稀疏矩阵）。并行化分为两步：

**Step 1: 上扫（Up-Sweep / Reduce）** — 构建部分和树
```
[a b c d e f g h]
 [a+b c+d e+f g+h]
   [a+b+c+d e+f+g+h]
     [a+b+c+d+e+f+g+h]
```

**Step 2: 下扫（Down-Sweep）** — 传播前缀和
```
[a b c d e f g h]
 [a a+b c+d ...]
   ...
```

### 复杂度

- 串行：O(n)
- 并行（work-efficient）：O(n) work, O(log n) span
- 实际应用中常使用**两阶段法**：先每线程计算局部前缀和，再合并

---

## 14.5 Batch Processing（批处理优化）

### 原理

将多个操作**批量提交**，减少同步开销（如锁竞争、系统调用、I/O）：

```cpp
// 低效: 每个操作都获取锁
for (auto& item : items) {
    std::lock_guard lock(mtx);
    shared_queue.push(item); // 锁开销占总时间 80%
}

// 高效: 批量处理
std::vector<Item> batch;
for (auto& item : items) {
    batch.push_back(item);
    if (batch.size() >= BATCH_SIZE) {
        std::lock_guard lock(mtx);
        for (auto& b : batch) shared_queue.push(b);
        batch.clear();
    }
}
```

### 适用场景

- 数据库批量插入
- 网络请求聚合
- 无锁队列的批量入队/出队
- 日志系统的缓冲写入

---

## 14.6 Work Stealing 深层机制

### 核心思想

每个线程维护自己的**双端队列（deque）**：
- 本地线程从队尾 push/pop（LIFO，利用缓存局部性）
- 空闲线程从其他队列的队头 steal（FIFO，减少争用）

### 为什么 Work Stealing 高效？

1. **局部性**：本地任务使用本地数据，缓存友好
2. **低争用**：窃取发生在单独的一端
3. **负载均衡**：空闲线程自动寻找工作
4. **无锁实现**：可用原子操作实现高效的 deque

### 关键实现细节

```
本地操作 (pop/push):    ← 队尾
  [任务N] [任务N-1] ... [任务2] [任务1]
                              → 窃取 (steal): 队头
```

---

## 14.7 知识体系交叉引用

| 本章主题 | 相关章节 |
|----------|----------|
| Pipeline | 第9章 线程池 (生产者-消费者模式) |
| Parallel Reduce | 第10章 并行算法 (transform_reduce) |
| Batch Processing | 第7章 无锁队列 (批量操作) |
| Work Stealing | 第9章 工作窃取线程池 |
| 数据并行 | 第8章 并发代码设计 (parallel_for_each) |

---

## 14.8 本章小结

并行算法设计的三条核心原则：

1. **分解** — 找到计算的独立单元（数据块或任务阶段）
2. **调度** — 将工作分配给线程（静态分块 vs 工作窃取）
3. **合并** — 安全地组合并行结果（归约、扫描、流水线）

记住：**不是所有算法都适合并行化**。小数据量时串行更快；浮点数的非结合运算需要谨慎处理；IO 密集型任务用异步而非多线程。
