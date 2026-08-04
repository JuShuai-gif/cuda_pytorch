# 第8章：并发代码设计

> 拥有线程不等于拥有并发性能。本章教你如何把工作拆分成能真正并行执行的任务。

---

## 8.1 线程间划分工作

### 8.1.1 按数据划分（Data Parallelism / 同构并行）

同样的操作，作用在不同的数据块上。这是最直观的并行方式。

**生活类比——搬家**：你有一卡车家具要搬进新家。最自然的做法是：每个人负责不同的区域——张三搬卧室、李四搬客厅、王五搬厨房。每个人做同样的事（搬运），但操作不同的数据（不同房间的物品）。

```cpp
// 并行版 std::for_each
template<typename Iterator, typename Func>
void parallel_for_each(Iterator first, Iterator last, Func f) {
    unsigned long const length = std::distance(first, last);
    if (!length) return;

    unsigned long const min_per_thread = 25;
    unsigned long const max_threads =
        (length + min_per_thread - 1) / min_per_thread;

    unsigned long const hardware_threads =
        std::thread::hardware_concurrency();

    unsigned long const num_threads =
        std::min(hardware_threads != 0 ? hardware_threads : 2,
                 max_threads);

    unsigned long const block_size = length / num_threads;

    std::vector<std::thread> threads(num_threads - 1);
    Iterator block_start = first;

    for (unsigned long i = 0; i < (num_threads - 1); ++i) {
        Iterator block_end = block_start;
        std::advance(block_end, block_size);
        threads[i] = std::thread([=] {
            std::for_each(block_start, block_end, f);
        });
        block_start = block_end;
    }
    std::for_each(block_start, last, f);

    for (auto& t : threads) {
        t.join();
    }
}
```

### 8.1.2 按任务划分（Task Parallelism / 异构并行）

不同线程做不同的事，像流水线一样传递中间结果。

**生活类比——流水线分工**：装修队进场。水电工铺线、泥瓦工贴瓷砖、油漆工刷墙、木工装柜子。每个人的工作内容不同，但可以部分重叠：水电工在卧室干活时，泥瓦工已经在客餐厅开工了。

```cpp
// 流水线模式：线程1 处理数据 → 通过队列传给 线程2
void pipeline_example() {
    ThreadSafeQueue<RawData> stage1_queue;
    ThreadSafeQueue<ProcessedData> stage2_queue;

    std::thread reader([&] {
        while (auto data = read_from_source()) {
            stage1_queue.push(std::move(*data));
        }
    });

    std::thread processor([&] {
        while (auto raw = stage1_queue.try_pop()) {
            auto processed = heavy_compute(*raw);
            stage2_queue.push(std::move(processed));
        }
    });

    std::thread writer([&] {
        while (auto result = stage2_queue.try_pop()) {
            write_to_destination(*result);
        }
    });
}
```

### 8.1.3 递归划分（Divide and Conquer）

适用于树形结构或可以被二分的数据。

```cpp
// 并行快速排序（概念示意）
template<typename Iterator>
void parallel_quick_sort(Iterator begin, Iterator end) {
    auto const size = std::distance(begin, end);
    if (size <= 1024) {  // 阈值：太小就用串行
        std::sort(begin, end);
        return;
    }

    auto pivot = *std::next(begin, size / 2);
    auto middle1 = std::partition(begin, end,
        [=](auto const& v) { return v < pivot; });
    auto middle2 = std::partition(middle1, end,
        [=](auto const& v) { return v == pivot; });

    // std::async 可以偷懒：如果线程池满了就同步执行
    auto f1 = std::async(std::launch::async,
        parallel_quick_sort<Iterator>, begin, middle1);
    auto f2 = std::async(std::launch::async,
        parallel_quick_sort<Iterator>, middle2, end);

    f1.wait();
    f2.wait();
}
```

---

## 8.2 性能因素

### 8.2.1 处理器数量

`std::thread::hardware_concurrency()` 返回逻辑核数（含超线程）。创建超过此数量的"计算密集"线程只会增加上下文切换开销。

**经验法则**：计算密集型 = `hardware_concurrency()` 个线程；I/O 密集型 = 可以适当超额订阅。

### 8.2.2 数据竞争与乒乓缓存（Ping-Pong Cache）

当多个核心的缓存行在它们之间来回"弹跳"时，性能剧烈下降。

**生活类比**：两个人用对讲机通信，但一次只能一个人说话。如果两人同时抢着说话，对讲机不断切换发送方向，实际有效信息传输量极低——这就是乒乓缓存。

```
核心1: 读 cache line → 修改 → 使核心2的缓存失效
核心2: 读 cache line → 修改 → 使核心1的缓存失效
→ 循环往复，CPU 的大部分时间花在缓存一致性协议上
```

### 8.2.3 伪共享（False Sharing）

```cpp
// 坏设计：两个线程各写一个计数器，但它们在同一个缓存行内
struct BadDesign {
    std::atomic<int> counter_a{0};  // thread 1 写
    std::atomic<int> counter_b{0};  // thread 2 写
};

// 好设计：用填充隔开
struct GoodDesign {
    alignas(64) std::atomic<int> counter_a{0};
    alignas(64) std::atomic<int> counter_b{0};
};
```

**检测方法**：`perf stat -e cache-references,cache-misses` 或 Intel VTune。

### 8.2.4 数据邻近度（Data Proximity）

写贴近的数据比写分散的数据更快——对缓存友好。

```cpp
// 对缓存友好：连续访问
for (int i = 0; i < N; ++i)
    array[i] = f(i);   // 每次循环大概率命中缓存

// 对缓存不友好：跳跃访问
for (int i = 0; i < N; ++i)
    array[rand() % N] = f(i);  // 频繁 cache miss
```

### 8.2.5 过度订阅（Oversubscription）

运行的线程数超过硬件核心数，CPU 频繁切换上下文。使用线程池而非随意创建 `std::thread`。

---

## 8.3 为多线程设计数据结构

### 8.3.1 独享优于共享

每个线程尽可能有自己的"本地"数据副本，只在必要时合并。

```cpp
// 坏设计：所有线程竞争一个全局计数器
std::atomic<long long> global_counter(0);
void worker_bad() {
    for (int i = 0; i < 1000000; ++i)
        global_counter.fetch_add(1);  // 高竞争
}

// 好设计：每个线程本地计数，最后合并
void worker_good() {
    thread_local long long local_count = 0;
    for (int i = 0; i < 1000000; ++i)
        ++local_count;                // 无竞争
    // 最后所有线程的 local_count 汇总
}
```

### 8.3.2 数据布局优化

- 经常一起读取的字段放在同一缓存行
- 不同线程频繁写的字段放在不同缓存行
- 只读数据可以共享，不需同步

---

## 8.4 设计注意事项

### 8.4.1 异常安全

线程中抛出的未被捕获的异常会导致 `std::terminate`。必须在线程入口函数中处理所有异常。

RAII（资源获取即初始化）是并发代码异常安全的基石：

```cpp
class ScopedThread {
    std::thread t_;
public:
    template<typename Callable>
    explicit ScopedThread(Callable&& f) : t_(std::forward<Callable>(f)) {}

    ~ScopedThread() {
        if (t_.joinable()) {
            t_.join();  // 或 detach()，视需求
        }
    }
    ScopedThread(const ScopedThread&) = delete;
    ScopedThread& operator=(const ScopedThread&) = delete;
};
```

### 8.4.2 可扩展性与 Amdahl 定律

> **Amdahl 定律**：加速比 = 1 / (串行部分比例 + 并行部分比例 / N)

如果程序有 10% 的代码必须串行执行，那么即使有无限多个处理器，加速比也不会超过 10 倍。

```
S(N) = 1 / (S + (1 - S) / N)

S = 串行比例
N = 处理器数量

示例：S = 0.1（10% 串行），N → ∞
S(∞) = 1 / 0.1 = 10   ← 最多快 10 倍
```

**启示**：提高可扩展性的关键在于**减少串行部分**，而不是增加线程数。

### 8.4.3 隐藏延迟

用并发掩盖等待时间。这是异步编程的核心思想。

```cpp
// 串行：总时间 = disk1_read + disk2_read
auto data1 = read_file("a.txt");  // 等待磁盘
auto data2 = read_file("b.txt");  // 再等待磁盘

// 并发：总时间 ≈ max(disk1_read, disk2_read)
auto f1 = std::async(std::launch::async, read_file, "a.txt");
auto f2 = std::async(std::launch::async, read_file, "b.txt");
auto data1 = f1.get();
auto data2 = f2.get();
```

---

## 8.5 实践案例

### 8.5.1 parallel_find（并行查找，支持提前退出）

```cpp
template<typename Iterator, typename MatchType>
Iterator parallel_find(Iterator first, Iterator last, MatchType match) {
    struct FindElement {
        void operator()(Iterator begin, Iterator end,
                        MatchType match,
                        std::promise<Iterator>* result,
                        std::atomic<bool>* done_flag) {
            for (; begin != end && !done_flag->load(); ++begin) {
                if (*begin == match) {
                    result->set_value(begin);
                    done_flag->store(true);
                    return;
                }
            }
        }
    };

    unsigned long const length = std::distance(first, last);
    if (!length) return last;

    unsigned long const block_size = length / std::thread::hardware_concurrency();

    std::atomic<bool> done_flag(false);
    std::promise<Iterator> result_promise;
    std::future<Iterator> result_future = result_promise.get_future();
    std::vector<std::thread> threads;

    Iterator block_start = first;
    while (block_start != last) {
        Iterator block_end = block_start;
        std::advance(block_end,
            std::min(block_size,
                     static_cast<unsigned long>(
                         std::distance(block_start, last))));

        threads.emplace_back(FindElement(),
                             block_start, block_end,
                             match, &result_promise, &done_flag);
        block_start = block_end;
    }

    // 等待结果（第一个找到的）
    auto result = result_future.get();
    done_flag.store(true);  // 通知所有线程停止

    for (auto& t : threads) {
        t.join();
    }
    return result;
}
```

**关键设计点**：
- `std::promise` / `std::future` 传递"找到了"的结果
- `std::atomic<bool>` 做全局"停止"标志
- 找到后立即通知其他线程停止，避免无谓计算

### 8.5.2 parallel_partial_sum（并行部分和 / 前缀和）

```cpp
// 分两个阶段：
// Phase 1: 每个线程对自己负责的块做 local partial sum
// Phase 2: 计算块间偏移，加到每个块的元素上

template<typename Iterator>
void parallel_partial_sum(Iterator first, Iterator last) {
    using value_type = typename Iterator::value_type;
    const unsigned long length = std::distance(first, last);
    const unsigned long hardware = std::thread::hardware_concurrency();
    const unsigned long block_size = length / hardware;

    // 存储每块的最后一个值（作为下一块的偏移基数）
    std::vector<value_type> block_last(hardware + 1, 0);
    std::vector<std::thread> threads(hardware - 1);

    // Phase 1: 并行计算各块内部的 partial sum
    Iterator block_start = first;
    for (unsigned long i = 0; i < (hardware - 1); ++i) {
        Iterator block_end = block_start;
        std::advance(block_end, block_size);
        threads[i] = std::thread(
            [](Iterator begin, Iterator end, value_type* last) {
                value_type sum = 0;
                for (auto it = begin; it != end; ++it) {
                    sum += *it;
                    *it = sum;
                }
                *last = sum;  // 存储块内总和
            }, block_start, block_end, &block_last[i]);
        block_start = block_end;
    }

    // 主线程处理最后一块
    {
        value_type sum = 0;
        for (auto it = block_start; it != last; ++it) {
            sum += *it;
            *it = sum;
        }
        block_last[hardware - 1] = sum;
    }

    for (auto& t : threads) t.join();

    // Phase 2: 计算块间偏移并应用到每块
    value_type offset = 0;
    for (unsigned long i = 0; i < hardware; ++i) {
        value_type temp = block_last[i];
        block_last[i] = offset;
        offset += temp;
    }

    // 应用偏移到每块（这里简化，实际需要用多线程再次并行）
    block_start = first;
    for (unsigned long i = 0; i < (hardware - 1); ++i) {
        Iterator block_end = block_start;
        std::advance(block_end, block_size);
        if (block_last[i] != 0) {
            for (auto it = block_start; it != block_end; ++it) {
                *it += block_last[i];
            }
        }
        block_start = block_end;
    }
}
```

---

## 8.6 工业场景

### 并行矩阵运算
BLAS 库（如 OpenBLAS、MKL）将大矩阵分块交给多线程并行计算，是数据并行的经典应用。

### MapReduce
Google 的 MapReduce 模型就是"按数据划分"的极致体现：Map 阶段多节点并行处理各自数据分片，Reduce 阶段合并结果。

### 图像/视频处理管道
一帧图像的滤波操作（如模糊、边缘检测）天然适合数据并行：每个线程处理图像的若干行或若干块。

### HTTP 服务器
按任务划分：Acceptor 线程接收连接，Worker 线程池处理请求。是典型的 pipeline + 数据并行混合模式。

---

## 8.7 常见坑点

1. **伪共享（False Sharing）**：多线程看似写不同变量，实际写同一缓存行。解决：`alignas(64)` 隔离或填充。

2. **过度同步**：能用 `thread_local` 解决的统计，就不该用全局 `std::atomic`。能用无锁解决的就看优先级，别一上来就加锁。

3. **错误的划分策略**：数据划分时任务粒度过细（每个元素一个任务），调度开销淹没计算收益。一般建议每个任务至少几千到几万次操作。

4. **忽略 Amdahl 定律**：串行部分（如合并结果、日志写入）占比再小，也会最终成为瓶颈。优化了 99% 却没管那 1%，最多加速 100 倍——但实际上那 1% 往往才是最慢的。

5. **创建太多线程**：`hardware_concurrency()` 返回 8 就创建 8 个线程——但如果系统还有 20 个其他进程，过度订阅反而更慢。

---

## 8.8 面试常问

| 问题 | 要点 |
|------|------|
| 什么是 Amdahl 定律？ | 加速比上限 = 1 / 串行比例，串行部分决定最终加速极限 |
| 伪共享是什么？如何解决？ | 不同核心写同一缓存行；用 alignas 填充隔离 |
| 数据并行 vs 任务并行的区别？ | 数据并行：相同操作不同数据；任务并行：不同操作组成流水线 |
| 如何划分并行任务？ | 按数据（均分）、按任务（流水线）、递归（分治） |
| Thread pool 为何优于裸创建线程？ | 避免线程创建销毁开销、控制并发度、防止过度订阅 |
| 如何实现并行版本的 find？ | 分块查找 + atomic flag 提前退出 + promise/future 传递结果 |
| 如何减少共享数据？ | thread_local、每线程本地副本、消息传递而非共享内存 |

---

## 我应该掌握什么

- [ ] 能区分数据并行、任务并行、递归划分，并知道各自适用场景
- [ ] 能写出简化版的 `parallel_for_each` 和 `parallel_find`
- [ ] 理解 Amdahl 定律并能用它估算加速比上限
- [ ] 能解释伪共享的产生原因和解决方法（alignas 填充）
- [ ] 知道 `std::thread::hardware_concurrency` 的意义和局限性
- [ ] 理解乒乓缓存如何影响性能
- [ ] 能用 RAII 保证线程在异常时的安全退出
- [ ] 能设计出减少数据共享的并发方案（thread_local、消息传递）
- [ ] 理解用 `std::async` / `std::future` 隐藏 I/O 延迟的好处
