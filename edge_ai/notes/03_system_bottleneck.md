# 系统瓶颈识别与优化

## 1. 线程调度陷阱

### 1.1 优先级反转（Priority Inversion）

**定义**：高优先级线程等待低优先级线程持有的锁，而低优先级线程又被中优先级线程抢占。

**经典场景**：
- 线程 A（高优先级）等待线程 C（低优先级）释放锁
- 线程 B（中优先级）抢占线程 C，导致 C 无法释放锁
- A 无限期等待 B，虽然 A 优先级高于 B

**解决方案**：
- **优先级继承（Priority Inheritance）**：持锁的低优先级线程临时继承等待者的高优先级
- **优先级天花板（Priority Ceiling）**：一把锁被赋予系统中需要它的最高线程优先级，持锁期间运行在该优先级
- Linux `pthread_mutexattr_setprotocol()` 支持 `PTHREAD_PRIO_INHERIT` 和 `PTHREAD_PRIO_PROTECT`

### 1.2 护航效应（Convoy Effect）

**定义**：一个长时间持有锁的线程导致大量等待该锁的线程堆积，释放锁后这些线程依次执行，造成资源利用率波动。

**表现**：CPU 利用率呈锯齿状——锁释放瞬间峰值，其他时间低负载。

**缓解**：
- 缩小临界区：只锁住必须互斥的最小代码段
- 读写锁分离：读多写少的场景用 `shared_mutex` (rwlock)
- RCU（Read-Copy-Update）：读操作完全无锁
- 无锁数据结构：用 CAS 等原子操作替代锁

### 1.3 假唤醒与条件变量

```cpp
// Wrong: may miss wakeup
if (!data_ready) cv.wait(lk);

// Correct: always use while loop
while (!data_ready) cv.wait(lk);
```

### 1.4 线程亲和性与隔离

```cpp
cpu_set_t cpuset;
CPU_ZERO(&cpuset);
CPU_SET(2, &cpuset);  // Pin to core 2
pthread_setaffinity_np(thread, sizeof(cpuset), &cpuset);
```

将关键线程绑定到专用核心，并将中断（IRQ）通过 `/proc/irq/*/smp_affinity` 导向其他核心。

## 2. 锁竞争：类型与检测

### 2.1 自旋锁（Spinlock）

- **原理**：忙等待循环检查锁，不进入内核态
- **适用**：临界区极短（纳秒~微秒级），锁持有时间 < 两次上下文切换开销
- **缺陷**：长时间持锁时浪费 CPU 且降低吞吐量
- `pthread_spin_lock()` / `std::atomic_flag` 实现

### 2.2 互斥锁（Mutex / std::mutex）

- **原理**：竞争失败时线程进入睡眠（内核态），释放锁时唤醒
- **适用**：临界区较长（毫秒级），或不确定持锁时间
- **开销**：上下文切换 ~ 1μs ~ 10μs
- **Linux futex**：`futex()` 系统调用，无竞争时在用户态完成

### 2.3 读写锁（RWLock / std::shared_mutex）

- **原理**：多个读者可以并发，写者独占
- **适用**：读多写少的场景
- **代价**：rwlock 本身的元数据操作比普通 mutex 重
- **注意**：写入频繁时性能反而差于普通 mutex（写入饥饿问题）

### 2.4 选择决策表

| 临界区长度 | 读多写少 | 读写均衡 | 写多读少 |
|-----------|---------|---------|---------|
| < 100ns | 无锁/原子 | 无锁/原子 | 无锁/原子 |
| 100ns ~ 1μs | Spinlock | Spinlock | Spinlock |
| 1μs ~ 100μs | RWLock | Mutex | Mutex |
| > 100μs | RWLock | Mutex | Mutex + 缩小临界区 |

### 2.5 锁竞争检测工具

- `perf lock`：内核锁竞争分析
- `perf record -e lock:lock_acquire -g`：跟踪锁获取事件
- `pthread_mutex_t` 的 `PTHREAD_MUTEX_ERRORCHECK` 属性检测死锁
- Intel VTune / AMD uProf 的锁等待分析

## 3. 内存拷贝开销

### 3.1 `memcpy` 的开销

- 小数据（< 128B）：函数调用开销占主导，可考虑编译期内联或直接赋值
- 中等数据（128B ~ 4KB）：SIMD 加速（AVX-512 单次 64B），接近内存带宽
- 大数据（> 4KB）：带宽饱和，非临时（non-temporal）存储可避免缓存污染

### 3.2 零拷贝（Zero-Copy）技术

| 技术 | 原理 | 适用场景 |
|------|------|---------|
| `sendfile()` | 内核态直接将文件数据传送到 socket，无需用户态拷贝 | 静态文件 HTTP 服务 |
| `splice()` | 内核态管道间数据转移 | 代理/中转服务 |
| 共享内存 (mmap) | 多个进程映射同一物理页 | 进程间大数据通信 |
| DMA 直接传输 | 外设直接读写内存 | GPU 数据传输、NVMe 读写 |
| `io_uring` | 用户态环形缓冲区提交 I/O 请求 | 高并发 I/O |
| BPF / XDP | 网卡驱动层直接处理数据包 | 高性能网络 |

### 3.3 何时拷贝不可避免

- 数据格式转换（序列化/反序列化）
- 安全隔离（跨信任边界）
- 并发安全（发送方可能修改原数据）
- 缓存局部性优化（数据重排布）

## 4. 缓存命中率与伪共享

### 4.1 缓存层次结构

| 缓存级别 | 大小 | 延迟 | 带宽 |
|---------|------|------|------|
| L1 Data | 32KB | ~1ns (4 cycles) | ~2 TB/s |
| L2 | 256KB ~ 1MB | ~4ns (12 cycles) | ~800 GB/s |
| L3 (LLC) | 8MB ~ 64MB | ~12ns (40 cycles) | ~400 GB/s |
| DRAM | 16GB ~ 512GB | ~100ns | ~50 GB/s |
| NVMe SSD | - | ~10μs | ~7 GB/s |

**规律**：每远离 CPU 一级，延迟增加 ~3-10x，带宽下降 ~2-5x。

### 4.2 伪共享（False Sharing）

**定义**：两个线程各自频繁修改同一缓存行（通常 64 字节）内的不同变量，导致缓存一致性协议（MESI）不断使对方缓存行失效。

```cpp
// Problem: a and b are on the same cache line
struct Bad {
    int a;          // Thread 1 modifies this
    int b;          // Thread 2 modifies this
};

// Fix: padding to separate cache lines
struct Good {
    alignas(64) int a;   // Thread 1 modifies this
    alignas(64) int b;   // Thread 2 modifies this - different cache line
};
```

**检测**：`perf stat -e cache-misses,cache-references` 观察 cache miss 率。

### 4.3 缓存抖动（Cache Thrashing）

**矩阵遍历顺序的经典案例**：

```cpp
// Bad: Column-major traversal of row-major matrix
// Each access jumps 1 row (e.g., 1024 * 4 bytes), likely cache miss
for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
        sum += matrix[j][i];  // Cache hostile

// Good: Row-major traversal of row-major matrix
// Consecutive accesses hit the cache line
for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
        sum += matrix[i][j];  // Cache friendly
```

性能差异可达 **10x ~ 50x**（取决于矩阵大小与缓存大小的关系）。

## 5. GPU 空闲模式

### 5.1 数据依赖停滞

GPU kernel 等待上游 kernel 的输出完成。表现为 GPU SM 利用率低但内存控制器忙碌。

**缓解**：在等待期间发射不依赖的计算任务（Occupancy 调度）。

### 5.2 Kernel 启动开销

每个 CUDA kernel launch 有 ~5-20μs 的开销（CPU→GPU 命令提交 + GPU 调度）。

**缓解**：
- 合并小 kernel 为一个大 kernel（kernel fusion）
- 使用 CUDA Graph 预录制并重放命令序列
- 减少 kernel 数量，避免过度拆分

### 5.3 GPU 利用率误区

`nvidia-smi` 显示的 GPU 利用率 = SM 活动周期占比，不等于计算效率。
- 100% 利用率 ≠ 高效：如果 kernel 是访存密集型的，SM 大量时间在等数据
- < 50% 利用率 ≠ 低效：如果 kernel 计算密集且数据已就绪

## 6. 使用 perf stat 检测缓存问题

```bash
# 核心指标
perf stat -e cycles,instructions,cache-references,cache-misses,branch-misses,L1-dcache-load-misses,LLC-load-misses ./program

# 关键推导指标
# IPC = instructions / cycles (> 2 为佳, < 0.5 说明大量停顿)
# Cache miss rate = cache-misses / cache-references
# Branch mispredict rate = branch-misses / branches
```

**典型问题信号**：
- IPC < 0.5 且 L1 miss rate > 10%：数据布局问题
- IPC < 0.5 且 branch miss rate > 5%：分支预测失败多
- LLC miss rate > 5%：工作集超过缓存大小

## 7. 常见性能反模式

| 反模式 | 表现 | 修复 |
|--------|------|------|
| 循环中动态分配内存 | `malloc()` 调用占比高 | 预分配池，复用内存 |
| 全局锁保护热点路径 | 锁竞争占比 > 10% CPU | 无锁数据结构、per-thread 数据 |
| 虚函数在热循环中调用 | 间接跳转导致分支预测失败 | CRTP、模板替代虚函数 |
| 过度使用 `shared_ptr` | 原子引用计数开销 | `unique_ptr` + 裸指针引用 |
| 日志在热路径中 | I/O 阻塞关键路径 | 异步日志队列、采样日志 |
| 小 I/O 频繁操作 | 系统调用开销占比高 | 批量攒批、`io_uring` |
| 锁内做 I/O | 长时间持锁阻塞所有线程 | I/O 移到锁外 |
| 数据结构选择不当 | `std::list` 遍历导致 cache miss | 改为 `std::vector`（连续内存） |

**核心思维**：性能优化的第一步永远是**测量**。在优化之前，用 profiler 数据定位真正的瓶颈，而不是凭直觉猜测。Amdahl 定律告诉我们：优化占比 1% 的代码，最多只能提升 1% 的整体性能。
