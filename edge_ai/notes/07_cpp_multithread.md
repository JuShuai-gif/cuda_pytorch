# C++ 多线程与实时系统编程

## 1. 线程生命周期

### 1.1 std::thread（C++11）

`std::thread` 是 C++11 引入的基础线程抽象。每个 `std::thread` 对象对应一个 OS 线程。

```cpp
#include <thread>
#include <iostream>

void worker(int id, const std::string &msg) {
    std::cout << "Thread " << id << ": " << msg << "\n";
}

std::thread t1(worker, 1, "hello");  // 立即启动
t1.join();                            // 等待线程结束

std::thread t2(worker, 2, "world");
t2.detach();                          // 分离，不再管理
// 注意：detach 后必须确保线程访问的资源仍然有效
```

**生命周期管理原则**：

- `join()` 或 `detach()` 必须在 `std::thread` 析构前调用，否则 `std::terminate`
- `joinable()` 检查线程是否可 join
- 推荐使用 RAII 包装，确保自动 join

```cpp
class ThreadGuard {
    std::thread &t;
public:
    explicit ThreadGuard(std::thread &t_) : t(t_) {}
    ~ThreadGuard() {
        if (t.joinable()) t.join();
    }
    ThreadGuard(const ThreadGuard&) = delete;
    ThreadGuard& operator=(const ThreadGuard&) = delete;
};
```

### 1.2 std::jthread（C++20）

`std::jthread` 解决了 `std::thread` 的两大痛点：析构时自动 join 和协作式取消。

```cpp
#include <thread>
#include <iostream>

void worker(std::stop_token stoken, int id) {
    while (!stoken.stop_requested()) {
        // 执行工作
        std::cout << "Worker " << id << " running\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::cout << "Worker " << id << " stopped\n";
}

std::jthread t(worker, 1);          // 自动 join
std::this_thread::sleep_for(std::chrono::seconds(1));
t.request_stop();                   // 请求停止
// t 析构时自动 join，不需要手动调用
```

**stop_token 机制**：

- `std::stop_source`：产生停止信号
- `std::stop_token`：检查是否已请求停止
- `std::stop_callback`：停止请求时执行回调

### 1.3 线程属性

```cpp
#include <pthread.h>

// 设置线程调度策略和优先级
pthread_attr_t attr;
pthread_attr_init(&attr);

// 显式创建 joinable 线程（默认）
pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

// 设置调度策略为 FIFO 实时调度
struct sched_param param;
param.sched_priority = 50;  // 1-99 for real-time
pthread_attr_setschedpolicy(&attr, SCHED_FIFO);
pthread_attr_setschedparam(&attr, &param);

// 设置栈大小（重要：避免栈溢出）
pthread_attr_setstacksize(&attr, 1024 * 1024);  // 1MB

pthread_t thread;
pthread_create(&thread, &attr, thread_func, nullptr);
pthread_attr_destroy(&attr);
```

**线程在不同阶段的约束**：

| 阶段 | 可设置属性 |
|------|------------|
| 创建前 | 分离状态、调度策略、优先级、栈大小、CPU 亲和性 |
| 运行中 | CPU 亲和性（pthread_setaffinity_np）、调度策略（root）、名称 |
| 结束后 | 不可修改，只能 join 获取返回值 |

## 2. 互斥锁类型

### 2.1 std::mutex

最基本的互斥锁，不可重入，不可超时。

```cpp
std::mutex mtx;
int shared_data = 0;

void increment() {
    mtx.lock();          // 阻塞直到获取锁
    shared_data++;
    mtx.unlock();        // 必须手动解锁（容易忘记）
}

// 推荐使用 RAII
void increment_safe() {
    std::lock_guard<std::mutex> lock(mtx);
    shared_data++;
}
```

### 2.2 std::recursive_mutex

允许同一线程多次 lock，需要对应次数的 unlock。

```cpp
std::recursive_mutex rmtx;

void recursive_func(int depth) {
    std::lock_guard<std::recursive_mutex> lock(rmtx);
    if (depth > 0) recursive_func(depth - 1);
}
// 使用场景：递归函数、多个成员函数相互调用且都需要加锁
// 不推荐过度使用：通常意味着设计可以优化
```

### 2.3 std::timed_mutex

支持带超时的 lock 尝试。

```cpp
std::timed_mutex tmtx;

void try_lock_with_timeout() {
    if (tmtx.try_lock_for(std::chrono::milliseconds(100))) {
        // 在 100ms 内获取到锁
        tmtx.unlock();
    } else {
        // 超时，执行降级逻辑
    }

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    if (tmtx.try_lock_until(deadline)) {
        tmtx.unlock();
    }
}
```

**应用场景**：避免死等——超时时执行降级逻辑、重试或报告错误。

### 2.4 std::shared_mutex（C++17）

读写锁：多个读者可同时持有，写者独占。

```cpp
std::shared_mutex smtx;
std::map<int, std::string> config;

// 读操作——共享锁
std::string get_config(int key) {
    std::shared_lock<std::shared_mutex> lock(smtx);
    return config[key];
}

// 写操作——独占锁
void set_config(int key, const std::string &val) {
    std::unique_lock<std::shared_mutex> lock(smtx);
    config[key] = val;
}
```

**性能对比**：

| 场景 | std::mutex | std::shared_mutex |
|------|-----------|-------------------|
| 100% 读 | 差 | 优——多读者并发 |
| 100% 写 | 略优 | 略差（额外开销） |
| 80% 读 / 20% 写 | 一般 | 优 |

对于机器人系统，传感器数据读取频繁但更新较少（如 IMU 校准参数），`shared_mutex` 是最佳选择。

## 3. 锁管理策略

### 3.1 std::lock_guard

最简单的 RAII 锁包装，构造时 lock，析构时 unlock。不可手动解锁。

```cpp
void critical_section() {
    std::lock_guard<std::mutex> lock(mtx);
    // 临界区代码
}  // lock 析构，自动解锁
```

### 3.2 std::unique_lock

灵活的 RAII 锁包装，支持延迟锁定、提前解锁、移动语义。

```cpp
std::mutex mtx;

// 延迟锁定
std::unique_lock<std::mutex> ulock(mtx, std::defer_lock);
// 做一些不需要锁的事情
ulock.lock();         // 手动锁定
// 临界区
ulock.unlock();       // 手动解锁

// 尝试锁定
if (ulock.try_lock()) {
    // 成功获取锁
}

// 配合条件变量
std::condition_variable cv;
cv.wait(ulock);       // wait 会原子地 unlock + 休眠
```

### 3.3 std::scoped_lock（C++17）

多锁同时锁定，避免死锁。内部使用 `std::lock` 的死锁避免算法。

```cpp
std::mutex mtx1, mtx2;

// Wrong: 可能导致死锁
void transfer_bad() {
    mtx1.lock();
    mtx2.lock();     // 如果另一线程以相反顺序锁定，死锁！
    // ...
}

// Correct: scoped_lock 保证无死锁
void transfer_good() {
    std::scoped_lock lock(mtx1, mtx2);
    // 原子地获取所有锁
}
```

### 3.4 锁的层级设计

```cpp
// 检查锁顺序的设计模式
class HierarchicalMutex {
    thread_local static uint64_t this_thread_prev_level;
    uint64_t level_;
public:
    explicit HierarchicalMutex(uint64_t level) : level_(level) {}
    void lock() {
        if (this_thread_prev_level <= level_) {
            throw std::logic_error("Lock order violation");
        }
        mtx_.lock();
        this_thread_prev_level = level_;
    }
    void unlock() {
        this_thread_prev_level = level_;
        mtx_.unlock();
    }
};
```

## 4. 原子操作与内存序

### 4.1 std::atomic 基础

```cpp
std::atomic<int> counter{0};

counter.fetch_add(1);                  // 原子递增
int old = counter.exchange(10);        // 原子交换
int expected = 0;
counter.compare_exchange_strong(expected, 1); // CAS

// 对 bool 的特殊支持
std::atomic<bool> flag{false};
flag.store(true);
bool v = flag.load();
```

### 4.2 内存序模型

C++ 定义了六种内存序，从弱到强：

```cpp
enum memory_order {
    memory_order_relaxed,   // 只保证原子性，无同步
    memory_order_consume,   // 消费语义（几乎不用）
    memory_order_acquire,   // 获取语义
    memory_order_release,   // 释放语义
    memory_order_acq_rel,   // 获取-释放语义
    memory_order_seq_cst    // 顺序一致性（默认，最强）
};
```

**relaxed——仅保证原子性**：

```cpp
std::atomic<int> cnt{0};

// 多线程执行，只保证 counter 最终正确，不保证任何内存可见性
cnt.fetch_add(1, std::memory_order_relaxed);
// 适合：简单的统计计数器
```

**acquire/release——发布-订阅同步**：

```cpp
std::atomic<bool> ready{false};
int data = 0;  // 非原子

// Thread 1 (producer)
data = 42;
ready.store(true, std::memory_order_release);  // 发布：之前的所有写可见

// Thread 2 (consumer)
while (!ready.load(std::memory_order_acquire)); // 获取：同步点
assert(data == 42);  // 保证为 true
```

**seq_cst——全局一致顺序**：

```cpp
// 所有 seq_cst 操作有一个全局的总顺序
// 开销最大（需要内存屏障），但最易理解
// 默认值，适合大多数场景
x.store(true);  // 等同于 x.store(true, memory_order_seq_cst)
```

### 4.3 内存序选择指南

| 场景 | 推荐内存序 |
|------|-----------|
| 简单计数器 | relaxed |
| 生产者-消费者（flag + data） | release / acquire |
| 多变量一致性 | seq_cst（默认） |
| 自旋锁实现 | acquire（lock）/ release（unlock） |
| 双重检查锁 | acquire / release |

### 4.4 Fence（栅栏）

```cpp
std::atomic<bool> flag{false};
int value = 0;

// Thread A
value = 100;
std::atomic_thread_fence(std::memory_order_release);  // 发布栅栏
flag.store(true, std::memory_order_relaxed);

// Thread B
while (!flag.load(std::memory_order_relaxed));
std::atomic_thread_fence(std::memory_order_acquire);  // 获取栅栏
assert(value == 100);  // 保证正确
```

## 5. 条件变量

### 5.1 基础用法

```cpp
std::mutex mtx;
std::condition_variable cv;
std::queue<int> data_queue;
bool done = false;

// Producer
void producer() {
    for (int i = 0; i < 100; i++) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            data_queue.push(i);
        }
        cv.notify_one();  // 唤醒一个消费者
    }
    {
        std::lock_guard<std::mutex> lock(mtx);
        done = true;
    }
    cv.notify_all();      // 唤醒所有消费者
}

// Consumer
void consumer(int id) {
    while (true) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [] { return !data_queue.empty() || done; });
        // wait 等价于：
        // while (!pred()) { cv.wait(lock); }

        if (!data_queue.empty()) {
            int val = data_queue.front();
            data_queue.pop();
            lock.unlock();  // 处理前释放锁
            // 处理数据（无需持锁）
        } else if (done) {
            break;
        }
    }
}
```

### 5.2 信号丢失与虚假唤醒

**信号丢失**：`notify_one()` 发生在 `wait()` 之前。

```cpp
// 错误示例：条件检查不在锁内
// Thread A                    Thread B
// if (!ready)                 ready = true;
//   cv.wait(lk);              cv.notify_one();
// → notify 在 wait 之前，信号丢失！

// 正确：条件变量必须与 mutex 配对保护条件
std::unique_lock<std::mutex> lock(mtx);
while (!ready) {               // 条件在锁内检查
    cv.wait(lock);
}
```

**虚假唤醒（Spurious Wakeup）**：

OS 可能在没有通知时唤醒线程，因此必须用 `while(!predicate)` 而非 `if(!predicate)`。

### 5.3 机器人系统中的典型应用

```cpp
// 传感器数据生产-消费
class SensorPipeline {
    std::mutex mtx_;
    std::condition_variable cv_;
    std::queue<LidarScan> scan_queue_;
    static constexpr size_t MAX_QUEUE = 10;

public:
    void produce(LidarScan scan) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return scan_queue_.size() < MAX_QUEUE; });
        scan_queue_.push(std::move(scan));
        lock.unlock();
        cv_.notify_one();
    }

    LidarScan consume() {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this] { return !scan_queue_.empty(); });
        LidarScan scan = std::move(scan_queue_.front());
        scan_queue_.pop();
        lock.unlock();
        cv_.notify_one();  // 通知生产者有空间了
        return scan;
    }
};
```

## 6. 线程池设计

### 6.1 核心组件

```cpp
class ThreadPool {
    std::vector<std::jthread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    bool stop_ = false;

public:
    ThreadPool(size_t num_threads = std::thread::hardware_concurrency()) {
        for (size_t i = 0; i < num_threads; i++) {
            workers_.emplace_back([this](std::stop_token stoken) {
                while (!stoken.stop_requested()) {
                    std::function<void()> task;
                    {
                        std::unique_lock lock(queue_mutex_);
                        cv_.wait(lock, [this, &stoken] {
                            return stop_ || !tasks_.empty()
                                || stoken.stop_requested();
                        });
                        if ((stop_ && tasks_.empty())
                            || stoken.stop_requested())
                            return;
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();
                }
            });
        }
    }

    template<typename F, typename... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>> {
        using ReturnType = std::invoke_result_t<F, Args...>;
        auto task = std::make_shared<std::packaged_task<ReturnType()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        std::future<ReturnType> result = task->get_future();
        {
            std::lock_guard lock(queue_mutex_);
            tasks_.emplace([task]() { (*task)(); });
        }
        cv_.notify_one();
        return result;
    }

    ~ThreadPool() {
        stop_ = true;
        cv_.notify_all();
        // jthread 自动 join
    }
};
```

### 6.2 工作窃取（Work Stealing）

每个工作线程维护自己的双端队列。空闲线程从繁忙线程的队列尾部"窃取"任务。

**优点**：

- 减少全局队列的竞争
- 自动负载均衡
- 递归并行任务（如分治法）执行效率高

**核心伪代码**：

```
task pop_task(worker_id):
    if local_deque[worker_id].not_empty():
        return local_deque[worker_id].pop_bottom()  // 自己的任务从底部取
    else:
        victim = random_other_worker()
        return victim.deque.steal_top()              // 窃取从顶部取
```

### 6.3 线程池大小选择

| 任务类型 | 推荐线程数 |
|----------|-----------|
| CPU 密集型 | `hardware_concurrency()` |
| IO 密集型 | `hardware_concurrency() * 2` 或更多 |
| 实时混合 | 预留 1-2 核心给实时线程 |
| GPU 辅助 | 1-2 线程负责 GPU 调度 |

## 7. 实时线程调度

### 7.1 Linux 调度策略

**SCHED_OTHER（CFS）**：

- 默认调度策略（时间片轮转的动态优先级）
- nice 值 -20 ~ +19（越小优先级越高）
- 不适合硬实时

**SCHED_FIFO**：

- 静态优先级 1~99，高优先级可抢占低优先级
- 运行直到主动 yield / 阻塞 / 被更高优先级抢占
- 没有时间片概念

**SCHED_RR**：

- 与 SCHED_FIFO 类似，但加入时间片轮转
- 同一优先级的线程轮流运行

```cpp
void set_realtime_priority(int priority) {
    struct sched_param param;
    param.sched_priority = priority;  // 1-99
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &param) != 0) {
        perror("pthread_setschedparam failed");
    }
    // 需要 CAP_SYS_NICE 权限或以 root 运行
}
```

### 7.2 优先级反转与优先级继承

```
场景（优先级：A 高 > B 中 > C 低）：
1. C 获取锁 L，被 A 抢占
2. A 尝试获取锁 L，阻塞（因为 C 持有）
3. B 抢占 C（B 优先级高于 C），导致 C 无法释放锁
4. A 被 B 阻塞，虽然 A 优先级最高
→ 这就是优先级反转！
```

**解决方案——优先级继承**：

```cpp
pthread_mutexattr_t attr;
pthread_mutexattr_init(&attr);
pthread_mutexattr_setprotocol(&attr, PTHREAD_PRIO_INHERIT);
// PTHREAD_PRIO_NONE: 不继承（默认）
// PTHREAD_PRIO_INHERIT: 持有者继承等待者的优先级
// PTHREAD_PRIO_PROTECT: 持有者运行在预设的最高优先级

pthread_mutex_t mtx;
pthread_mutex_init(&mtx, &attr);
```

### 7.3 CPU 亲和性

```cpp
void pin_to_core(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}
```

**机器人系统亲和性建议**：

- Core 0：OS 任务和中断
- Core 1-2：实时控制循环（运动控制）
- Core 3-5：传感器数据处理（视觉、激光雷达）
- Core 6-7：规划与决策

### 7.4 实时编程最佳实践

1. **避免动态内存分配**：预分配所有缓冲区
2. **禁止缺页中断**：`mlockall(MCL_CURRENT | MCL_FUTURE)` 锁住内存
3. **避免系统调用**：`open`、`write`、`malloc` 等可能阻塞
4. **禁用 swap**：或至少 swap 关键内存
5. **定时器选择**：`clock_nanosleep(CLOCK_MONOTONIC, ...)` 避免 wall-clock 跳变
6. **Watchdog**：硬件/软件看门狗检测死循环

## 8. 常见多线程 Bug

### 8.1 死锁（Deadlock）

两个或多个线程相互等待对方释放资源。

```cpp
// Deadlock example
std::mutex A, B;

// Thread 1: lock A → lock B
// Thread 2: lock B → lock A
// 同时发生 → 死锁！
```

**解决方案**：

- 加锁顺序一致
- `std::lock` 同时锁多个 mutex
- `std::scoped_lock`（C++17）
- 超时锁 + 回退重试

### 8.2 竞态条件（Race Condition）

多个线程并发访问共享数据，结果依赖执行顺序。

```cpp
// 经典 TOCTOU (Time-of-check to time-of-use)
std::map<int, std::string> registry;

// Thread 1
if (registry.find(key) == registry.end()) {  // 检查
    registry[key] = value;                   // 使用（中间可能被插入）
}

// Fix: 操作原子化
{
    std::lock_guard lock(mtx);
    registry.try_emplace(key, value);
}
```

### 8.3 ABA 问题

无锁编程中的经典问题：值从 A 变为 B 再变回 A，CAS 操作误以为没有变化。

```cpp
// ABA problem in lock-free stack
std::atomic<Node*> top;

// Thread 1 reads A (top = A)
// Thread 2 pops A, pops B, pushes A back
// Thread 1's CAS succeeds because top is still A
// but A's next pointer may now point to wrong node!

// Solution: tagged pointer (version counter + pointer)
struct TaggedPtr {
    Node* ptr;
    uint64_t tag;  // Incremented on each modification
};
std::atomic<TaggedPtr> top;
```

### 8.4 数据竞争（Data Race）

未通过同步机制保护的并发访问，C++ 标准下是未定义行为。

```cpp
// 即使 int 在硬件上是原子的，C++ 标准下仍是数据竞争
int counter = 0;
// Thread 1: counter++;  ← 数据竞争！
// Thread 2: counter++;

// 必须使用 std::atomic<int>
std::atomic<int> counter{0};
```

使用 ThreadSanitizer 检测：

```bash
g++ -fsanitize=thread -g -O1 program.cpp
./a.out
```

## 9. 机器人实时系统整合

### 9.1 典型架构

```
┌─────────────────────────────────────────┐
│          传感器 → 环形缓冲区 →            │
│   感知线程(SCHED_FIFO, core 1)            │
│              ↓ 无锁队列                   │
│   规划线程(SCHED_RR, core 2-3)            │
│              ↓ 共享内存                   │
│   控制线程(SCHED_FIFO, core 4, 1kHz)      │
│              ↓ 安全网关                   │
│   执行器通信线程(SCHED_FIFO, core 5)       │
└─────────────────────────────────────────┘
```

### 9.2 关键指标

| 指标 | 目标值 | 测量方法 |
|------|--------|---------|
| 控制循环抖动 | < 50μs | `clock_gettime` 记录周期性 |
| 端到端延迟 | < 10ms (感知→执行) | 时间戳追踪 |
| 锁持有时间 | < 100μs | 记录 lock/unlock 时间差 |
| 上下文切换 | < 1000/s | `perf stat -e context-switches` |

### 9.3 实用工具

```bash
# 查看实时线程的延迟
cyclictest -t 1 -p 80 -n -i 1000 -l 10000

# 查看线程调度信息
ps -eo pid,tid,class,rtprio,comm | grep robot

# 追踪线程事件
perf sched record -- ./robot_app
perf sched latency -s sort
```

