# 第 9 章：高级线程管理

## 章节概述

前面章节我们学会了创建线程、用互斥锁保护数据、用条件变量同步线程——这些都算"基础招式"。第 9 章开始进入**高级线程管理**领域，核心就两件事：**如何高效地管理一群线程**（线程池），以及**如何优雅地让正在跑的线程停下来**（线程中断）。

---

## 9.1 线程池（Thread Pool）

### 9.1.1 为什么要用线程池？

**生活类比**：开一家银行网点。如果每来一个客户你就当场招聘一个柜员，办完业务就把他开掉，那你的招聘成本会高得离谱，柜员刚上手就得走人，服务质量也差。

线程也一样——创建和销毁线程是需要系统调用的，开销不小。更关键的是，如果短时间内涌入 1000 个任务你就创建 1000 个线程，操作系统光做线程切换就累死了（这叫**上下文切换开销**），真正干活的时间反而很少。

线程池的思路：**提前招好一批柜员（工作线程）坐在柜台前等着，来一个客户就处理一个，处理完不离开，继续等下一个。**下班时间到了（程序结束），统一结账走人。

```cpp
// 朴素线程池的设计骨架
class thread_pool {
    std::atomic_bool done;                          // 停业标志
    threadsafe_queue<std::function<void()>> work_queue;  // 任务队列
    std::vector<std::thread> threads;               // 工作线程们
    // ...
};
```

核心组成三要素：
1. **任务队列**（work queue）：客户排队的等候区
2. **工作线程组**（worker threads）：坐在柜台后面的柜员
3. **同步机制**（条件变量 + 互斥锁）：叫号系统

### 9.1.2 简单线程池设计

最朴素的实现：启动时开 N 个线程，每个线程的 `run` 函数就是一个无限循环——从队列取任务，执行，再取下一个。

```cpp
class simple_thread_pool {
    std::atomic_bool done;
    std::queue<std::function<void()>> work_queue;
    std::mutex mtx;
    std::condition_variable cv;
    std::vector<std::thread> workers;

    void worker_loop() {
        while (!done) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [this] { return done || !work_queue.empty(); });
                if (done && work_queue.empty()) return;
                task = std::move(work_queue.front());
                work_queue.pop();
            }
            task();  // 不在锁内执行，提高并发度
        }
    }

public:
    simple_thread_pool(size_t num_threads = std::thread::hardware_concurrency()) {
        for (size_t i = 0; i < num_threads; ++i)
            workers.emplace_back(&simple_thread_pool::worker_loop, this);
    }

    template<typename F>
    void submit(F f) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            work_queue.push(std::function<void()>(f));
        }
        cv.notify_one();
    }

    ~simple_thread_pool() {
        done = true;
        cv.notify_all();
        for (auto& t : workers) t.join();
    }
};
```

这里有个重要细节：`task()` 在**释放锁之后**才调用。如果拿着锁执行任务，所有线程就排他性地干活了，你线程池的意义直接归零。

### 9.1.3 等待提交的任务——future 返回

上面的设计有一个致命缺陷：`submit` 只负责"扔任务"，不给你任何反馈。如果我想知道任务执行完了没有、或者想拿返回值，怎么办？

这就需要用 `std::packaged_task<>` 配对 `std::future<>`：

```cpp
template<typename F>
auto submit(F f) -> std::future<decltype(f())> {
    // packaged_task 包装可调用对象，生成 future
    auto task = std::make_shared<std::packaged_task<decltype(f())()>>(std::move(f));
    std::future<decltype(f())> res = task->get_future();
    {
        std::lock_guard<std::mutex> lock(mtx);
        work_queue.push([task]() { (*task)(); });
    }
    cv.notify_one();
    return res;
}
```

**生活类比**：这次你去银行不只是"交材料"，而是一张"回执单"（future）。你拿着回执单可以去旁边喝杯咖啡，等你需要结果的时候回柜台看一眼——如果业务办完了，结果就在那里；如果没办完，你就在那里等。

`std::packaged_task` 就像一个信封，你把任务装进去，它就自动生成一个回执（future）。任务执行完，结果自动投递到 future 里。

### 9.1.4 可等待任务的线程池

有了 future 返回，调用方就可以选择：
- **非阻塞检查**：`future.wait_for(0s)` —— 看一眼，有了拿结果，没有继续忙别的
- **阻塞等待**：`future.get()` —— 不拿到结果不走

一个完整的可等待线程池通常还要处理：
- `submit` 的模板参数推导（用 `decltype` 之类的技巧）
- 任务队列中存储 `std::function<void()>` 的类型擦除
- `shared_ptr` 管理 `packaged_task` 的生命周期（因为 lambda 捕获的是副本）

### 9.1.5 线程池的停止机制

这是线程池设计中最容易出 bug 的地方。

**错误的做法**：`done = true` 之后立刻 `join`。但此时线程可能正在 `wait`，也可能正在执行一个耗时很长的任务。你需要确保：
1. 通知所有线程"下班了"
2. 线程把**当前正在执行的任务做完**（不能杀一半）
3. 线程检查队列里的**剩余任务也不再处理**
4. 等所有线程都退出后再析构

```cpp
~thread_pool() {
    done = true;           // 1. 挂出停业牌
    cv.notify_all();       // 2. 叫醒所有在等的柜员
    for (auto& t : workers) {
        if (t.joinable()) t.join();  // 3. 等每个人把手头的事办完
    }
}
```

**生活类比**：银行关门铃响了。已经在柜台办业务的客户（正在执行的任务）要办完才能走，但门外排队的新客户（队列中的任务）就请明天再来了。你不能把正在签字的客户赶出去。

### 9.1.6 工作窃取（Work Stealing）

这是线程池的高级优化技术，解决"忙的忙死，闲的闲死"问题。

**生活类比**：银行有 3 个柜台。A 柜台排了 20 个人，B 和 C 柜台门可罗雀。B 柜员的经理过来说："你去帮 A 分担一点。"于是 B 走到 A 那边，从他队伍的**尾部**拿了几份材料过来处理。这就是 work stealing。

技术实现：
- 每个工作线程**都有自己的任务队列**（而非全局共享一个）
- 线程优先从自己的队列取任务（LIFO，后进先出，类似栈）
- 当自己的队列空了，就随机挑一个"倒霉同事"，**从他的队列尾部偷任务**（FIFO，先进先出）
- 被偷的线程不受影响，因为它从头部取，偷窃者从尾部取

为什么本地队列用 LIFO？**缓存局部性**。刚放进去的任务数据很可能还在 CPU 缓存里，从尾部取正好是最"热"的那个。

```cpp
// 工作窃取的简化概念代码
class work_stealing_queue {
    // 线程自己操作：从头部 push/pop（LIFO）
    void push_front(task);    // 自己加任务
    task pop_front();         // 自己取任务

    // 偷窃者操作：从尾部偷（FIFO）
    task steal_back();        // 别人来偷，从尾部拿
};
```

**工业场景**：
- Intel TBB（Threading Building Blocks）的任务调度器就是基于 work stealing
- Java ForkJoinPool 也是同样的设计
- C++ 的 `std::execution::par` 底层很多实现也用了 work stealing

---

## 9.2 中断线程

### 9.2.1 为什么 C++ 标准库没有直接的中断机制？

POSIX 有 `pthread_cancel`，Java 有 `Thread.interrupt()`，但 C++11 的 `std::thread` 故意没有提供"强制终止"的接口。原因是：**强制杀线程可能导致资源泄漏、锁未释放、数据不一致**——因为你在线程执行的任意位置把它杀了，它没机会做清理。

**生活类比**：你正在切菜，别人突然从背后把你拽走。菜刀掉桌上还好，掉脚上呢？厨房里的煤气灶谁来关？——所以不能暴力拉人，你得说"先停一下，收个尾"。

C++ 的思路是：**协作式中断**——不是"杀掉你"，而是"跟你说一声能不能停一下"。被中断的线程需要**主动检查**中断标志，在安全的时机自己停下来。

### 9.2.2 中断点（Interruption Point）的概念

中断点是线程在执行过程中**主动检查是否被中断**的位置。只有在中段点，线程才有机会响应中断请求。

常见的中断点：
- 等待条件变量时
- 等待 future 时
- 执行 `sleep` 时
- 任何显式调用 `interruption_point()` 的地方

```cpp
// 概念：中断点的伪代码
void interruption_point() {
    if (this_thread_is_interrupted()) {
        throw thread_interrupted();  // 抛出中断异常，栈回退
    }
}
```

### 9.2.3 interruptible_thread 的实现

核心设计：
1. 每个可中断线程持有一个 `interrupt_flag`（中断标志）
2. `interrupt()` 方法设置该标志
3. 线程在各个阻塞点检查标志并响应

```cpp
class interrupt_flag {
    std::atomic<bool> flag{false};
    std::condition_variable* cv{nullptr};
    std::condition_variable_any* cv_any{nullptr};
    std::mutex mtx;
public:
    void set() {
        flag.store(true);
        std::lock_guard<std::mutex> lock(mtx);
        if (cv) cv->notify_all();
        if (cv_any) cv_any->notify_all();
    }
    bool is_set() const { return flag.load(); }

    // 注册/注销条件变量，让 set() 能唤醒它们
    void set_cv(std::condition_variable* c) { cv = c; }
    void clear_cv() { cv = nullptr; }
    // ... cv_any 同理
};

class interruptible_thread {
    std::thread internal_thread;
    std::shared_ptr<interrupt_flag> flag;

    template<typename F>
    static void wrapper(std::shared_ptr<interrupt_flag> flag, F f) {
        // 把 flag 存入线程局部存储（thread_local）
        this_thread_interrupt_flag = flag.get();
        try {
            f();  // 执行用户函数
        } catch (thread_interrupted&) {
            // 中断异常到达这里，正常退出
        }
    }

public:
    template<typename F>
    interruptible_thread(F f)
        : flag(std::make_shared<interrupt_flag>())
        , internal_thread(wrapper<F>, flag, std::move(f)) {}

    void interrupt() { flag->set(); }
    void join() { internal_thread.join(); }
    // ...
};
```

设计要点：
- `interrupt_flag` 用 `shared_ptr` 共享，因为线程函数和外部都要访问它
- 用 `thread_local` 存储当前线程的中断标志指针，这样 `interruption_point()` 可以在任何地方被调用
- 中断通过异常（`thread_interrupted`）向上传播，利用 RAII 自动清理资源

### 9.2.4 在条件变量等待时中断

这是中断实现中最精妙的部分。`condition_variable::wait` 本身是阻塞的，怎么中断？

**问题**：线程正在 `cv.wait(lock)` 上睡觉，你设了中断标志它也不知道。

**解法**：把中断标志关联到条件变量上。当 `interrupt()` 被调用时，不仅设标志，还主动 `notify` 条件变量，把线程唤醒。线程醒了之后在 `wait` 的谓词里检查中断标志，发现被中断了，抛出异常。

```cpp
// 中断感知的 wait 包装
template<typename Predicate>
void interruptible_wait(std::condition_variable& cv,
                        std::unique_lock<std::mutex>& lock,
                        Predicate pred) {
    interruption_point();  // 进 wait 前先检查
    auto* flag = this_thread_interrupt_flag;
    flag->set_cv(&cv);     // 注册条件变量

    // 用 wait_for 循环模拟 wait，每次醒来检查中断
    while (!pred()) {
        if (cv.wait_for(lock, std::chrono::milliseconds(1)) ==
            std::cv_status::timeout) {
            // 超时醒来，循环回去再检查 pred 和中断标志
        }
        interruption_point();
    }
    flag->clear_cv();      // 注销
}
```

关键技巧：`wait_for` 带超时的 wait，这样即使 `notify` 丢失了，线程定期也会醒来检查中断标志。

### 9.2.5 在 std::condition_variable_any 等待时中断

`condition_variable_any` 比 `condition_variable` 更灵活——它可以用任何满足 Lockable 要求的锁，不限于 `std::mutex`。

处理思路与上面类似，但因为 `condition_variable_any` 的内部实现差异，需要对应的 `set_cv_any` / `clear_cv_any` 重载。

### 9.2.6 中断其他阻塞调用

并非所有阻塞调用都能中断。能中断的：
- 条件变量等待 ✓
- `std::future::wait` ✓（future 本身有定时版本可轮询）
- `std::this_thread::sleep_for` ✓（用 sleep 的小片段循环替代）

不能直接中断的：
- 被互斥锁阻塞（`lock()` 不返回就不能检查中断点） ✗
- 文件 I/O 阻塞 ✗（这是操作系统管的事）
- 网络 I/O 阻塞 ✗

对于这些情况，要么在调用前先检查中断标志，要么改用带超时的版本（`try_lock_for` 等）。

---

## 工业场景

| 场景 | 对应技术 |
|------|----------|
| **Web 服务器线程池** | 主线程 accept 连接，把 socket 提交到线程池，工作线程处理 HTTP 请求。线程数一般设为 CPU 核数×2（混合 I/O 密集型） |
| **AI 推理任务调度** | GPU 推理任务排队，线程池中每个线程绑定一个 CUDA stream，work stealing 平衡多 GPU 负载 |
| **消息处理管道** | 多阶段流水线，每个阶段一个线程池，阶段间用有界队列连接，形成生产者-消费者链 |

---

## 常见坑点

1. **任务抛异常导致线程静默死亡**
   - 在 `worker_loop` 中用 `try-catch` 包裹 `task()` 调用，记录日志但不要让线程退出
2. **析构顺序反了**
   - 必须先设 `done` 通知所有线程，再 `join` 等它们退出，最后销毁队列。如果先销毁队列而线程还活着 → UB
3. **用全局队列导致锁竞争**
   - 线程数 > 核数时，全局队列的互斥锁会成为瓶颈。work stealing 的多队列设计能显著缓解
4. **`notify_one()` vs `notify_all()`**
   - `submit` 中应该用 `notify_one()`，因为只需要唤醒一个空闲线程。用 `notify_all()` 会造成惊群效应（thundering herd）——所有线程同时醒来抢锁，只有一个拿到任务，其他的又回去睡觉，白白浪费 CPU
5. **虚假唤醒没处理好**
   - `cv.wait(lock, predicate)` 用谓词版本，别用无谓词版本。虚假唤醒在 POSIX 系统上确实会发生

---

## 面试常问

**Q：如何设计一个线程池？**
- 先说三要素：任务队列、工作线程、同步机制
- 再说进阶特性：future 返回、优雅停止、work stealing
- 最后谈参数调优：线程数怎么定（CPU 密集型=核数，I/O 密集型=核数×2~4）

**Q：work stealing 的原理？**
- 每个线程有独立的双端队列
- 自己用 LIFO（局部性），别人偷用 FIFO（减少竞争）
- 随机选择受害者或轮询
- 优点：负载均衡、减少全局锁竞争

**Q：如何安全地停止一个线程池？**
- 协作式中断，不强制杀线程
- `done` 标志 + `notify_all` 唤醒所有线程
- 线程检查 `done` 标志后退出循环
- 主线程 `join` 所有工作线程
- 任务异常要隔离，不能因为一个任务崩了整个线程

---

## 我应该掌握什么

- [ ] 线程池的核心三要素是哪三个
- [ ] 为什么 `submit` 需要用 `packaged_task` + `future`
- [ ] 线程池正确停止的 4 个步骤
- [ ] work stealing 的双端队列操作规则（LIFO 本地、FIFO 偷窃）
- [ ] 什么是协作式中断，为什么 C++ 不提供强制终止
- [ ] `interrupt_flag` 如何与条件变量联动实现等待时中断
- [ ] `thread_local` 在中断机制中的作用
- [ ] 线程池析构时常见的顺序错误是什么
- [ ] `notify_one` vs `notify_all` 在 `submit` 中的选择
- [ ] 能独立写出一个带 future 返回的简易线程池
