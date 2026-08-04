# 第12章：现代 C++20 并发新特性

> C++20 为并发编程带来了大量新特性，从更安全的线程管理到更高效的同步原语。本章系统讲解 jthread、stop_token、semaphore、latch、barrier 和协程基础。

---

## 12.1 std::jthread — 自带 Join 的线程

### 原理

`std::jthread`（C++20）是 `std::thread` 的改进版，核心区别：

1. **自动 join**：析构时自动调用 `join()`，不会像 `std::thread` 那样析构未 join 的线程导致 `std::terminate`
2. **内置停止机制**：通过 `std::stop_token` 支持协作式线程取消
3. **可中断等待**：`condition_variable_any::wait` 可通过 stop_token 中断

```cpp
// std::thread: 忘记 join/detach 会 crash
{ std::thread t([]{ /* ... */ }); } // core dump!

// std::jthread: 自动 join，安全
{ std::jthread t([]{ /* ... */ }); } // 自动 join，安全
```

### 适用场景

- 所有需要 `std::thread` 的场景（可完全替代）
- 需要优雅取消的长时间运行任务
- RAII 管理线程生命周期

### 优缺点

| 优点 | 缺点 |
|------|------|
| 自动 join，防止资源泄露 | 析构时 join 可能会阻塞主线程 |
| 内置 stop_token 支持 | 性能开销极小但非零 |
| 接口完全兼容 std::thread | — |

### 常见错误

- 误以为 jthread 析构是即时的——它会阻塞直到线程完成
- 在 jthread 已 join 后再次 join

---

## 12.2 std::stop_token — 协作式线程取消

### 原理

C++20 引入了**协作式取消**框架，核心组件：

- `std::stop_source`：停止信号的**发送者**，可向关联的 stop_token 发出停止请求
- `std::stop_token`：停止信号的**接收者**，可查询是否被请求停止
- `std::stop_callback`：停止时的**回调注册**

```cpp
std::stop_source source;
std::jthread worker([token = source.get_token()]() {
    while (!token.stop_requested()) {
        // 工作循环
    }
});

source.request_stop(); // 请求停止
worker.join();         // 等待线程结束
```

**生活类比**：消防警报系统。`stop_source` 是拉响警报的按钮，`stop_token` 是每个房间的警铃——按下按钮后所有警铃同时响起。`stop_callback` 相当于"听到警报后自动关燃气"的联动装置。

### 适用于 condition_variable 的停止

```cpp
std::condition_variable_any cv;
std::mutex mtx;

std::stop_source source;
std::jthread worker([&](std::stop_token token) {
    std::unique_lock lock(mtx);
    cv.wait(lock, token, []{ return false; }); // 可被 stop_token 中断
});
```

### 常见错误

- 忘记传入 stop_token：`std::jthread t([&]() { ... })` 不会自动获得 stop_token
- 将 stop_requested 检查放在耗时操作中间无法及时响应

---

## 12.3 std::counting_semaphore / std::binary_semaphore

### 原理

信号量（Semaphore）是最古老的同步原语之一（Dijkstra, 1965），C++20 终于将其标准化：

- `std::counting_semaphore<N>`：计数信号量，内部计数器最大值为 N
- `std::binary_semaphore`：`std::counting_semaphore<1>` 的别名，等价于布尔信号量

核心操作：
- `acquire()`：计数器减 1，若为 0 则阻塞
- `release(n)`：计数器加 n，唤醒等待线程

```cpp
std::counting_semaphore<5> slots{5}; // 最多 5 个并发资源

void worker() {
    slots.acquire();          // 获取资源
    // 使用资源 ...
    slots.release();          // 归还资源
}
```

**生活类比**：停车场。有 5 个车位（`counting_semaphore<5>`），每进一辆车 `acquire` 一次（车位-1），每出一辆车 `release` 一次（车位+1）。车位满时，后来的车只能排队等待。

### 适用场景

- 限制并发访问数量（如数据库连接池）
- 生产者-消费者（比条件变量更简洁）
- 限流器（Rate Limiter）
- 实现 Barrier / Latch 的底层原语

### 与 mutex + condition_variable 对比

| | semaphore | mutex + cv |
|---|---|---|
| 复杂度 | 低 | 中 |
| 性能 | 通常更高（内核无竞争时走 fast path） | 相当 |
| 表达能力 | 计数型同步 | 任意条件同步 |
| 标准 | C++20 | C++11 |

---

## 12.4 std::latch — 一次性同步点

### 原理

`std::latch` 是一个**一次性**倒计数器：

- 初始化时设定一个计数 N
- 线程调用 `arrive_and_wait()` 或 `count_down()` 使计数减 1
- 当计数减到 0 时，所有等待线程被释放
- 不可重置，用完即弃

```cpp
std::latch done{3}; // 等待 3 个任务完成

for (int i = 0; i < 3; ++i) {
    std::thread([&, i]() {
        do_work(i);
        done.count_down(); // 标记完成
    }).detach();
}

done.wait(); // 阻塞直到 3 个任务全部完成
```

**生活类比**：起跑倒计时。裁判喊"3-2-1-跑！"——所有运动员在"跑"之前都在等待。`latch(3)` 的计数减到 0 就是那声"跑"。

### 适用场景

- 等待一组线程全部初始化完成
- 多阶段流水线中等待当前阶段所有参与者就绪
- 一次性门控（gate）

---

## 12.5 std::barrier — 可重用同步点

### 原理

`std::barrier` 是**可重用**的同步屏障：

- 设定一个预期到达数量 N
- 线程调用 `arrive_and_wait()` 阻塞，直到 N 个线程都到达
- 所有线程被同时释放，barrier 自动重置可用

关键是 barrier 有一个**完成回调（completion function）**——每次屏障打开时自动调用，可用于阶段轮转。

```cpp
std::barrier sync{4}; // 4 个线程同步

void worker(int id) {
    for (int phase = 0; phase < 10; ++phase) {
        do_phase_work(id, phase);
        sync.arrive_and_wait(); // 所有线程在此同步
    }
}
```

**生活类比**：网络游戏的回合制。每回合所有玩家提交操作后，服务器统一结算——这就是 barrier。下一回合 barrier 自动重置，继续使用。

### latch vs barrier vs semaphore (选择指南)

| 特性 | latch | barrier | semaphore |
|------|-------|---------|-----------|
| 可重用 | 否 | 是 | 是 |
| 计数方向 | 递减 | 递减 | 递减 |
| 完成回调 | 否 | 是 | 否 |
| 典型场景 | 一次性门控 | 多阶段同步 | 资源控制 |

---

## 12.6 C++20 协程基础

### 原理

C++20 协程是**无栈协程（stackless coroutine）**：

- `co_await`：挂起当前协程，等待异步操作完成
- `co_yield`：挂起并产生一个值给调用者
- `co_return`：协程返回最终值

C++20 只提供了协程的**语言基础设施**（关键字、promise_type、awaiter 协议），未提供标准库的协程类型（如 `Task<T>`、`Generator<T>`）。这些需要自己实现或使用第三方库（如 cppcoro）。

```cpp
#include <coroutine>
#include <iostream>

struct ReturnObject {
    struct promise_type {
        ReturnObject get_return_object() { return {}; }
        std::suspend_never initial_suspend() { return {}; }
        std::suspend_never final_suspend() noexcept { return {}; }
        void unhandled_exception() {}
        void return_void() {}
    };
};

// 简单协程：被 co_await 挂起，恢复后继续
ReturnObject simple_coroutine() {
    std::cout << "Hello ";
    co_await std::suspend_always{};  // 挂起点
    std::cout << "World\n";
    co_return;
}
```

**生活类比**：做饭。你放上水烧（co_await），转头去切菜。水开了（事件触发），回来下面。而不是一直盯着锅——这就是协程的"异步不阻塞"。

### 适用场景

- 异步 I/O（网络、文件）
- 生成器（惰性序列）
- 状态机简化
- 并发任务组合（结构化并发）

### 注意事项

- C++ 协程学习曲线陡峭（需理解 promise_type、awaiter、handle）
- 目前标准库缺少开箱即用的协程类型
- 推荐结合 cppcoro 或 boost::asio 的协程支持使用
- 无栈协程不支持嵌套挂起（与 Go goroutine 不同）

---

## 12.7 知识体系交叉引用

| 本章主题 | 相关章节 | 关系 |
|----------|----------|------|
| jthread | 第2章 线程管理 | 替代 std::thread |
| stop_token | 第4章 同步 | 条件变量可中断等待 |
| semaphore | 第3章 mutex | 更轻量的同步原语 |
| latch/barrier | 第9章 线程池 | 线程池初始化/阶段同步 |
| 协程 | 第4章 async | 更高效的异步模型 |

---

## 12.8 本章小结

C++20 的并发新特性让多线程编程更安全、更简洁：

- `jthread` 消除了 "忘记 join" 的错误
- `stop_token` 提供了标准化的线程取消机制
- `semaphore/latch/barrier` 填补了同步原语的空白
- 协程为异步编程打开了新大门（虽然基础设施仍需完善）

这些特性不是可选的"糖衣语法"，而是解决实际问题的工程利器。建议在所有新项目中优先使用 `jthread` 替代 `std::thread`，用 `latch/barrier` 替代手写的条件变量同步。
