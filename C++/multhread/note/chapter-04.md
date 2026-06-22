# 第4章：同步并发操作

## 章节概述

第3章讲的是怎么用锁**防**——防止多线程同时访问共享数据。本章讲的是怎么让线程**配合**——等你干完了通知我、等我拿到结果再往下走、限定时间内完不成就超时。这是从"被动防御"到"主动协作"的升级。

---

## 4.1 条件变量——线程间的"电话通知"

### 4.1.1 问题场景：忙等待的愚蠢

假设线程A要等线程B准备好数据：

```cpp
// ❌ 忙等待（Busy Waiting）：CPU 空转，浪费资源
bool data_ready = false;
// 线程A
while (!data_ready) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
}
// 处理数据...
```

这就像你不停给餐厅打电话问"我的外卖好了没？每秒钟打一次"，店家和你都疯了。

### 4.1.2 条件变量登场

```cpp
#include <mutex>
#include <condition_variable>

std::mutex mtx;
std::condition_variable cv;
bool data_ready = false;
Data shared_data;

// 线程B：生产者
void producer() {
    Data data = prepare_data();
    {
        std::lock_guard lk(mtx);
        shared_data = data;
        data_ready = true;
    }
    cv.notify_one();  // "外卖好了，来取吧"
}

// 线程A：消费者
void consumer() {
    std::unique_lock<std::mutex> lk(mtx);
    cv.wait(lk, [] { return data_ready; });  // "等通知，顺便帮我锁好门"
    process_data(shared_data);
    lk.unlock();
}
```

> **生活类比**：
> - **条件变量** = 外卖 App 的"取餐通知"。你不用刷屏问，饿了躺沙发上等推送。
> - **`wait()`** = 你注册了通知，然后放下手机等。通知到了自然醒，醒来时外卖已经准备好了。
> - **`notify_one()`** = 商家点了一下"出餐"，你的手机震了一下。
> - **`notify_all()`** = 商家群发"所有等这锅的同学都好了"，一堆人同时收到。

### 4.1.3 `wait()` 的工作原理

```cpp
// wait(lk, predicate) 等价于：
while (!predicate()) {
    wait(lk);  // 解锁 mtx，阻塞自己
    // 被 notify 唤醒后，重新锁定 mtx
}
```

关键细节：
1. `wait()` 在被调用时**解锁互斥量**，然后阻塞。
2. 被 `notify` 唤醒后，**重新锁定互斥量**，然后返回。
3. 使用带谓词的 `wait(lk, predicate)` 版本，自动处理虚假唤醒。

### 4.1.4 虚假唤醒（Spurious Wake）

**虚假唤醒**：线程在没有任何通知的情况下被唤醒。这是底层（OS/硬件）的实现细节，不是 bug。

```cpp
// ❌ 有 bug 的写法：只检查一次
cv.wait(lk);  // 可能被虚假唤醒，但 data_ready 还是 false
// 直接处理 data → 吃坏肚子

// ✅ 正确写法：循环检查（或等价地用带谓词的 wait）
while (!data_ready) {
    cv.wait(lk);
}

// ✅ 更简洁（完全等价）：
cv.wait(lk, [] { return data_ready; });
```

**永远使用带谓词的 `wait()` 版本**，标准库帮你做了循环检查。

### 4.1.5 `notify_one()` vs `notify_all()`

| `notify_one()` | `notify_all()` |
|---|---|
| 唤醒**一个**等待线程 | 唤醒**所有**等待线程 |
| 不确定唤醒哪一个 | 所有等待线程依次获取锁 |
| 适用于只需一个线程处理的场景 | 适用于所有等待线程都需要响应的场景 |
| 最常用（写线程通知读线程） | 如全局状态变更需要所有线程知道 |

### 4.1.6 生产者-消费者完整示例

```cpp
#include <queue>
#include <mutex>
#include <condition_variable>

template<typename T>
class ThreadSafeQueue {
    mutable std::mutex mtx;
    std::queue<T> data_queue;
    std::condition_variable cv;

public:
    void push(T new_value) {
        {
            std::lock_guard lk(mtx);
            data_queue.push(std::move(new_value));
        }
        cv.notify_one();  // 通知一个等待的消费者
    }

    // 阻塞等待，直到有数据可取
    T wait_and_pop() {
        std::unique_lock<std::mutex> lk(mtx);
        cv.wait(lk, [this] { return !data_queue.empty(); });
        T value = std::move(data_queue.front());
        data_queue.pop();
        return value;
    }

    // 非阻塞尝试
    bool try_pop(T& value) {
        std::lock_guard lk(mtx);
        if (data_queue.empty()) return false;
        value = std::move(data_queue.front());
        data_queue.pop();
        return true;
    }

    bool empty() const {
        std::lock_guard lk(mtx);
        return data_queue.empty();
    }
};
```

---

## 4.2 Future / Promise——并发的"快递单号"

### 4.2.1 核心概念

条件变量解决的是"等待某个时刻"的问题。但有时你需要的是**"等待某个结果"**。Future/Promise 就是为此而生。

> **生活类比**：
> - **`std::promise`** = 快递单号生成器。快递公司说"这个单号对应一个包裹，我会往里放东西"。
> - **`std::future`** = 快递单号。你拿着单号，随时可以查询"包裹到了没？"。
> - **`promise.set_value(x)`** = 快递员把包裹放进储物柜。
> - **`future.get()`** = 你拿着单号取件。如果包裹没到，你就等着（阻塞）；到了就拿走。

### 4.2.2 基本用法

```cpp
#include <future>

void thread_func(std::promise<int> p) {
    // 模拟耗时计算
    std::this_thread::sleep_for(std::chrono::seconds(1));
    p.set_value(42);  // 放入结果
}

int main() {
    std::promise<int> p;
    std::future<int> f = p.get_future();  // 拿到"单号"

    std::thread t(thread_func, std::move(p));  // promise 只能移动

    std::cout << "Waiting for result...\n";
    int result = f.get();  // 阻塞，直到 set_value(42)
    std::cout << "Got: " << result << "\n";  // Got: 42

    t.join();
}
```

**关键：`get()` 只能调一次**。调用后 future 失效。如果需要多次获取，用 `std::shared_future`。

### 4.2.3 `std::future` vs `std::shared_future`

```cpp
std::promise<std::string> p;
std::future<std::string> f = p.get_future();

// ❌ 错误：future 不可复制
// auto f2 = f;

// ✅ 可以移动
auto f2 = std::move(f);

// ✅ shared_future 可以四处传阅
std::shared_future<std::string> sf = f2.share();
auto sf2 = sf;  // 复制是允许的！
```

| `std::future` | `std::shared_future` |
|---|---|
| 独占式 | 共享式 |
| `get()` 只能调用一次 | `get()` 可以多次调用 |
| 不可复制，只可移动 | 可复制 |
| 单次消费 | 多次读取（如多个线程等同一个结果） |

### 4.2.4 `std::async`——最简单的异步

不需要手动创建线程和 promise：

```cpp
#include <future>

int compute(int x, int y) {
    return x + y;
}

int main() {
    // launch::async：立即在新线程中执行
    auto f1 = std::async(std::launch::async, compute, 10, 20);

    // launch::deferred：延迟执行，只在 get()/wait() 时在当前线程执行
    auto f2 = std::async(std::launch::deferred, compute, 5, 15);

    // 默认：由实现决定（通常是 async|deferred）
    auto f3 = std::async(compute, 3, 7);

    std::cout << f1.get() << "\n";  // 30
    std::cout << f2.get() << "\n";  // 20
}
```

### 4.2.5 `std::packaged_task`——打包任务

把一个可调用对象和一个 future 绑在一起：

```cpp
std::packaged_task<int(int, int)> task(
    [](int a, int b) { return a * b; }
);

std::future<int> f = task.get_future();

// 可以在任何线程中执行
std::thread t(std::move(task), 6, 7);
t.join();

std::cout << f.get() << "\n";  // 42
```

`packaged_task` 的优势：
- 可以把任务放进任务队列里等待执行（线程池核心）。
- `promise` 只暴露了 "set value" 端，`packaged_task` 包装的是整份工作。

### 4.2.6 异常传递

```cpp
std::promise<int> p;
auto f = p.get_future();

try {
    p.set_exception(
        std::make_exception_ptr(std::runtime_error("Boom!"))
    );
} catch (...) {
    // p 的析构也会存储 broken_promise 异常
}

try {
    f.get();  // 这里会重新抛出异常
} catch (const std::exception& e) {
    std::cout << e.what() << "\n";  // Boom!
}
```

如果 `promise` 在 `set_value` 之前被销毁，`future.get()` 会抛出 `std::future_error`（`broken_promise`）。

---

## 4.3 限时等待——给你的等加个期限

### 4.3.1 时间工具

```cpp
#include <chrono>

// Duration：时间段
std::chrono::milliseconds ms(1500);             // 1500ms
std::chrono::seconds sec(3);                     // 3s
auto combined = std::chrono::milliseconds(1500) + std::chrono::seconds(3);

// Time Point：时刻
auto start = std::chrono::steady_clock::now();
// ... 干活 ...
auto end = std::chrono::steady_clock::now();
auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
```

| 时钟类型 | 用途 |
|---------|------|
| `std::chrono::system_clock` | 系统时钟，可转为日历时间，但可能被用户/系统调整 |
| `std::chrono::steady_clock` | 单调时钟，从不回拨，适合测量时间间隔 |
| `std::chrono::high_resolution_clock` | 最高精度时钟（通常是 steady 或 system 的别名） |

### 4.3.2 带超时的 `wait`

```cpp
std::mutex mtx;
std::condition_variable cv;
bool done = false;

void wait_with_timeout() {
    std::unique_lock lk(mtx);

    // wait_for：等最多 500 毫秒
    if (cv.wait_for(lk, std::chrono::milliseconds(500),
                    [] { return done; })) {
        std::cout << "Done!\n";
    } else {
        std::cout << "Timeout!\n";
    }

    // wait_until：等到某个时间点
    auto deadline = std::chrono::steady_clock::now()
                    + std::chrono::seconds(2);
    if (cv.wait_until(lk, deadline, [] { return done; })) {
        std::cout << "Done before deadline!\n";
    }
}
```

### 4.3.3 带超时的 `future`

```cpp
std::future<int> f = std::async(std::launch::async, [] {
    std::this_thread::sleep_for(std::chrono::seconds(5));
    return 42;
});

// 最多等 1 秒
if (f.wait_for(std::chrono::seconds(1)) == std::future_status::ready) {
    std::cout << f.get() << "\n";
} else {
    std::cout << "Too slow!\n";
}
```

`future_status` 三个值：
- `ready`：结果已就绪
- `timeout`：超时了
- `deferred`：使用了 `launch::deferred`，计算还没开始

---

## 4.4 使用同步简化并发代码

### 4.4.1 函数式编程（FP）与并发

函数式编程的核心思想：**纯函数**（Pure Function）——相同的输入永远得到相同的输出，没有副作用（不修改外部状态）。

```cpp
// 有副作用（共享状态，需要锁）
void impure_add(std::vector<int>& v, int x) {
    for (auto& val : v) val += x;
}

// 无副作用（输入不可变，返回新对象）
std::vector<int> pure_add(const std::vector<int>& v, int x) {
    std::vector<int> result;
    result.reserve(v.size());
    for (int val : v) result.push_back(val + x);
    return result;  // 返回全新对象，不影响外部
}
```

**FP 与并发的天然亲近**：纯函数没有共享状态 → 不需要锁 → 没有数据竞争 → 天生线程安全。

写并发代码时，尽可能用**不可变数据** + **纯函数** 的模式，能大幅减少锁的使用。

### 4.4.2 Actor 模型（消息传递）

不让线程共享数据，而是让每个线程只管理自己的数据，通过消息队列通信：

```cpp
// 简化版的 Actor 模式
class Actor {
    std::queue<std::function<void()>> mailbox;
    std::mutex mtx;
    std::condition_variable cv;
    std::thread worker;

    void run() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock lk(mtx);
                cv.wait(lk, [this] { return !mailbox.empty(); });
                task = std::move(mailbox.front());
                mailbox.pop();
            }
            task();
        }
    }

public:
    // 发送消息（把任务放入邮箱）
    void send(std::function<void()> msg) {
        {
            std::lock_guard lk(mtx);
            mailbox.push(std::move(msg));
        }
        cv.notify_one();
    }
};
```

这其实就是**任务队列 + 工作线程**的雏形，也是线程池的基础。

---

## 4.5 常见坑点

### 坑1：虚假唤醒忘记循环检查

```cpp
// ❌ 危险
cv.wait(lk);
// 虚假唤醒后 data 可能还没准备好

// ✅ 带谓词的 wait
cv.wait(lk, [] { return data_ready; });
```

### 坑2：条件变量的"丢失唤醒"

```cpp
// 生产者
void producer() {
    data_ready = true;
    cv.notify_one();  // ← 此时消费者可能还没开始 wait
}
// 消费者
void consumer() {
    std::this_thread::sleep_for(1s);
    cv.wait(lk, []{ return data_ready; });  // ← 已经检查 data_ready=true，不会阻塞
}
```

带谓词的 `wait` 已经自动处理了这种情况（先检查谓词，为 true 就不阻塞）。

### 坑3：`future.get()` 只能调用一次

```cpp
auto f = std::async([]{ return 42; });
int x = f.get();  // OK
int y = f.get();  // ❌ 第二次调用：未定义行为
```

需要多次获取用 `shared_future`。

### 坑4：`std::async` 的默认启动策略

```cpp
auto f = std::async(do_something);  // 默认策略：async|deferred
// 实现可能选择 deferred（延迟执行），那就是在当前线程同步执行
// 如果你的逻辑依赖"一定在新线程中执行"，明确指定 std::launch::async
auto f2 = std::async(std::launch::async, do_something);
```

### 坑5：promise 在 set_value 前被销毁

```cpp
std::future<int> f;
{
    std::promise<int> p;
    f = p.get_future();
}  // p 析构，没 set_value → future_error (broken_promise)

f.get();  // 抛出异常
```

---

## 4.6 工业场景

### 事件循环（Event Loop）

```cpp
class EventLoop {
    std::queue<std::function<void()>> events;
    std::mutex mtx;
    std::condition_variable cv;
    bool stop = false;

public:
    void run() {
        while (true) {
            std::function<void()> event;
            {
                std::unique_lock lk(mtx);
                cv.wait(lk, [this] { return stop || !events.empty(); });
                if (stop && events.empty()) break;
                event = std::move(events.front());
                events.pop();
            }
            event();
        }
    }

    void post(std::function<void()> f) {
        {
            std::lock_guard lk(mtx);
            events.push(std::move(f));
        }
        cv.notify_one();
    }
};
```

### 异步任务调度

用 `std::packaged_task` + 线程池，实现"提交任务→拿 future→等结果"模式：

```cpp
template<typename Func>
auto submit_task(Func&& f)
    -> std::future<decltype(f())> {
    using ResultType = decltype(f());
    auto task = std::make_shared<std::packaged_task<ResultType()>>(
        std::forward<Func>(f));
    auto future = task->get_future();
    // 把 task 放入任务队列
    task_queue.push([task] { (*task)(); });
    return future;
}
```

### 流处理

多个处理阶段通过线程安全队列连接：
```
输入 → [线程1：解析] → 队列A → [线程2：转换] → 队列B → [线程3：输出]
```
每个阶段等待上一阶段的输出（用条件变量），处理完后通知下一阶段。

---

## 4.7 本章小结

| 工具 | 场景 | 类比 |
|------|------|------|
| `condition_variable` | 等待某个条件成立 | 外卖取餐通知 |
| `future` + `promise` | 等待一次性结果 | 快递单号 |
| `async` | 最简单的异步 | 一键下单 |
| `packaged_task` | 打包任务放进队列 | 预制菜 |
| `shared_future` | 多个线程等同一结果 | 多人看同一个快递物流页面 |
| 超时等待 | 给等待加截止时间 | 设置最大等待时间，超时就放弃 |

核心思想：**从"防"到"协"**。第3章教的是锁住共享数据防止同时写，本章教的是通过等待和通知来协调线程执行顺序。

---

## 4.8 面试常问

**Q1: 条件变量的 `wait()` 为什么要传入 mutex？**

`wait()` 需要 mutex 来保护条件（谓词中的共享数据）。在阻塞之前，`wait()` 会对 mutex 解锁（让其他线程有机会修改条件），被唤醒后重新锁定 mutex（让你能安全地读取条件）。如果不用 mutex，条件的检查和线程挂起之间存在窗口期，可能丢失通知。

**Q2: 什么是虚假唤醒？如何处理？**

虚假唤醒是线程在没有收到通知的情况下被唤醒（OS/硬件层面的实现特性）。处理方式：使用带谓词的 `cv.wait(lk, predicate)`，它会在内部循环检查唤醒后条件是否真正满足。

**Q3: `std::async` 的 `launch::async` 和 `launch::deferred` 区别？**

- `launch::async`：保证在新线程中立即执行。
- `launch::deferred`：延迟执行，直到 `get()` 或 `wait()` 被调用时才在当前线程执行。
- 默认（不指定）：由实现选择（通常是 `async | deferred`）。

**Q4: `future` 和 `shared_future` 的区别？**

- `future` 独占式，`get()` 只能调用一次，不可复制，只可移动。
- `shared_future` 共享式，可复制，`get()` 可多次调用，适合多个线程等同一结果。

---

## 4.9 推荐练习

1. 实现一个完整的 `ThreadSafeQueue`，支持阻塞 pop 和非阻塞 try_pop。
2. 用 `condition_variable` 实现交替打印（线程A打印奇数，线程B打印偶数，交替输出 1 2 3 4...）。
3. 实现一个简化版线程池：用 `packaged_task` 提交任务并返回 `future`。
4. 写一个程序，用 `wait_for` 实现带超时的任务等待，模拟"3秒内计算不完就放弃"。
5. 用 `std::async` + `shared_future` 实现：多个线程等待同一个计算结果。

---

## 掌握清单

- [ ] 理解条件变量的核心作用：避免忙等待，实现线程间协作
- [ ] 知道 `wait()` 为什么需要传入 mutex
- [ ] 能解释虚假唤醒，并始终坚持使用带谓词的 `wait()`
- [ ] 能写出完整的生产者-消费者模型
- [ ] 知道 `notify_one()` 和 `notify_all()` 的适用场景
- [ ] 理解 `std::promise` / `std::future` 的配对工作原理
- [ ] 会用 `std::async` 执行异步任务，理解两种启动策略
- [ ] 会用 `std::packaged_task` 打包任务
- [ ] 知道 `shared_future` 的用途和与 `future` 的区别
- [ ] 会用 `wait_for` / `wait_until` 实现超时等待
- [ ] 知道三个时钟类型（system_clock、steady_clock、high_resolution_clock）的用途
- [ ] 理解函数式编程思想如何帮助并发编程
- [ ] 了解 Actor 模型的基本思想
