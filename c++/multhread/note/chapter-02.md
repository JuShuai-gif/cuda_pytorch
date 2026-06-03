# 第2章：线程管理

## 章节概述

第1章我们知道了怎么"生"一个线程。这一章教你如何"养"好它——创建、传参、等待、分离、转移所有权。线程不是写完就完事的，管理不好轻则崩溃，重则数据损坏。本章的核心思想是：**用 RAII 管理线程，就像用智能指针管理内存**。

---

## 1. 线程管理基础

### 1.1 创建线程——一切从 `std::thread` 开始

```cpp
#include <thread>

// 方式1：传普通函数
void do_work();
std::thread t1(do_work);

// 方式2：传可调用对象（仿函数）
class Task {
public:
    void operator()() const {
        std::cout << "Working...\n";
    }
};
std::thread t2(Task{});

// 方式3：Lambda（最常用）
std::thread t3([] {
    std::cout << "From lambda\n";
});
```

### 1.2 C++ 的"最棘手的语法解析"（Most Vexing Parse）

```cpp
// ❌ 这会声明一个函数！不是创建线程！
std::thread t(Task());  // t 是一个函数名，返回 std::thread，接受 Task(*)() 参数

// ✅ 正确做法1：用花括号初始化
std::thread t{Task()};

// ✅ 正确做法2：多一层括号
std::thread t((Task()));

// ✅ 正确做法3：Lambda（推荐，彻底避免歧义）
std::thread t([] { Task{}(); });
```

> **生活类比**：这个坑就像中文里的"咬死了猎人的狗"——到底是谁死了？编译器和我们理解的不是一回事。

---

### 1.3 `join()`——等你回来再散伙

```cpp
std::thread t(do_work);
t.join();  // 主线程阻塞，直到 t 执行完毕
// 此时 t 已经完结，不再是 joinable
```

`join()` 等价于：**"我在咖啡店门口等你，你不出来我不走"**。

### 1.4 `detach()`——各走各路，永不相见

```cpp
std::thread t(background_task);
t.detach();  // t 与主线程分离，独立运行
// 主线程不再能控制 t，t 变成"幽灵线程"
```

`detach()` 等价于：**"你先走吧，我自己能找到路"**。但有个致命问题：detach 后线程可能访问主线程中已销毁的局部变量。

```cpp
// ❌ 危险！detach 后局部变量可能已经销毁
void dangerous() {
    int local_var = 42;
    std::thread t([&local_var] {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        std::cout << local_var;  // local_var 可能已经被销毁！
    });
    t.detach();
}  // local_var 在这里销毁，但线程可能还在 sleep
```

> **生活类比**：你给朋友一张你家门禁卡（共享局部变量引用），然后你退租搬走了。两秒后朋友拿着卡来开门——房子已经是别人的了。

---

## 2. RAII 线程管理——用对象生命周期管线程

### 2.1 `thread_guard` 模式

核心思想：**把 `std::thread` 包装成一个 RAII 对象，析构函数自动 `join()`**。

```cpp
class thread_guard {
    std::thread& t;
public:
    explicit thread_guard(std::thread& t_) : t(t_) {}
    ~thread_guard() {
        if (t.joinable()) {
            t.join();  // 自动等待线程结束
        }
    }
    // 禁止拷贝
    thread_guard(const thread_guard&) = delete;
    thread_guard& operator=(const thread_guard&) = delete;
};

void safe_function() {
    std::thread t(do_work);
    thread_guard g(t);  // 即使抛出异常，g 析构时会自动 join
    // ... 可能抛出异常的代码 ...
}
```

**为什么需要这个？** 回顾第1章的坑：如果 `do_work()` 和 `t.join()` 之间有异常抛出，`join()` 永远执行不到，程序 `std::terminate()`。`thread_guard` 保证无论如何都会 `join()`。

> **生活类比**：`thread_guard` 就像自动关灯系统。你离开房间时不管有没有手动关灯，系统都会替你关上。

### 2.2 `joining_thread`——升级版

```cpp
class joining_thread {
    std::thread t;
public:
    joining_thread() noexcept = default;

    template<typename Callable, typename... Args>
    explicit joining_thread(Callable&& f, Args&&... args)
        : t(std::forward<Callable>(f), std::forward<Args>(args)...) {}

    explicit joining_thread(std::thread t_) noexcept
        : t(std::move(t_)) {}

    joining_thread(joining_thread&& other) noexcept
        : t(std::move(other.t)) {}

    joining_thread& operator=(joining_thread&& other) noexcept {
        if (t.joinable()) t.join();
        t = std::move(other.t);
        return *this;
    }

    joining_thread& operator=(std::thread other) noexcept {
        if (t.joinable()) t.join();
        t = std::move(other);
        return *this;
    }

    ~joining_thread() {
        if (t.joinable()) t.join();
    }

    // ... 转发 swap、joinable、get_id 等
};
```

`joining_thread` 直接拥有线程，移动赋值时先 join 旧线程再接手新线程。这是一种 **"你不结束我就不走"** 的语义。

> 注意：C++20 引入的 `std::jthread` 就是标准库版的 `joining_thread`，还额外支持协作式中断。如果你用 C++20，直接用 `std::jthread`。

---

## 3. 向线程函数传递参数

### 3.1 基本规则：参数被"复制"进线程内部存储

```cpp
void f(int i, const std::string& s);

std::thread t(f, 3, "hello");  // "hello" 先被转成 std::string
                                // 然后复制到线程内部存储
```

真正传给 `f` 的 `s` 指向的是线程内部的 `std::string` 副本，而不是外面的 `const char*`。

### 3.2 传引用要用 `std::ref`（最大坑点之一）

```cpp
void update(int& value) { value *= 2; }

int x = 10;
// ❌ 错误！x 被复制了，线程内修改的是副本
std::thread t1(update, x);
t1.join();
// x 仍然等于 10！

// ✅ 正确：用 std::ref 包装引用
std::thread t2(update, std::ref(x));
t2.join();
// x 等于 20！
```

**为什么？** `std::thread` 的构造函数会把所有参数"值复制"一遍（学名：decay-copy），然后以右值形式传给可调用对象。`std::ref` 生成一个 `std::reference_wrapper`，内部存的是指针，所以复制的是指针而不是对象本身。

> **生活类比**：默认传参就像给你画了一张房子的图，不是给你房子的钥匙。`std::ref` 才是给钥匙。

### 3.3 传指针——小心生命周期

```cpp
void process_data(BigData* data);

void caller() {
    BigData local_data;
    std::thread t(process_data, &local_data);
    // ⚠️ 如果 t.detach()，local_data 销毁后指针悬空
    t.detach();
}  // local_data 销毁，detach 的线程还在用指针！
```

**原则**：传指针给 detach 线程时，必须保证指针指向的对象在程序整个生命周期内有效，或使用 shared_ptr。

### 3.4 传成员函数

```cpp
class X {
public:
    void do_work(int n) { std::cout << n; }
};

X my_x;
std::thread t(&X::do_work, &my_x, 42);
//                   ^^^^^^ 对象指针作为第二个参数
t.join();
```

---

## 4. 转移线程所有权

### `std::thread`——只可移动，不可复制

```cpp
std::thread t1(do_work);
std::thread t2 = t1;          // ❌ 编译错误！不可复制
std::thread t3 = std::move(t1); // ✅ 所有权转移给 t3
// 此时 t1 为空，t1.joinable() == false
```

这个设计是有意为之：**每个 `std::thread` 对象最多管理一个线程**。不允许多个对象管理同一线程，避免了"谁该 join"的歧义。

### 实际应用：从函数返回线程

```cpp
std::thread create_worker() {
    return std::thread(do_work);  // 自动 move
}

void launch_job() {
    std::thread t = create_worker();
    t.join();
}
```

### 把线程放入容器

```cpp
std::vector<std::thread> threads;
for (int i = 0; i < 10; ++i) {
    threads.emplace_back([i] { do_work(i); });
}
for (auto& t : threads) {
    t.join();  // 等待所有线程完成
}
```

---

## 5. 运行时决定线程数量

```cpp
unsigned n = std::thread::hardware_concurrency();
std::cout << "硬件支持 " << n << " 个并发线程\n";
```

- 这个函数返回 CPU 核心数（逻辑核心，含超线程），是创建线程数的参考上限。
- 返回值可能为 0（实现无法确定时），需要兜底处理。

### 并行累加的示例

```cpp
template<typename Iterator, typename T>
T parallel_accumulate(Iterator first, Iterator last, T init) {
    unsigned long const length = std::distance(first, last);
    if (!length) return init;

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
    std::vector<T> results(num_threads);

    // 启动 num_threads-1 个子线程
    Iterator block_start = first;
    for (unsigned long i = 0; i < (num_threads - 1); ++i) {
        Iterator block_end = block_start;
        std::advance(block_end, block_size);
        threads[i] = std::thread(
            [](Iterator begin, Iterator end, T& result) {
                result = std::accumulate(begin, end, T{});
            },
            block_start, block_end, std::ref(results[i]));
        block_start = block_end;
    }
    // 最后一个块在主线程计算
    results[num_threads - 1] = std::accumulate(
        block_start, last, T{});

    // join 所有子线程
    for (auto& t : threads) t.join();

    return std::accumulate(results.begin(), results.end(), init);
}
```

---

## 6. 标识线程

```cpp
std::thread t(do_work);
std::thread::id id1 = t.get_id();              // 从 thread 对象获取
std::thread::id id2 = std::this_thread::get_id(); // 当前线程的 id

// 可以比较、输出、用作关联容器 key
if (id1 == std::thread::id{}) {
    // id1 是空的（代表"没有线程"）
}
std::cout << id1 << std::endl;  // 输出类似 "140736073955072"
```

---

## 7. 关键坑点总结

| 坑点 | 解释 |
|------|------|
| `std::thread t(Task())` | 被解析为函数声明，用花括号或 Lambda |
| 忘记 join/detach | 析构时 crash |
| 异常路径跳过 join | 用 `thread_guard` 或 `joining_thread` |
| 传引用忘写 `std::ref` | 修改的是副本，不是你期望的变量 |
| detach 后访问局部变量 | 引用/指针/迭代器悬空 |
| `const char*` 隐式转换 | `std::thread t(f, "hello")` 中转换可能发生在子线程，如果来不及转成 string 就悬空了 |

---

## 8. 工业场景

### 线程池核心
本章的线程创建、所有权转移、容器管理是构建线程池的基础：
```cpp
class SimpleThreadPool {
    std::vector<std::thread> workers;
public:
    void add_task(std::function<void()> f) {
        workers.emplace_back(std::move(f));
    }
    ~SimpleThreadPool() {
        for (auto& t : workers) t.join();
    }
};
// 真正的线程池还要任务队列、工作窃取等（第9章）
```

### 工作线程管理
比如 CUDA 程序中，常见模式是根据 `hardware_concurrency()` 启动 N 个 CPU 线程做数据预处理，每个线程给一个 GPU stream 喂数据。

---

## 9. 面试常问

**Q1: `join()` 和 `detach()` 的区别？**

- `join()`：调用线程阻塞等待被调线程完成，之后被调线程对象不再 joinable。
- `detach()`：被调线程与线程对象分离，独立运行，线程对象不再 joinable。注意：detach 后无法再控制或等待那个线程。

**Q2: `thread_guard` 的原理和作用？**

`thread_guard` 是一个 RAII 包装器，在析构函数中自动调用 `join()`。作用是防止因异常导致 `join()` 被跳过，从而避免 `std::terminate()`。

**Q3: 为什么 `std::thread` 不可复制只能移动？**

每个 `std::thread` 对象代表对一个底层线程的独占所有权。如果允许复制，就会出现两个对象"拥有"同一个线程的情况，析构时两个对象都尝试 join 同一个线程会导致未定义行为。

---

## 10. 推荐练习

1. 实现 `joining_thread` 类，包含移动构造和移动赋值。
2. 写一个程序，用 `std::vector` 管理 10 个线程，每个线程计算一段数组的和，主线程汇总。
3. 重现"传参忘记 `std::ref`"的 bug，确认修改无效，然后用 `std::ref` 修复。
4. 故意 detach 一个访问局部变量的线程，观察随机崩溃。

---

## 掌握清单

- [ ] 能写出正确的 `std::thread` 创建代码（避免 Most Vexing Parse）
- [ ] 理解 `join()` vs `detach()` 的语义和适用场景
- [ ] 能写出 `thread_guard` 类
- [ ] 知道 `std::ref` 的必要性，理解参数拷贝机制
- [ ] 会用 `std::move` 转移线程所有权
- [ ] 能把线程放入 `std::vector` 并正确管理
- [ ] 会用 `std::thread::hardware_concurrency()` 确定线程数
- [ ] 知道 detach 线程访问局部变量的危险性
- [ ] 了解 C++20 的 `std::jthread` 是 joining_thread 的标准版
