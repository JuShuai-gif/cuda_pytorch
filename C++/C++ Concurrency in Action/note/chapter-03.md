# 第3章：线程间共享数据

## 章节概述

前两章我们已经会"生"线程和"养"线程了。现在面对真正的难题：多个线程想访问同一份数据时怎么办？你不希望一个线程在写数据的时候另一个线程正在读一半。本章的核心是**互斥量（Mutex）**——并发的"交通信号灯"。

---

## 3.1 共享数据的问题

### 问题一：数据竞争（Data Race）

```cpp
int counter = 0;

void increment() {
    counter++;  // 翻译成汇编：读→加1→写回，不是原子操作！
}

// 启动 10000 个线程各执行一次 increment
// 最终 counter 几乎一定不是 10000
```

**为什么 `++` 不是原子的？** 翻译成机器指令后：

```
mov eax, [counter]    // 1. 从内存读到寄存器
add eax, 1            // 2. 在寄存器里加1
mov [counter], eax    // 3. 写回内存
```

三个步骤之间另一个线程可能横插一脚：

```
时间 →
线程A: 读counter(0) ─→ 加1 ─→ 写回(1)
线程B:      读counter(0) ─→ 加1 ─→ 写回(1)  // 也读到0！
// 两次加1操作，counter 只从0变成了1，丢了一次更新！
```

> **生活类比**：就像你跟室友同时往 Excel 表格里填数字。你读到的余额是100，室友也读到100。你加50写成150，室友加30写成130。最终账面是130，但实际应该180。你丢了室友的50——这就是没锁的结果。

### 问题二：不变量破坏

**不变量**（Invariant）是一种"必须始终成立"的准则。比如双向链表中，A 的 next 指向 B 时，B 的 prev 必须指向 A。

```cpp
// 从双向链表中删除一个节点
void remove(Node* n) {
    n->prev->next = n->next;   // 步骤1
    n->next->prev = n->prev;   // 步骤2
}
```

如果线程A执行完步骤1后被线程B中断，此时链表的不变量被破坏（A 的 next 指向了 B，但 B 的 prev 还没更新）。这就是数据竞争导致的不变量破坏。

---

## 3.2 互斥量（Mutex）——并发的红绿灯

### 3.2.1 `std::mutex` + `std::lock_guard`

```cpp
#include <mutex>

std::mutex mtx;
int counter = 0;

void safe_increment() {
    std::lock_guard<std::mutex> guard(mtx);
    counter++;  // 现在安全了
}
// guard 在作用域结束时自动解锁
```

> **生活类比**：
> - `std::mutex` 就像**厕所的门锁**——一次只能一个人进去。
> - `std::lock_guard` 就像**自动冲水+开门系统**——你出去时自动帮你锁好（解锁），不需要手动操作。
> - 如果不用 `lock_guard`，手动 `lock()`/`unlock()`，就像手动锁门开门——忘了一次就出事故。

### 3.2.2 `std::unique_lock`——更灵活的锁

```cpp
std::mutex mtx;

void flexible_lock() {
    std::unique_lock<std::mutex> lk(mtx);
    // ... 持有锁的代码 ...

    lk.unlock();  // 可以提前解锁！lock_guard 做不到
    // ... 不需要锁的代码 ...

    lk.lock();    // 可以重新上锁
    // ... 再次持有锁的代码 ...

}  // 析构时如果还锁着，自动解锁
```

`unique_lock` 还支持：
- `std::defer_lock`：构造时不自动 lock
- `std::try_to_lock`：尝试 lock，失败也不阻塞
- `std::adopt_lock`：假设 mutex 已经被当前线程 lock

| 特性 | `lock_guard` | `unique_lock` |
|------|-------------|---------------|
| 自动 lock/unlock | ✅ | ✅ |
| 手动 unlock | ❌ | ✅ |
| 延迟 lock | ❌ | ✅ (`std::defer_lock`) |
| 可移动 | ❌ | ✅ |
| 开销 | 最小 | 稍大（多一个 bool 标记） |
| 使用建议 | 简单场景首选 | 需要灵活控制时使用 |

### 3.2.3 `std::lock()`——同时锁住多个互斥量

**死锁场景**：两个线程都需要锁两把锁，但获取顺序不同。

```cpp
std::mutex m1, m2;

// 线程A
void threadA() {
    std::lock_guard<std::mutex> lk1(m1);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    std::lock_guard<std::mutex> lk2(m2);  // 等 m2...但 m2 被线程B锁住了
}

// 线程B
void threadB() {
    std::lock_guard<std::mutex> lk1(m2);   // 先锁 m2
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    std::lock_guard<std::mutex> lk2(m1);   // 等 m1...但 m1 被线程A锁住了
}
// 互相等待 → 死锁！
```

**解决方案**：用 `std::lock()` 原子地锁住多个互斥量。

```cpp
void safe_multi_lock() {
    std::unique_lock<std::mutex> lk1(m1, std::defer_lock);
    std::unique_lock<std::mutex> lk2(m2, std::defer_lock);
    std::lock(lk1, lk2);  // 原子地锁住两个，内部避免死锁
    // 需要配合 unique_lock + defer_lock 使用
}
```

**C++17 的 scoped_lock**（更简洁）：

```cpp
void safe_multi_lock_cpp17() {
    std::scoped_lock guard(m1, m2);  // 等价于上面的代码，更简洁
}
```

---

## 3.3 死锁——并发的幽灵

### 死锁的四种必要条件

1. **互斥**：资源不能共享，每次只能一个线程使用
2. **占有并等待**：持有资源的同时等待其他资源
3. **不可抢占**：资源不能被强制释放
4. **循环等待**：A 等 B，B 等 C，C 等 A，形成环

**打破任意一个条件就能避免死锁**。实践中我们主要打破第2和第4条。

### 避免死锁的策略

#### 策略1：固定锁顺序（Lock Ordering）

所有线程都按相同顺序获取锁：

```cpp
// 规定：永远先锁 m1 再锁 m2
void threadA() {
    std::lock_guard lk1(m1);
    std::lock_guard lk2(m2);
}
void threadB() {
    std::lock_guard lk1(m1);  // 也是先 m1！
    std::lock_guard lk2(m2);
}
```

但大型项目中难以保证。`std::lock()` 和 `std::scoped_lock` 帮你自动处理。

#### 策略2：层次锁（Hierarchical Mutex）

给每个 mutex 分配一个层级编号，规定只能从高层级向低层级获取锁：

```cpp
hierarchical_mutex high_level(10000);
hierarchical_mutex low_level(5000);
hierarchical_mutex other_mutex(6000);

void safe_function() {
    std::lock_guard lk1(high_level);  // 层级 10000
    std::lock_guard lk2(low_level);   // 层级 5000，允许（10000 > 5000）
}

void bad_function() {
    std::lock_guard lk1(low_level);   // 层级 5000
    std::lock_guard lk2(high_level);  // 层级 10000，违规！抛出异常
}
```

本质是在运行时检测锁顺序违规，将死锁风险提前暴露。

#### 策略3：避免嵌套锁

持有锁的时候不要再获取别的锁。如果必须，用 `std::lock` 同时获取。

#### 策略4：避免在持有锁时调用用户提供的代码

你写的库函数持锁时回调了用户的函数，用户的函数可能再去拿同一把锁 → 死锁。这种错误最难排查。

---

## 3.4 保护共享数据的其他方式

### 3.4.1 只初始化一次——`std::call_once`

```cpp
std::unique_ptr<Resource> resource;
std::once_flag resource_flag;

Resource& get_resource() {
    std::call_once(resource_flag, [] {
        resource.reset(new Resource{});
    });
    return *resource;
}
```

保证 `Resource` 只被初始化一次，且线程安全。等价于 C++11 之前用"双检查锁"（Double-Checked Locking）实现的效果，但 `std::call_once` 是标准库保证的、无 bug 的实现。

> **C++11 的静态局部变量初始化本身就是线程安全的**！所以更简单的写法是：
> ```cpp
> Resource& get_resource() {
>     static Resource resource;  // C++11 保证线程安全
>     return resource;
> }
> ```

### 3.4.2 读写锁——`std::shared_mutex`（C++17）

读多写少的场景，用共享锁提升并发度：

```cpp
#include <shared_mutex>

class ThreadSafeCache {
    mutable std::shared_mutex mtx;
    std::map<int, std::string> cache;

public:
    // 读操作：共享锁，多个读线程可以同时持有
    std::string get(int key) const {
        std::shared_lock lk(mtx);  // 共享锁，不互斥
        return cache.at(key);
    }

    // 写操作：独占锁，写的时候没人能读
    void put(int key, std::string value) {
        std::lock_guard lk(mtx);   // 独占锁
        cache[key] = std::move(value);
    }
};
```

> **生活类比**：图书馆的阅览室。多个读者可以同时在里面看书（共享锁），但管理员进去整理书架时，所有人都得出来，只能他一个人在里面（独占锁）。

### 3.4.3 递归锁——`std::recursive_mutex`

同一个线程可以多次 lock 同一把 `recursive_mutex`：

```cpp
std::recursive_mutex rmtx;

void recursive_func(int n) {
    std::lock_guard lk(rmtx);
    if (n > 0) {
        recursive_func(n - 1);  // 再次 lock 同一把锁，OK
    }
}
```

> **警告**：递归锁大多时候是设计不良的信号。使用递归锁通常意味着你的锁粒度有问题，或者职责划分不清。优先考虑重构。

---

## 3.5 常见坑点

### 坑1：接口中的竞争条件——`stack` 的 `top()` + `pop()`

```cpp
// ❌ 不安全的栈接口
template<typename T>
class UnsafeStack {
    std::stack<T> data;
    std::mutex mtx;

public:
    bool empty() const {
        std::lock_guard lk(mtx);
        return data.empty();
    }

    T top() {
        std::lock_guard lk(mtx);
        return data.top();
    }

    void pop() {
        std::lock_guard lk(mtx);
        data.pop();
    }
};

// 使用侧存在竞争条件：
// if (!stk.empty()) {             // ← 此时非空
//     auto val = stk.top();       // ← 另一个线程可能在这之间 pop
//     stk.pop();
// }
```

**解决方案**：合并 `top` 和 `pop` 为一个原子操作：

```cpp
std::shared_ptr<T> pop() {
    std::lock_guard lk(mtx);
    if (data.empty()) throw std::runtime_error("empty stack");
    auto res = std::make_shared<T>(data.top());
    data.pop();
    return res;
}
```

### 坑2：向锁外传递被保护数据的引用/指针

```cpp
class BadDesign {
    std::mutex mtx;
    std::vector<int> data;
public:
    const std::vector<int>& get_data() const {
        std::lock_guard lk(mtx);
        return data;  // ❌ 返回了被保护数据的引用！
    }  // 锁在这里释放，但引用已经传出去了
};

// 调用者拿着 data 的引用，但锁已经释放了
// 其他线程可能同时修改 data → 数据竞争 + 迭代器失效
```

**原则**：**永远不要将锁保护下的数据的指针或引用传递到锁的作用域之外**。

### 坑3：锁粒度过大或过小

- **锁太大**：持有锁的时间太长，降低并发度，甚至变成串行。
- **锁太小**：频繁加锁解锁，开销大，且容易出现逻辑漏洞。

---

## 3.6 工业场景

### 共享缓存

```cpp
class SharedCache {
    mutable std::shared_mutex mtx;
    std::unordered_map<std::string, std::string> cache;
public:
    std::string get(const std::string& key) const {
        std::shared_lock lk(mtx);  // 读共享
        return cache.at(key);
    }
    void put(const std::string& key, std::string val) {
        std::lock_guard lk(mtx);   // 写独占
        cache[key] = std::move(val);
    }
};
```

### 数据库连接池

连接池的"取连接"和"还连接"都需要互斥保护，且常用 `unique_lock` 配合条件变量（第4章讲）实现"等待可用连接"。

### 配置热加载

后台线程定期检查配置文件，主线程读取配置。用 `std::shared_mutex` 实现"写少读多"的并发访问。

---

## 3.7 本章小结

- 共享数据 + 多线程 = 数据竞争 + 不变量破坏，必须用同步机制保护。
- `std::mutex` + `std::lock_guard` 是基本武器，`std::unique_lock` 提供更灵活的控制。
- `std::lock()` / `std::scoped_lock` 解决多锁死锁问题。
- 死锁的四种条件，实践中主要靠**固定锁顺序**和**同时锁**来避免。
- 接口设计要把竞争条件挡在外面（如 `top+pop` 合并）。
- **绝对不要**把被保护数据的指针/引用传到锁外。
- 读多写少用 `std::shared_mutex`。
- 初始化只需一次用 `std::call_once` 或 C++11 静态局部变量。

---

## 3.8 面试常问

**Q1: 什么是死锁？如何避免？**

死锁是多个线程互相等待对方持有的锁，形成循环依赖，谁也动弹不了。
避免方法：
1. 固定锁顺序（所有线程按相同顺序获取锁）
2. 用 `std::lock()` 同时获取多把锁
3. 使用 `std::scoped_lock`（C++17）
4. 层级锁在运行时检测违规

**Q2: `lock_guard` 和 `unique_lock` 的区别？**

| `lock_guard` | `unique_lock` |
|---|---|
| 构造时必定 lock，析构必定 unlock | 支持 `defer_lock`、`try_to_lock`、`adopt_lock` |
| 不可手动 unlock | 可随时 unlock 和 relock |
| 不可移动 | 可移动 |
| 零额外开销 | 有一个 bool 标记是否持有锁 |

**Q3: `std::shared_mutex` 的适用场景？**

适合**读多写少**场景。允许多个读者同时持有共享锁，写者持有独占锁。典型应用：缓存、配置管理、只读数据结构。

**Q4: 为什么要用 `std::call_once`？**

保证在多线程环境下某个初始化只执行一次。它是 C++ 标准保证正确性的"只执行一次"机制。C++11 之后，局部静态变量的初始化也线程安全，常用于单例模式。

---

## 3.9 推荐练习

1. 写一个线程安全的计数器类，提供 `increment()` 和 `get()` 方法。
2. 复现死锁场景（两把锁，两个线程，不同顺序），然后用 `std::lock` 修复。
3. 实现层次锁 `hierarchical_mutex`。
4. 用 `std::shared_mutex` 实现一个简单的读写缓存。
5. 重写 `std::stack`，把 `top()` 和 `pop()` 合并为单个线程安全操作。

---

## 掌握清单

- [ ] 能解释数据竞争的本质（`++` 为什么不是原子的）
- [ ] 会用 `std::mutex` + `std::lock_guard` 保护临界区
- [ ] 理解 `std::unique_lock` 的三种 `defer_lock`/`try_to_lock`/`adopt_lock`
- [ ] 知道死锁的四个必要条件
- [ ] 能用 `std::lock()` 或 `std::scoped_lock` 同时锁住多个互斥量
- [ ] 理解层次锁的原理
- [ ] 能设计线程安全的类（不泄露被保护数据的引用）
- [ ] 会用 `std::shared_mutex` 实现读写锁
- [ ] 会用 `std::call_once` 实现线程安全的一次性初始化
- [ ] 知道为什么递归锁通常是设计不良的信号

---

## 3.10 recursive_mutex 递归锁

### 原理

`std::recursive_mutex` 允许同一线程多次锁定同一个 mutex。内部维护一个**锁计数**：

- `lock()` 计数 +1（若当前线程已持有）
- `unlock()` 计数 -1
- 计数归零时释放锁给其他线程

### 典型场景

```cpp
class Widget {
    std::recursive_mutex mtx_;
public:
    void update() {
        std::lock_guard lock(mtx_);
        // ...
        refresh(); // 内部方法也需要锁
    }
    void refresh() {
        std::lock_guard lock(mtx_); // 同一线程再次加锁 OK
        // ...
    }
};
```

### 为什么不推荐

| 问题 | 说明 |
|------|------|
| 隐藏设计缺陷 | 通常意味着锁粒度太大或 API 设计有问题 |
| 计数不可见 | 不清楚当前持锁次数，增加心智负担 |
| 性能开销 | 比 `std::mutex` 略慢，需要维护计数器 |
| cv 不兼容 | 不能与 `condition_variable` 一起使用 |

### 替代方案

提取**不加锁的内部实现**函数：

```cpp
void update_impl(); // 不加锁
void update() { std::lock_guard lk(mtx_); update_impl(); }
```

---

## 3.11 timed_mutex 超时锁

### 原理

`std::timed_mutex` 提供带超时的锁获取：

- `try_lock_for(duration)` — 等待指定时长
- `try_lock_until(time_point)` — 等到指定时间点

### 核心价值

避免**无限阻塞**，实现优雅降级：

```cpp
std::timed_mutex tmtx;
if (tmtx.try_lock_for(100ms)) {
    // 在 100ms 内获得锁
    do_work();
    tmtx.unlock();
} else {
    // 超时: 走 fallback 逻辑
    handle_timeout();
}
```

### Mutex 类型速查

| 类型 | 超时 | 递归 | 共享 |
|------|:----:|:----:|:----:|
| `mutex` | | | |
| `recursive_mutex` | | ✓ | |
| `timed_mutex` | ✓ | | |
| `recursive_timed_mutex` | ✓ | ✓ | |
| `shared_mutex` | | | ✓ |
| `shared_timed_mutex` | ✓ | | ✓ |
