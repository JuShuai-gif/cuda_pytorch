# 第5章：C++ 内存模型和原子类型操作

> 本章是 C++ 并发编程的理论基石。理解内存模型，才能真正写出正确的无锁并发代码。

---

## 5.1 内存模型基础

### 5.1.1 对象与内存位置

C++ 标准将内存划分为一个个**内存位置（memory location）**。一个标量类型（如 `int`、`char`、指针）占用一个内存位置，位域中相邻的非零位域也共享一个内存位置。

**核心规则：两个线程可以同时访问不同的内存位置而不互相干扰；但若同时访问同一内存位置，且至少一个是写操作，就可能产生数据竞争（data race）。**

**生活类比**：把内存想象成一栋公寓楼的信箱墙。每个信箱就是一个"内存位置"。两个人同时往不同信箱投递信件，互不影响；但若两人同时伸手去 21 号信箱——一个放信、一个取信——就会撞在一起。这就是数据竞争。

### 5.1.2 修改顺序

对于每个原子对象，C++ 规定该对象的所有修改操作构成了一个**全局一致的修改顺序（modification order）**。这就像是给每次修改打上了一个隐式的"时间戳"，所有线程观察到的修改顺序是一致的。

**生活类比**：Git 的 commit 历史。每次 `git commit` 都会生成一个唯一的 commit hash，所有开发者在 `git pull` 之后看到的提交历史一定是相同的线性序列——不会出现"小张认为 A 在 B 之前，但小李认为 B 在 A 之前"的混乱局面。

### 5.1.3 Happens-Before 关系

**happens-before** 是 C++ 内存模型中最重要的关系之一。它定义了操作之间的可见性保证：

- 单线程内，前面的语句 happens-before 后面的语句（程序顺序）
- 线程创建时，父线程的代码 happens-before 子线程的执行
- 线程 join 时，子线程的执行 happens-before join 之后的代码
- 互斥锁解锁 happens-before 后续加锁
- 原子变量的 store（带 release）happens-before 后续 load（带 acquire 且读到该值）

**生活类比**：餐厅后厨的菜单系统。厨师 A 做完一道菜，把"完成"牌翻过去——这个动作（store-release）标志着菜做好了。服务员过来看到"完成"牌（load-acquire），就知道这道菜可以端走了。翻牌的动作"发生在"端走之前——happens-before。

---

## 5.2 原子操作

### 5.2.1 std::atomic 模板

`std::atomic<T>` 是 C++ 原子操作的入口。可以用任何可平凡拷贝的类型特化它。对于整数和指针类型，还额外提供了 `fetch_add`、`fetch_sub` 等算术操作。

```cpp
std::atomic<int> counter(0);       // 原子整数
std::atomic<bool> flag(false);     // 原子布尔
std::atomic<int*> ptr(nullptr);    // 原子指针
```

### 5.2.2 关键原子操作

| 操作 | 语义 | 使用场景 |
|------|------|----------|
| `store(value, order)` | 写入值 | 发布数据 |
| `load(order)` | 读取值 | 消费数据 |
| `exchange(value, order)` | 交换并返回旧值 | 开关标志、获取所有权 |
| `compare_exchange_strong(expected, desired)` | CAS（比较并交换） | 无锁数据结构的核心 |
| `compare_exchange_weak(expected, desired)` | CAS 弱版本（允许虚假失败） | 循环中使用更高效 |
| `fetch_add(value, order)` | 加法并返回旧值 | 计数器 |

**生活类比**——`compare_exchange` 就像是自动咖啡机的"补货检测"：你说"如果剩余杯子数是 5，就补到 10"（expected=5, desired=10）。机器检查后发现确实剩 5 个，就执行补货，返回 true。如果中途被人拿走一个变成 4，机器就会拒绝操作，并告诉你"不对，实际是 4"（expected 被更新为 4）。

```cpp
// compare_exchange 的典型使用模式
std::atomic<int> value(0);
int expected = 0;
int desired = 1;
// 如果 value == expected，则 value = desired，返回 true
// 否则 expected = value，返回 false
while (!value.compare_exchange_weak(expected, desired)) {
    // CAS 失败，expected 已被更新为当前值
    desired = expected + 1;   // 根据最新值重新计算 desired
}
```

### 5.2.3 ABA 问题

ABA 问题是 CAS 操作的经典陷阱：线程 A 读取值为 A，线程 B 将值改为 B 再改回 A，线程 A 的 CAS 成功——但 A 不知道中间发生过变化。

**生活类比**：你看到停车位是空的，去取车钥匙的 5 秒里，别人的车开走又开回来同一辆车——你回来看到车位上的车好像没变，但实际上发生过变化。对于简单值无所谓，但如果是指向动态内存的指针，原来的地址可能已被释放并重新分配了完全不同的内容。

解决方案：**Tagged Pointer**（标记指针），在指针高位附加一个版本号，每次修改时版本号递增，这样 A→B→A 在版本号上不再是 "A"。

---

## 5.3 内存序（Memory Order）

内存序决定了原子操作之间**除了原子性之外**的可见性和排序约束。这是 C++ 并发模型中最难但最重要的概念。

**生活类比——记录员在小隔间里记笔记**：

想象一个办公室里有多个记录员（CPU 核心），每人都在自己的小隔间（缓存）里记笔记。他们共享一面公告板（主内存），但隔间之间看不到彼此的即时笔记。

- **`memory_order_seq_cst`（顺序一致性）**：所有记录员排队写公告板，写完一个、下一个才写。所有人都看到完全一致的写入顺序。最安全，但也最慢。

- **`memory_order_release` / `memory_order_acquire`（获取-释放序）**：记录员 A 写完一组笔记后，把写好的白板推到走廊（release）。记录员 B 到走廊拿白板（acquire）时，能看到 A 写的所有内容。但 B 自己隔间里的笔记，A 看不到。

- **`memory_order_relaxed`（松散序）**：记录员各自在隔间里随便写，公告板上的内容是乱序的。唯一保证：不会出现半个数字。只保证原子性，不保证顺序。

### 5.3.1 六种内存序详解

| 内存序 | CPU 重排限制 | 典型场景 |
|--------|-------------|----------|
| `memory_order_seq_cst` | 禁止一切重排 | 默认选择，语义最直观 |
| `memory_order_acquire` | 之后的读写不能移到该操作之前 | 读锁/消费者 |
| `memory_order_release` | 之前的读写不能移到该操作之后 | 写锁/生产者 |
| `memory_order_acq_rel` | 同时具有 acquire 和 release 语义 | 单个 RMW 操作中生产+消费 |
| `memory_order_consume` | 依赖该值的操作不能重排到前面 | **不推荐使用，编译器支持弱** |
| `memory_order_relaxed` | 无任何排序约束 | 纯计数器，不关联其他数据 |

### 5.3.2 典型代码示例

**松散序（计数器）**：
```cpp
std::atomic<int> visitors(0);

// 多个线程各自增加计数，无需关心顺序
void record_visit() {
    visitors.fetch_add(1, std::memory_order_relaxed);
}

// 最终打印总数（松散读出的值本身就代表"某个时刻的近似值"）
int get_count() {
    return visitors.load(std::memory_order_relaxed);
}
```

**获取-释放序（生产者-消费者）**：
```cpp
std::atomic<bool> ready(false);
int data = 0;  // 非原子变量

// 生产者
void producer() {
    data = 42;                                     // (1) 准备数据
    ready.store(true, std::memory_order_release);   // (2) 发布"准备好了"
}

// 消费者
void consumer() {
    while (!ready.load(std::memory_order_acquire)); // (3) 等待信号
    assert(data == 42);  // (4) 一定成立！因为 (2) 的 release 与 (3) 的 acquire 配对
}
```

**顺序一致性（自旋锁）**：
```cpp
std::atomic_flag lock = ATOMIC_FLAG_INIT;

void spin_lock_acquire() {
    while (lock.test_and_set(std::memory_order_acquire)) {
        while (lock.test(std::memory_order_relaxed)); // test-and-test-and-set
    }
}

void spin_lock_release() {
    lock.clear(std::memory_order_release);
}
```

### 5.3.3 栅栏（Fence）

`std::atomic_thread_fence` 是一个独立的内存屏障，不与特定原子变量绑定。

```cpp
std::atomic<bool> x(false), y(false);
std::atomic<int> z(0);

// 线程 1
void write_x_then_y() {
    x.store(true, std::memory_order_relaxed);   // (1)
    std::atomic_thread_fence(std::memory_order_release);  // (2) 栅栏
    y.store(true, std::memory_order_relaxed);   // (3)
}

// 线程 2
void read_y_then_x() {
    while (!y.load(std::memory_order_relaxed)); // (4)
    std::atomic_thread_fence(std::memory_order_acquire);  // (5) 栅栏
    if (x.load(std::memory_order_relaxed))      // (6)
        ++z;
}
```

如果线程 2 看到 `y == true`，那么栅栏 (2) 和 (5) 配对，保证线程 2 能看到线程 1 在栅栏之前的所有写入，即 `x == true`。

**生活类比**：栅栏就像是工地上"今日施工范围"的警戒线。在线的一侧，所有工作必须完成；跨过线之后，才能看到另一侧的所有成果。

### 5.3.4 `memory_order_consume` 为什么不被推荐？

它的设计目标是比 acquire 更轻量——仅约束**数据依赖链**上的操作顺序。但编译器几乎从不正确实现它，普遍直接退化为 acquire。C++17 标准已将其标记为"不鼓励使用"。**实际项目请用 acquire。**

---

## 5.4 工业场景

### 无锁计数器
```cpp
class LockFreeCounter {
    std::atomic<uint64_t> count_{0};
public:
    void increment() {
        count_.fetch_add(1, std::memory_order_relaxed);
    }
    uint64_t get() const {
        return count_.load(std::memory_order_relaxed);
    }
};
```
适用场景：QPS 统计、网络包计数、引用计数（需配合 acquire/release）。

### 自旋锁
适用于临界区极短（几十个 CPU 周期）的场景，避免系统调用的开销。

### RCU（Read-Copy-Update）
读者无锁读取，写者复制后修改再原子切换指针。Linux 内核大量使用。C++ 实现需要 `memory_order_consume`（或退化为 acquire）。

### 信号处理（Signal Handler）
信号处理函数中只能使用 `std::atomic_flag` 和 lock-free 的 `std::atomic`。自旋锁不能用于信号处理（会死锁）。

---

## 5.5 常见坑点

1. **把 `memory_order_relaxed` 当作默认选择**：松散序不保证可见性顺序。绝大多数场景应该用默认的 seq_cst 或显式的 release/acquire。

2. **`compare_exchange_weak` 的虚假失败**：弱版本即使 `expected == current` 也可能返回 false。必须在循环中使用。

3. **数据竞争不仅仅是原子变量的问题**：用 release 写原子变量 + acquire 读原子变量，的确能保证非原子数据（在上文中的 `data = 42`）的可见性——但前提是 acquire 必须成功读到 release 写出的值。忘掉这个配对关系是最常见的 bug。

4. **原子性 ≠ 无锁**：`std::atomic` 内部可能使用互斥锁（对大类型）。用 `std::atomic<T>::is_lock_free()` 检查。

5. **ABA 问题被忽视**：任何基于指针 CAS 的无锁结构，都必须考虑 ABA。解决方案：tagged pointer 或 hazard pointer。

---

## 5.6 面试常问

| 问题 | 要点 |
|------|------|
| `memory_order_relaxed` vs `seq_cst` | relaxed 只保原子性，不保顺序；seq_cst 保全局总序 |
| acquire/release 如何配对？ | release-store 与 acquire-load 读到同一值时建立 happens-before |
| 什么是 ABA 问题？ | 值 A→B→A，CAS 误以为没变；用 tagged pointer 解决 |
| `compare_exchange_strong` vs `weak` | strong 不会虚假失败但可能有额外开销；weak 允许虚假失败，适合循环 |
| 什么时候能用 `memory_order_relaxed`？ | 纯计数，不和其他共享数据有任何关联 |

---

## 我应该掌握什么

- [ ] 能解释 happens-before 和 synchronizes-with 的关系
- [ ] 能独立写出正确的 `compare_exchange_weak` 循环
- [ ] 能说出六种 memory_order 各自的含义和适用场景
- [ ] 能写出生产者-消费者的 release/acquire 配对代码
- [ ] 知道 ABA 问题是什么，以及如何用 tagged pointer 解决
- [ ] 能用栅栏（fence）替代变量上的内存序标签
- [ ] 知道 `memory_order_consume` 为什么不应使用
- [ ] 能用原子操作实现一个正确的自旋锁
