# 第7章：无锁并发数据结构设计

> 锁就像红绿灯——保证安全，但强制等待。无锁结构像环岛——只要不撞上，你可以不停车直接通过。

---

## 7.1 定义和意义

### 7.1.1 阻塞（Blocking）、无锁（Lock-Free）、无等待（Wait-Free）

| 分类 | 定义 | 类比 |
|------|------|------|
| **阻塞（Blocking）** | 一个线程的延迟/失败会导致其他线程被无限期阻塞 | 有人长时间占着卫生间，外面的人只能干等 |
| **无锁（Lock-Free）** | 保证在任意时刻，至少有一个线程能取得进展 | 多人协作的白板——也许有人要重试，但总有人在推进 |
| **无等待（Wait-Free）** | 每个操作都在有限步骤内完成，不受其他线程影响 | 每人一个独立工作台，互不干扰 |

### 7.1.2 为什么要无锁？

- **避免死锁**：无锁结构天然免疫死锁
- **避免优先级反转**：低优先级线程持锁阻塞高优先级线程——无锁不存在此问题
- **更好的尾部延迟**：信号处理/中断上下文中必须无锁
- **极限性能**：避免内核态切换和缓存失效风暴

**代价**：代码复杂、难调试、内存回收棘手。

---

## 7.2 无锁数据结构实例

### 7.2.1 Treiber Stack（无锁栈）

最经典的无锁结构。核心操作：CAS 竞争修改 `head_` 指针。

```cpp
#include <atomic>
#include <memory>

template<typename T>
class LockFreeStack {
    struct Node {
        T data;
        Node* next;
        Node(T val) : data(std::move(val)), next(nullptr) {}
    };

    std::atomic<Node*> head_{nullptr};

public:
    void push(T value) {
        Node* new_node = new Node(std::move(value));
        new_node->next = head_.load(std::memory_order_relaxed);

        // CAS 循环：不断尝试把 head 指向新节点
        while (!head_.compare_exchange_weak(
                   new_node->next, new_node,
                   std::memory_order_release,
                   std::memory_order_relaxed)) {
            // new_node->next 已被更新为当前 head，继续重试
        }
    }

    std::shared_ptr<T> pop() {
        Node* old_head = head_.load(std::memory_order_relaxed);

        do {
            if (old_head == nullptr) {
                return std::shared_ptr<T>();  // 栈空
            }
        } while (!head_.compare_exchange_weak(
                     old_head, old_head->next,
                     std::memory_order_acquire,
                     std::memory_order_relaxed));

        // CAS 成功，old_head 已被取下
        std::shared_ptr<T> res = std::make_shared<T>(
            std::move(old_head->data));
        delete old_head;
        return res;
    }
};
```

**问题**：`delete old_head` 是危险的——可能另一个线程仍在使用该节点的指针。这就引入了**内存回收问题**。

### 7.2.2 Michael-Scott Queue（无锁队列）

```cpp
#include <atomic>
#include <memory>

template<typename T>
class LockFreeQueue {
    struct Node {
        std::shared_ptr<T> data;
        std::atomic<Node*> next;
        Node() : next(nullptr) {}
    };

    std::atomic<Node*> head_;
    std::atomic<Node*> tail_;

public:
    LockFreeQueue() {
        Node* dummy = new Node();
        head_.store(dummy);
        tail_.store(dummy);
    }

    void push(T value) {
        auto new_data = std::make_shared<T>(std::move(value));
        Node* new_node = new Node();
        // 注意：data 必须在新节点添加到队列之前赋值
        // 否则 consumer 可能看到未初始化的 data
        new_node->data = new_data;

        Node* old_tail;
        Node* null_next = nullptr;

        while (true) {
            old_tail = tail_.load(std::memory_order_relaxed);
            Node* next = old_tail->next.load(std::memory_order_acquire);

            // 检查 tail 是否过期（别人已经插入了但没更新 tail）
            if (old_tail == tail_.load(std::memory_order_relaxed)) {
                if (next == nullptr) {
                    // 尝试链接新节点
                    if (old_tail->next.compare_exchange_weak(
                            null_next, new_node,
                            std::memory_order_release,
                            std::memory_order_relaxed)) {
                        break;  // 链接成功
                    }
                } else {
                    // 帮助落后的 tail 前进
                    tail_.compare_exchange_weak(
                        old_tail, next,
                        std::memory_order_release,
                        std::memory_order_relaxed);
                }
            }
        }

        // 更新 tail（可能失败——无所谓，别人会帮我们更新）
        tail_.compare_exchange_weak(
            old_tail, new_node,
            std::memory_order_release,
            std::memory_order_relaxed);
    }

    std::shared_ptr<T> pop() {
        Node* old_head;
        Node* next;

        while (true) {
            old_head = head_.load(std::memory_order_relaxed);
            next = old_head->next.load(std::memory_order_acquire);

            if (old_head == head_.load(std::memory_order_relaxed)) {
                if (next == nullptr) {
                    return std::shared_ptr<T>();  // 队列空
                }
                if (head_.compare_exchange_weak(
                        old_head, next,
                        std::memory_order_release,
                        std::memory_order_relaxed)) {
                    break;
                }
            }
        }

        std::shared_ptr<T> res = next->data;
        // 危险：delete old_head 可能被其他线程同时访问
        delete old_head;
        return res;
    }
};
```

**生活类比——无锁就像多人在白板上协作**：

传统方式（加锁）：大家排队等一支笔。无锁方式：每个人拿不同颜色的笔，直接在白板上写。如果有人发现他打算写的区域已被别人改了，就擦掉重写（CAS 失败重试）。虽然有人可能要重写几次，但整体没人空等——总有人的笔在动。

### 7.2.3 助处理（Helping）模式

注意到上面 push 代码中，如果发现 `tail` 指针落后了（`next != nullptr`），当前线程会**帮助**其他线程完成 tail 更新。这就是 helping 模式——"顺手帮个忙"，让整体进展更快。

**生活类比**：快递员搬货上货车。如果你发现上一个人放的包裹挡住了路，你不会干等着——你会顺手把它推到位，然后再放自己的包裹。

---

## 7.3 设计指导

### 7.3.1 内存回收问题

无锁结构最大的难点不是 CAS 循环，而是**安全地释放节点**。

**危险场景**：
```
线程 A：从 pop 中取出 old_head，准备 delete
线程 B：也在 pop 中，正持有 old_head 的指针，准备读取 old_head->next
→ A delete 后 B 访问悬垂指针 → 未定义行为
```

三大解决方案：

| 方案 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| **Hazard Pointers** | 线程公开声明正在使用哪些指针，删除前检查 | 较简单 | 每个线程需要记录正在读的指针 |
| **Reference Counting** | 节点带原子引用计数 | 直观 | 引用计数的 CAS 本身也有竞争 |
| **Epoch-Based Reclamation** | 分"纪元"回收内存，活动线程都不在旧纪元时安全释放 | 高性能 | 内存占用可能较高 |

#### Hazard Pointer 简化版

```cpp
// 每个线程维护一个 hazard pointer 集合
std::atomic<void*> hazard_pointers[MAX_THREADS][MAX_HAZARDS];

// 读取时：将自己的 HP 指向正在读的节点
void set_hazard(int thread_id, int hp_index, void* ptr) {
    hazard_pointers[thread_id][hp_index].store(ptr);
}

// 释放时：检查所有线程的 HP，确认没人持有该节点后再释放
void safe_delete(void* ptr) {
    // 将 ptr 加入待删除列表
    // 定期检查所有线程的 hazard pointers
    // 如果没有任何 HP 指向 ptr，则真正 delete
}
```

### 7.3.2 ABA 问题与 Tagged Pointer

ABA 是 CAS 在指针场景下的经典陷阱（详见第 5 章）。无锁栈中最容易触发：

```
1. 栈：A → B → C
2. 线程 1 读到 head = A，准备 CAS(head, A→B)
3. 线程 2 pop A, pop B, push A (ABA!)
4. 线程 1 的 CAS 成功，但 B 已被释放/重用，next 是垃圾数据
```

**解决方案**：在指针的高位附加一个计数器（tagged pointer）。

```cpp
// x86-64 下指针只使用低 48 位，高 16 位可用于 tag
template<typename T>
struct TaggedPointer {
    T* ptr;
    uint16_t tag;
};

// 或使用双字 CAS（但 std::atomic 不支持 128-bit 操作直接封装）
// 实际项目中常用：boost::lockfree::stack / folly::AtomicLinkedList
```

### 7.3.3 伪共享（False Sharing）

无锁结构中线程频繁修改相邻的内存位置，可能导致缓存行在 CPU 间乒乓。

```cpp
// 错误：counter1 和 counter2 可能在同一缓存行
std::atomic<int> counter1(0);
std::atomic<int> counter2(0);

// 正确：强制对齐到不同缓存行（通常 64 字节）
alignas(64) std::atomic<int> counter1(0);
alignas(64) std::atomic<int> counter2(0);
```

### 7.3.4 忙等与让步

当 CAS 循环长时间失败时：
- `std::this_thread::yield()`：让出 CPU 给其他线程
- `_mm_pause()`（x86）：提示 CPU 这是自旋等待，降低功耗

```cpp
for (int i = 0; !head_.compare_exchange_weak(old, new_node); ++i) {
    if (i > MAX_SPIN_COUNT) {
        std::this_thread::yield();
        i = 0;
    }
}
```

---

## 7.4 工业场景

### 高性能日志系统
多生产者往无锁环形缓冲区写日志，单消费者异步刷盘。吞吐可达百万 QPS。

### 网络 I/O
DPDK、Seastar 等框架大量使用无锁队列在用户态做数据包分发。

### 实时系统 / 游戏引擎
不允许线程被阻塞——帧循环中任何一次锁等待都可能导致掉帧。

### 内存分配器
`jemalloc` 和 `tcmalloc` 使用无锁技术管理线程本地缓存。

---

## 7.5 常见坑点

1. **ABA 问题被忽视**：任何基于 CAS 指针的无锁结构第一反应就该问"有没有 ABA 风险"。

2. **忘记内存回收**：`delete` 在无锁上下文中几乎总是错的。必须有安全回收机制。

3. **伪共享毁灭性能**：多线程各自写"私有的"原子变量，但因为挤在同一缓存行上，性能反而比单线程还差。

4. **测试不足**：无锁代码的 bug 可能几个月才触发一次。需要压力测试、ThreadSanitizer、Relacy Race Detector 等工具。

5. **过早优化**：绝大多数场景下，一把 `std::mutex` 就足够好了。**只在性能分析显示锁是瓶颈时才考虑无锁方案**。

---

## 7.6 面试常问

| 问题 | 要点 |
|------|------|
| lock-free 和 wait-free 的区别？ | lock-free：总有线程能推进；wait-free：每个操作都在有限步内完成 |
| 什么是 ABA 问题？如何解决？ | CAS 误以为值没变。用 tagged pointer 附加版本号 |
| 无锁栈的核心实现？ | Treiber Stack：CAS head 指针，注意内存回收 |
| 无锁队列的核心实现？ | Michael-Scott Queue：head/tail + CAS，注意 helping |
| 内存回收有哪些方案？ | Hazard Pointers、引用计数、Epoch-Based Reclamation |
| 退避策略有哪些？ | PAUSE → Yield → Exponential → Randomized，逐步升级 |
| 环形缓冲区如何实现无锁？ | head/tail 分属消费者/生产者，各自用 CAS 保护 |
| Epoch Reclamation 原理？ | 线程注册当前 epoch，所有线程离开后批量回收 |
| 为什么无锁代码中不能直接用 delete？ | 可能有其他线程同时在读，导致 use-after-free |
| 伪共享是什么？如何解决？ | 不同线程写同一缓存行；用 alignas(64) 隔离 |

---

## 我应该掌握什么

- [ ] 能区分 blocking、lock-free、wait-free 三个概念
- [ ] 能手写 Treiber Stack 的 push/pop（含 CAS 循环）
- [ ] 能解释 Michael-Scott Queue 为什么要 helping
- [ ] 知道 ABA 问题在无锁结构中的具体表现形式
- [ ] 至少了解一种内存回收方案（hazard pointer 或 epoch-based）
- [ ] 知道伪共享是什么以及如何用缓存行对齐避免
- [ ] 能评估一个场景是否真的需要无锁方案
- [ ] 知道 ThreadSanitizer 等工具可以检测无锁代码的 bug

---

## 7.7 退避策略 (Backoff)

### 原理

CAS 循环失败时立即重试会导致大量总线争用和缓存无效化。退避策略通过在失败后引入不同长度的等待来减少争用。

### 五种策略对比

| 策略 | 延迟 | 适用场景 |
|------|------|----------|
| No Backoff | 0 | 极低竞争、单次 CAS |
| PAUSE | ~140 cycles | 通用，降低功耗 |
| Yield | 上下文切换 | 高竞争，系统重负载 |
| Exponential | 递增延迟 | 自适应、通用最佳 |
| Randomized | 随机延迟 | 避免同步重试风暴 |

### 选择指南

- 低竞争 → PAUSE
- 中竞争 → Exponential Backoff
- 高竞争 → Exponential + PAUSE 组合
- 不确定 → Randomized (最稳定)

---

## 7.8 Epoch-Based Reclamation

### 原理

将时间划分为"代"（Epoch），删除对象放入当前 epoch 的回收列表。当所有线程都离开某个 epoch 后，该 epoch 的回收列表可安全释放。

### 三步流程

1. **注册**：读线程进入临界区前记录当前 epoch
2. **退休**：删除时把对象放入当前 epoch 的 retire list
3. **回收**：检查所有线程是否已离开目标 epoch → 安全 delete

### Epoch vs Hazard Pointer

| | Epoch | Hazard Pointer |
|---|---|---|
| 读开销 | 极低 (一次 store) | 中等 (多次 store) |
| 回收延迟 | 批量 | 近即时 |
| 实现复杂度 | 中 | 中 |
| 内存积压 | 可能有 | 较少 |

### 适合场景

- 读多写少的无锁结构
- 读路径极度敏感（不肯负担 hazard pointer 的开销）
- 可接受批量回收延迟
