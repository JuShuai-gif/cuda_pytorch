# CS149 第 16 讲：细粒度同步与无锁编程

**PDF**：Lecture 16 - Implementing Locks, Fine-Grained Synchronization, and Lock-Free Programming

**课程**：Stanford CS149，2025 年秋季

---

## 本讲核心问题

1. 锁应该怎么实现，为什么不同锁的扩展性差别这么大？
2. 细粒度锁什么时候值得用，什么时候会得不偿失？
3. 什么叫 lock-free，它和“没有锁”有什么区别？
4. ABA 问题、无锁队列、无锁栈为什么难？

---

## 1. 并发程序中的三种典型失败形态

### 1.1 死锁（deadlock）

系统中还有待完成工作，但线程互相等待，谁也无法继续。

经典四条件：

- 互斥
- 占有且等待
- 不可抢占
- 循环等待

### 1.2 活锁（livelock）

线程没有真正阻塞，却不断重试、不断失败，系统忙个不停但没有有效进展。

### 1.3 饥饿（starvation）

系统整体在前进，但某些线程长期得不到资源或机会。

### 1.4 为什么要区分

- 死锁是“彻底卡死”
- 活锁是“白忙活”
- 饥饿是“整体前进但个体不公平”

后面的锁设计与无锁算法都在不同角度避免这些问题。

---

## 2. 锁实现：不是一个 `mutex` 名字就结束了

### 2.0.1 TAS 锁的汇编级实现

```asm
lock:
    ts R0, mem[addr]    ; 测试并置 1
    bnz R0, lock         ; 若已锁则自旋
    st mem[addr], #0     ; 释放：写 0
```

`ts` 指令语义：原子地读取内存值到 R0，同时将内存设置为 1。

### 2.0.2 x86 `cmpxchg` 指令详细规范

```asm
lock cmpxchg dst, src
; 比较 EAX 与 dst
; 若相等：ZF=1, dst ← src
; 若不相等：ZF=0, EAX ← dst
```

`lock` 前缀指定操作的原子性。

### 2.0.3 TAS 锁的缓存一致性流量分析

多处理器环境下 TAS 锁的完整一致性流量（持锁期间与释放后的争夺）：
- P1 持锁 → P2/P3 不断发出 BusRdX → 失效行 → 失败重试
- P1 释放 → 发出 BusRdX 写入 0 → 下一轮竞争开始
- 每次 test-and-set 都涉及**写入**——即使只为了判断能不能拿锁

### 2.0.4 锁的理想性能特征五维度

| 维度 | TAS 锁评分 |
|---|---|
| 低延迟 | ★（低争用下好） |
| 低互连流量 | ★（差） |
| 可扩展性 | ★（差） |
| 低存储成本 | ★★★★★（仅一个 int） |
| 公平性 | ★（无公平保证） |

### 2.1 Test-and-Set（TAS）锁

特点：

- 实现简单
- 低争用时进入开销小

缺点：

- 高争用时每次自旋都在写同一位置
- 会造成极高一致性流量和互连压力

### 2.2 Test-and-Test-and-Set（TTAS）锁

思路：

- 自旋阶段先做只读轮询
- 只有看到可能可用时再尝试原子获取

这样能减少无意义的失效风暴。

### 2.2.1 TTAS 锁的精确伪代码

```c
void lock(volatile int* lock) {
    while (1) {
        while (*lock != 0);  // 只读自旋（避免写总线）
        if (test_and_set(lock) == 0)
            break;  // 成功获取
    }
}
```

关键点：内层 `while (*lock != 0)` 只做只读轮询，**只有看到可能可用时才尝试原子获取**。注意 `*lock` 不应被编译器优化到寄存器中。

### 2.2.2 TTAS 的缓存流量优势

等待阶段仅用 BusRd（共享读）而非 BusRdX（独占写）。释放锁时才触发一次 BusRdX 失效所有等待者的共享副本。相比 TAS 每次自旋都产生 BusRdX 要好得多。

复杂度分析：
- TTAS 等待阶段：每次释放产生 O(P) 次失效（每个等待处理器一次）
- 但若所有处理器都缓存了锁变量 → O(P²) 互连流量
- TAS：每个等待处理器**每次 test** 都产生失效

### 2.3 Ticket Lock

思路：

- 每个线程先取一个号
- 按号等待轮到自己

优点：

- 公平（FIFO）
- 每次解锁只需推进一个共享计数
- 一致性流量较可控

### 2.3.1 Ticket Lock 精确实现

```c
struct lock {
    int next_ticket;
    int now_serving;
};

void lock(lock* l) {
    int my_ticket = atomic_increment(&l->next_ticket);
    while (l->now_serving != my_ticket);  // 只读等待
}

void unlock(lock* l) {
    l->now_serving++;
}
```

关键设计：使用 `atomic_increment` 取号，不需要原子操作来获取锁（只需等待 `now_serving` 等于自己的票号）。

### 2.4 CAS Lock

利用 compare-and-swap 构造更灵活的加锁与原子更新方案。

> 对应源码：`lecture16_part1.cpp`
> 内容：TAS、TTAS、ticket、CAS 风格锁以及基于 CAS 的原子更新思路。

### 2.4.1 用 atomicCAS 构造任意 Fetch-and-Op

```c
int atomicCAS(int* addr, int compare, int val) {
    int old = *addr;
    if (*addr == compare) *addr = val;
    return old;
}

// 用 CAS 循环实现 atomic_min
void atomic_min(int* addr, int value) {
    int old, new_val;
    do {
        old = *addr;
        new_val = min(old, value);
    } while (atomicCAS(addr, old, new_val) != old);
}
```

### 2.4.2 CAS 锁的两种实现

**基本版**：
```c
void lock(int* l) {
    while (atomicCAS(l, 0, 1) == 1);
}
```

**优化版**（先只读自旋再用 CAS）：
```c
void lock(int* l) {
    while (1) {
        while (*l == 1);       // 只读自旋
        if (atomicCAS(l, 0, 1) == 0) break;
    }
}
```
优化版在高争用下可能更高效。

### 2.4.3 Load-Linked / Store-Conditional (LL/SC)

ARM 使用 LDREX/STREX 而非单一原子指令：
```c
int load_linked(int* x);           // 加载值并标记地址
bool store_conditional(int* x, int v);  // 仅当 x 未被其他处理器写时才存储
```
- 不保证原子性，只保证条件性——若中途被修改则 SC 失败
- 实现依赖于缓存一致性协议跟踪"exclusive access"状态

### 2.4.4 C++11 `atomic<T>` 示例

```cpp
std::atomic<int> counter(0);
counter++;  // 原子递增
int old = counter.load();  // 原子读
int expected = old;
counter.compare_exchange_strong(expected, old + 1);  // CAS
bool lock_free = counter.is_lock_free();  // 是否无锁实现
```

原子性可能通过互斥锁或硬件原子指令实现。默认提供 sequential consistency 内存序。

### 2.4.5 CUDA 原子操作完整列表

`atomicAdd`、`atomicSub`、`atomicExch`、`atomicMin`、`atomicMax`、`atomicInc`、`atomicDec`、`atomicCAS`、`atomicAnd`、`atomicOr`、`atomicXor`（含 int/float/unsigned 重载）

---

## 3. 为什么锁扩展性差异巨大

### 3.1 临界区时间不是唯一因素

锁性能不只看：

- 临界区长短

还要看：

- 获取 / 释放锁时产生多少一致性消息
- 自旋线程是在读还是在写同一 cache line
- 是否公平
- 是否会导致某线程长期饿死

### 3.2 一致性视角看锁

很多锁实现的真正性能差别来自：

- 释放锁时是否让所有等待线程一起竞争同一行
- 自旋时是否不断制造缓存失效

所以锁实现本质上是一个：

- 共享状态组织
- 缓存一致性流量管理
- 公平性与延迟权衡

的问题。

---

## 4. 细粒度锁：用更多锁换更多并行

### 4.0.1 Hand-over-Hand 链表插入的完整代码

```c
struct Node { int value; Node* next; Lock* lock; };

void List::insert(int val) {
    Node* new_node = new Node(val);
    lock(head->lock);
    Node* cur = head;
    lock(cur->next->lock);
    Node* nxt = cur->next;
    unlock(cur->lock);
    
    while (nxt->value < val) {
        cur = nxt;
        nxt = nxt->next;
        lock(nxt->lock);
        unlock(cur->lock);
    }
    
    new_node->next = nxt;
    cur->next = new_node;
    unlock(cur->lock);
    unlock(nxt->lock);
}
```

### 4.0.2 细粒度锁的额外代价

遍历现在涉及**内存写入**（因为要写锁变量），这会"污染"缓存——原本只读的遍历现在在缓存一致性层面变成读写操作。

### 4.0.3 无锁链表插入（CAS 实现）

```c
void insert_after(Node* prev, Node* new_node) {
    do {
        new_node->next = prev->next;
    } while (!atomicCAS(&prev->next, new_node->next, new_node));
}
```

无锁版本没有加锁开销，也没有每节点存储锁的开销。但**删除操作复杂得多**——存在经典的"删除 B 时有人在 B 之后插入 E"的竞态问题。

### 4.1 基本动机

粗粒度锁简单，但会把大量本可并行的操作串行化。
细粒度锁希望：

- 只锁真正冲突的局部部分
- 让对不同区域的操作并发执行

### 4.2 Hand-over-Hand（链式传锁）

以链表遍历为例：

1. 先锁当前节点
2. 再锁下一个节点
3. 然后释放前一个节点

### 4.3 为什么它能避免死锁

因为锁获取顺序固定沿链表方向推进，不会形成循环等待。

### 4.4 它的代价

- 每走一步都要加解锁
- 每个节点都要存锁
- 正确性推理更复杂
- 细粒度不等于低开销

> 对应源码：`lecture16_part2.cpp`
> 内容：全局锁链表与 hand-over-hand 锁链表的对比。

---

## 5. Lock-Free：保证系统级前进，而不是线程都顺利

### 5.1 什么叫 lock-free

lock-free 的正式含义不是“代码里没写锁”，而是：

- 无论如何调度，总有某个线程能在有限步内完成操作

### 5.2 与 blocking 的差别

在 blocking 结构中：

- 一个持锁线程若被抢占或崩溃，别人可能全部停住

在 lock-free 结构中：

- 某个线程失败或重试，不应无限期阻止全系统进展

### 5.3 为什么这在某些系统特别重要

- 操作系统内核
- 数据库
- 高并发 runtime
- 被抢占风险高的环境

这些地方尤其怕“拿着锁的人被挂起”。

---

## 6. CAS 循环：无锁结构的核心模板

无锁代码常见模式：

1. 读取当前共享状态
2. 在本地推导“下一状态”
3. 用 CAS 尝试提交
4. 若失败，说明别人抢先更新了，重试

### 6.1 这为什么难

- 你必须保证读取到的旧状态在重试期间仍可安全解释
- 必须处理并发失败、重试风暴和内存回收

---

## 7. ABA 问题：CAS 最经典的陷阱

### 7.1 问题描述

线程 A 读取某指针值为 `A`，准备 CAS。
期间线程 B 把它改成 `B`，又改回 `A`。
线程 A 此时 CAS 看到值还是 `A`，误以为“什么都没发生”。

### 7.2 为什么危险

虽然位模式回到原值，但数据结构语义可能已经完全变了：

- 中间节点可能被弹出
- 对象可能已回收或重用
- 旧 next 指针可能不再合法

### 7.3 常见解法

- 指针加版本号 / 计数器
- 双宽 CAS
- hazard pointers
- epoch-based reclamation

> 对应源码：`lecture16_part3.cpp`
> 内容：无锁栈、ABA 反例、计数器版本修复思路。

---

## 8. 单生产者单消费者队列：为什么它相对简单

### 8.0.1 单生产者单消费者有界队列完整代码

```c
struct SPSC_queue {
    int data[N];
    int head, tail;
};

void push(int val) {
    if (tail == MOD_N(head - 1)) return;  // 满
    data[tail] = val;
    tail = MOD_N(tail + 1);
}

int pop() {
    if (head == tail) return -1;  // 空
    int val = data[head];
    head = MOD_N(head + 1);
    return val;
}
```
关键约束：仅两个线程（一个生产者一个消费者），线程从不相互等待。

### 8.0.2 无界队列的内存回收（Reclaim）机制

- 节点分配和释放都由**同一线程（生产者）**执行
- 每次 push 执行延迟删除（从消费者已 pop 的节点中选择安全的删除）
- 5 步完整工作流程：push 3, push 10 → pop 3 → pop 10 → push 5 触发 reclaim

### 8.0.3 无锁栈 ABA 问题的 12 步时间线

1. Thread 0 开始 pop：读 old_top=A, new_top=B
2-4. Thread 1 pop 出 A, 修改 A 的值, push A 回去
5-6. Thread 1 push D
7-11. Thread 0 仍认为 old_top=A, new_top=B
12. Thread 0 的 CAS 成功（设置 top=B）→ **D 丢失！栈结构损坏！**

关键注释：A, B, C, D 是节点**地址**，不是节点值。

### 8.0.4 DCAS 与双宽 CAS 解决方案

```c
// 使用 pop_count 计数器 + double_compare_and_swap
old_count = s->pop_count;
new_top = old_top->next;
new_count = old_count + 1;
success = DCAS(&s->top, &s->pop_count,
               old_top, old_count,
               new_top, new_count);
```

x86 支持：`cmpxchg8b`（8 字节 CAS，两个 32-bit）、`cmpxchg16b`（16 字节 CAS，两个 64-bit）。确保 top 和 count 字段在内存中连续即可。

### 8.0.5 Use-After-Free 问题

即使在 DCAS 之后仍存在问题：pop 路径中 `old.top->value` 的访问可能在 `old.top` 已被其他线程释放之后发生。"old top might have been freed at this point (by some other thread that popped it)."

### 8.0.6 Hazard Pointer 完整实现

```c
// 每线程：
hazard_ptr;      // 声明正在访问的节点
retireList[];    // 待删除节点列表

void retire(Node* node) {
    retireList.push(node);
    if (retireList.size() > THRESHOLD) {
        for each node in retireList:
            if no_thread_has_hazard_ptr_for(node):
                delete node;
    }
}
```
遍历所有线程的 hazard 指针，只删除没有任何线程引用的节点。这是 lock-free 编程中内存回收的经典方案。

### 8.0.7 课程总结与事务内存预告

- 细粒度锁减少争用、最大化并行度
- Lock-free 是非阻塞方案，但实现棘手，正确性有其自身的开销
- **"A lock-free design does not eliminate contention — CAS can fail under heavy contention, requiring spins."**
- 预告下一讲：事务内存——一种更通用的机制，允许系统推测操作将成功完成，并在其他线程确实修改结构时提供 "abort" 操作的能力。

### 8.1 关键性质

如果一个队列只有：

- 一个生产者修改尾指针
- 一个消费者修改头指针

那么很多冲突会自然消失。

### 8.2 为什么能少很多同步

- 头尾修改权责天然分离
- 不需要多个生产者争同一尾部更新
- 不需要多个消费者争同一头部更新

### 8.3 这说明什么

并发数据结构设计中，**限制并发模式本身**就是一种强大的简化手段。

> 对应源码：`lecture16_part4.cpp`
> 内容：有界 / 无界 SPSC 队列、循环缓冲与链表队列示例。

---

## 9. 细粒度锁与无锁，如何选择

### 9.1 细粒度锁适合

- 数据结构分区清晰
- 冲突相对局部
- 需要较好可读性与较低实现复杂度

### 9.2 无锁适合

- 不希望线程被持锁者阻塞
- 被抢占风险高
- 极端延迟或鲁棒性要求高

### 9.3 真正的权衡

无锁并不总更快。它常常意味着：

- 更复杂的正确性证明
- 更棘手的内存回收
- 可能的高重试开销

所以是否值得，取决于场景，而不是“无锁一定更高级”。

---

## 常见误区

1. **误区：锁性能只由原子指令开销决定。**
   一致性流量和公平性同样决定扩展性。
2. **误区：细粒度锁一定优于粗粒度锁。**
   它可能因加锁次数太多而更慢。
3. **误区：无锁就是更快的无阻塞代码。**
   lock-free 的关键是进展保证，不等于最佳性能。
4. **误区：CAS 成功就代表正确。**
   ABA 与内存回收问题会让“看似成功”的 CAS 仍然错误。

---

## 对应源码

| 文件 | 主题 | 重点 |
|---|---|---|
| `lecture16_part1.cpp` | 锁实现 | TAS、TTAS、ticket、CAS 对比 |
| `lecture16_part2.cpp` | 细粒度链表锁 | hand-over-hand 的并行性与复杂度 |
| `lecture16_part3.cpp` | 无锁栈 | CAS 循环、ABA、版本计数 |
| `lecture16_part4.cpp` | 无锁队列 | SPSC 条件下如何简化同步 |

---

## 学完本讲应做到

- 能从一致性流量视角理解不同锁的扩展性差异。
- 能解释 hand-over-hand 锁为什么有效、代价又在哪里。
- 能说清楚 lock-free 的正式含义。
- 能识别 ABA 与内存回收是无锁结构中的核心难点。

