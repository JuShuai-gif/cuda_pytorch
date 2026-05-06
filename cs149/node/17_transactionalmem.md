# CS149 第 17 讲：事务内存（上）

**PDF**：Lecture 17 - Transactional Memory

**课程**：Stanford CS149，2025 年秋季

---

## 本讲核心问题

1. 为什么锁、原子和无锁结构仍然让并发编程很痛苦？
2. 事务内存试图提供什么更高层抽象？
3. `atomic {}` 与 `lock()/unlock()` 的区别到底是什么？
4. 事务内存中的版本管理与冲突检测有哪些设计空间？

---

## 1. 为什么还需要更高层同步抽象

### 1.0.1 "Between a Lock and a Hard Place"

锁的核心矛盾——强制在"并发度→性能"与"竞争/死锁风险→正确性"之间做权衡：
- 粗粒度锁：低并发、高正确性概率
- 细粒度锁：高并发、低正确性概率

这是经典的双关命名（Lock ≈ 锁，Between a Rock and a Hard Place ≈ 左右为难）。

### 1.0.2 声明式 vs 命令式抽象

| 维度 | 声明式 | 命令式 |
|---|---|---|
| 任务并行 | "Execute all these independent 1000 tasks" | "Spawn N threads. Assign work from shared queue" |
| 同步 | "Perform this set of operations atomically" | "Acquire lock, operate, release lock" |

TM 提供的是声明式同步抽象。

### 1.0.3 TM 语义的三要素

1. **原子性**：提交时全部写入同时生效，abort 时写入如同从未发生
2. **隔离性**：无其他处理器可在提交前看到写入
3. **可串行化**：事务看起来按某个串行顺序提交——但*确切提交顺序不保证*

TM 可类比于"在一组读写地址上维护我们曾在 cache coherence 中为单地址维护的属性"。

### 1.1 低层同步原语的问题

- 原子指令太底层，难直接组织复杂临界区
- 锁容易产生死锁、优先级反转、组合性差
- 无锁结构虽然强大，但实现难度极高

### 1.2 程序员真正想表达的通常是

- “这一段代码必须原子地、隔离地执行”

而不是：

- “请用某种具体锁顺序帮我实现它”

这就引出了事务内存（TM）的价值：

- 让程序员描述**需要什么语义**
- 而不是亲自指定所有同步机制细节

---

## 2. 事务语义：数据库思想进入并发内存编程

### 2.1 原子性（atomicity）

事务中的所有写入要么全部生效，要么全部不生效。

### 2.2 隔离性（isolation）

事务执行期间，其他线程不应看到它的半成品状态。

### 2.3 可串行化（serializability）

多个事务的最终效果应当等价于某个串行顺序依次执行。

### 2.4 为什么这很吸引人

它允许程序员写出接近顺序逻辑的并发代码，而不必手工管理大量锁顺序与局部互斥细节。

---

## 3. `atomic {}` 不等于简单的 `lock/unlock`

### 3.0.1 Java HashMap 的三种方案演进

1. **原始 HashMap**：get 无锁 → 非线程安全
2. **synchronized HashMap**（粗粒度锁）：线程安全但限制并发
3. **事务 HashMap**：`atomic { return m.get(key); }` — 线程安全且易编程，性能取决于工作负载和实现

### 3.0.2 双向链表 PushLeft 事务示例

```c
// 需要原子地更新两个指针
atomic {
    new_node->right = leftSentinel->right;
    new_node->left = leftSentinel;
    leftSentinel->right->left = new_node;
    leftSentinel->right = new_node;
}
```

### 3.0.3 锁的不可组合性：transfer 死锁

```java
synchronized(A) {
    synchronized(B) {
        A.withdraw(amount);
        B.deposit(amount);
    }
}
// 同时：transfer(A,B) 和 transfer(B,A) → DEADLOCK
```

在 TM 中：两个 `atomic` 事务可并发提交，系统负责冲突检测和串行化。

### 3.0.4 用 flag 同步的示例说明 `atomic ≠ lock`

两个线程各自持有不同锁、设置 flag、然后自旋等待对方——这是**不能用 `atomic` 直接替换 `synchronized` 的典型反例**，因为锁在此处还被用于**条件同步**（超越原子性的目的）。

### 3.0.5 拆分 atomic 块的原子性违规

```c
// 错误：两个 atomic 块之间存在 window
atomic { ptr = A; }
// ← Thread 2 在这里设置 ptr = NULL
atomic { B = ptr->field; }  // 空指针解引用！
```

尽管每个 atomic 块内部是原子的，但**两个块之间的间隙**破坏了整体的原子性意图。

### 3.1 声明式 vs 命令式

- `atomic {}` 是声明式语义：要求系统保证这段逻辑具有事务语义。
- `lock/unlock` 是命令式原语：程序员手工指定如何互斥。

### 3.2 为什么两者不等价

同一段 `atomic {}` 可以用：

- 粗粒度锁实现
- 细粒度锁实现
- 软件事务内存实现
- 硬件事务内存实现

因此 `atomic` 关注的是“做成什么”，不是“怎么做”。

### 3.3 事务的额外价值

- 天然支持失败回滚
- 更容易组合多个子操作
- 对读读并发更友好
- 在某些场景下可避免复杂全局锁排序

> 对应源码：`lecture17_part1.cpp`
> 内容：银行转账中锁方案与事务语义方案的对比、组合性示例。

---

## 4. 为什么事务内存吸引人

### 4.1 易用性

程序员只需声明一段代码应保持原子与隔离，不必从零设计锁协议。

### 4.2 组合性

如果两个事务化函数各自正确，组合后通常也更容易保持正确。

锁则常见问题是：

- 单个函数内部锁顺序没问题
- 组合在一起后却引入新死锁风险

### 4.3 故障原子性

如果事务中途失败：

- 可以 abort 并回滚
- 不会像“线程死在持锁状态”那样把结构锁死

---

## 5. 事务实现的两个核心维度

事务内存实现设计空间很大，但第 17 讲先抓住两个主轴：

1. **数据版本管理（versioning）**
2. **冲突检测策略（conflict detection）**

---

## 6. 数据版本管理：eager 与 lazy

### 6.0.1 Eager Versioning 四步生命周期

1. **Begin**: 内存 X=10, undo log 空
2. **Write x←15**: 内存 X=15, undo log: X=10（原地写，记旧值）
3. **Commit**: 清空 undo log（数据已就地）
4. **Abort**: 从 undo log 恢复 X=10

### 6.0.2 Lazy Versioning 四步生命周期

1. **Begin**: 内存 X=10, write buffer 空
2. **Write x←15**: 内存仍是 X=10, write buffer: X=15
3. **Commit**: 内存统一更新为 X=15
4. **Abort**: 直接丢弃 write buffer，内存保持 X=10

### 6.0.3 Pessimistic vs Optimistic 冲突检测

**悲观检测四个案例**：Success、Early detect and stall、Abort、No progress（需处理 livelock）

**乐观检测四个案例**：Success、Abort（先提交者胜出）、Success、Forward progress

| 维度 | Pessimistic (a.k.a Eager) | Optimistic (a.k.a Lazy/Commit) |
|---|---|---|
| 优点 | 早检测、少浪费工作、可将 abort 转 stall | 有进展保证、批量通信和检测 |
| 缺点 | 无进展保证、某些情况更多 abort、每 load/store 细粒度通信 | 检测得晚、有公平性问题 |

### 6.1 Eager / Undo-Log

做法：

- 事务直接写真实内存
- 同时把旧值记到 undo log 中

优点：

- commit 便宜，因为数据已写到位

缺点：

- abort 成本高，要逐项回滚
- 如果线程崩溃，系统必须确保不会留下半提交状态

### 6.2 Lazy / Write Buffer

做法：

- 事务写入先进入私有缓冲
- commit 时再批量刷新到共享内存

优点：

- abort 很便宜，直接丢弃缓冲即可
- 容易保证外界看不到半成品

缺点：

- commit 成本更高
- 需要在提交时做更系统的冲突处理与写回

> 对应源码：`lecture17_part2.cpp`
> 内容：eager 与 lazy 版本管理的行为、提交与回滚路径对比。

---

## 7. 冲突检测：pessimistic 与 optimistic

### 7.1 Pessimistic（悲观）

假设冲突很可能发生，因此：

- 尽早检测
- 甚至提前加锁或阻止冲突访问

优点：

- 冲突早发现，减少无用工作

缺点：

- 每次访问都可能多出检查成本
- 更容易让事务相互等待

### 7.2 Optimistic（乐观）

假设冲突不常见，因此：

- 先大胆执行
- 在提交时集中验证

优点：

- 日常访问路径更轻
- 若冲突少，整体更高效

缺点：

- 冲突晚发现，可能白做不少工作

### 7.3 为什么没有统一最好答案

它取决于：

- 冲突概率
- 事务大小
- 回滚成本
- 是否需要进展保证

---

## 8. 锁与事务的性能关系不是简单替代

事务并不总比锁快，也不总比锁慢。真正决定效果的是：

- 事务粒度是否合适
- 冲突率是否低
- 读多写少还是写热点很强
- 实现是软件、硬件还是混合

### 8.1 事务常有优势的场景

- 读多写少
- 组合复杂、锁顺序难维护
- 希望避免过度串行化

### 8.2 锁仍然很有价值的场景

- 访问模式简单稳定
- 冲突热点非常明确
- 事务开销相对太高

---

## 常见误区

1. **误区：事务内存就是“自动帮你加锁”。**
   更准确地说，它提供更高层事务语义，可由多种机制实现。
2. **误区：`atomic {}` 一定比锁更快。**
   速度取决于实现方式与冲突模式。
3. **误区：事务只解决性能问题。**
   它同样在解决组合性和编程复杂度问题。
4. **误区：悲观检测一定更安全，乐观检测一定更高效。**
   实际上两者是成本结构不同的权衡。

---

## 对应源码

| 文件 | 主题 | 重点 |
|---|---|---|
| `lecture17_part1.cpp` | 锁 vs 事务语义 | 组合性、死锁风险、声明式原子块 |
| `lecture17_part2.cpp` | 版本管理 | eager / lazy 提交与回滚 |
| `lecture17_part3.cpp` | 冲突检测与事务行为 | optimistic / pessimistic 的时机与代价 |

---

## 学完本讲应做到

- 能解释事务内存试图提升的抽象层次。
- 能清楚区分 `atomic {}` 与显式锁的语义差异。
- 能比较 eager / lazy 版本管理与 optimistic / pessimistic 检测。
- 能理解事务内存的主要优势不仅是性能，还包括组合性与编程简化。

