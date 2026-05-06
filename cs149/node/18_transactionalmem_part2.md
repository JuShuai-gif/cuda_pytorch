# CS149 第 18 讲：事务内存（下）与课程总结

**PDF**：Lecture 18 - Transactional Memory Part II + Course Wrap Up

**课程**：Stanford CS149，2025 年秋季

---

## 本讲核心问题

1. 软件事务内存（STM）和硬件事务内存（HTM）分别怎么做？
2. 事务记录、读集合、写集合、版本号这些结构为什么必要？
3. 为什么 HTM 很快但通常不能单独保证前进？
4. 回看整门课，哪些思想贯穿了所有主题？

---

## 1. 事务实现空间回顾

不同 TM 系统通常围绕三类设计选择展开：

1. **版本管理**：eager 还是 lazy
2. **读冲突检测**：optimistic 还是 pessimistic
3. **写冲突检测**：optimistic 还是 pessimistic

第 18 讲的重点是：

- 这些策略在软件和硬件里分别如何落地
- 实现细节会引入哪些现实开销

---

## 2. 软件事务内存（STM）的核心运行时结构

### 2.0.1 TM 实现空间的具体系统举例

| 系统 | 版本管理 | 读检测 | 写检测 | 类型 |
|---|---|---|---|---|
| Sun TL2 | Lazy | Optimistic | Optimistic | STM |
| MS OSTM | Lazy | Optimistic | Pessimistic | STM |
| Intel STM | Eager | Optimistic | Pessimistic | STM |
| Intel STM | Eager | Pessimistic | Pessimistic | STM |
| Stanford TCC | Lazy | Optimistic | Optimistic | HTM |
| MIT LTM/Intel VTM | Lazy | Pessimistic | Pessimistic | HTM |
| Wisconsin LogTM | Eager | Pessimistic | Pessimistic | HTM |

### 2.0.2 STM 代码转换对照

```c
// 原始 atomic 代码
atomic {
    bar.x = foo.x;
    bar.y = foo.y;
}

// 转换后
txn_desc* tx = tmTxnBegin();
int temp_x = tmRd(tx, &foo.x);  // 事务读屏障
tmWr(tx, &bar.x, temp_x);       // 事务写屏障
int temp_y = tmRd(tx, &foo.y);
tmWr(tx, &bar.y, temp_y);
tmTxnCommit(tx);
```

### 2.0.3 数据到事务记录的三种映射方式

1. **Java/C#**：在每个对象中嵌入事务记录（对象级）
2. **C/C++**：基于地址哈希到全局事务记录表（cache-line 或 word 粒度）
3. **字段/数组元素级哈希**：`f(obj.hash, field.index)` → 更少的假冲突，更多查找开销

### 2.0.4 Intel McRT STM 的 32 位事务记录格式

- **LSB=0**: 被写者锁定（指向拥有者事务的指针）
- **LSB=1**: 未锁定（31 位为最后提交的时间戳/版本号）
- 全局时间戳递增 by 2（因为 LSB 用于锁标志）

### 2.0.5 STM 完整操作规范

```c
// STM Read (Optimistic)
value = *addr;  // eager: 直接读内存
validate: 数据未被锁定 且 版本 ≤ 本地时间戳
          → 否则验证整个 read-set 一致性
插入 read-set; return value;

// STM Write (Pessimistic)
validate(); acquire lock; 插入 write-set;
创建 undo log; *addr = new_value;  // 原地写

// STM Commit
原子递增全局时间戳 by 2;
检查是否有最近提交的事务;
释放 write-set 中所有锁 + 设置版本号;
```

### 2.0.6 STM 四大挑战

1. 软件屏障的开销
2. 函数克隆（事务内外需要不同版本的函数）
3. 鲁棒的争用管理
4. 内存模型（强原子性 vs 弱原子性）

### 2.0.7 STM 编译器优化：从单体屏障到分解屏障

编译器优化三步：
1. 单体屏障形式：`tmTxnBegin/tmWr/tmRd/tmTxnCommit`——隐藏可优化的冗余
2. 分解屏障：`txnOpenForWrite/txnLogObjectInt`——暴露冗余
3. 优化后：合并冗余操作 → **单线程开销 <40%（vs 无并发控制），<30%（vs 基于锁的同步）**

### 2.0.8 STM 开销量化

- STM 每线程 slowdown 为 2-8 倍（因软件屏障）
- "大部分时间花在 read barriers 和 commit 上——大多数应用读的数据比写的多"

### 2.1 事务描述符（Transaction Descriptor）

每个活动事务通常需要维护：

- 事务状态
- 读集合（read set）
- 写集合（write set）
- undo log 或 write buffer
- 本地版本时间戳等信息

### 2.2 事务记录（Transaction Record）

共享内存中的对象或地址范围，通常要与某种元数据关联：

- 当前版本号
- 是否被事务独占写入
- 当前持有写权限的事务是谁

### 2.3 为什么这些结构无法省略

因为 STM 不能直接依赖硬件缓存协议帮它自动跟踪事务集，所以它必须：

- 显式记录自己看过什么
- 显式记录自己打算写什么
- 显式验证这些东西提交前是否仍然合法

---

## 3. STM 的典型执行路径

### 3.1 读路径

一个事务读取对象时，通常会：

1. 读对象当前值
2. 检查对象版本或锁状态是否合法
3. 把该对象加入 read set

### 3.2 写路径

一个事务写对象时，通常会：

- 若是 eager：先记录旧值，再原地写入
- 若是 lazy：把新值暂存到 write set / buffer 中

### 3.3 提交路径

提交往往要做：

- 获取必要写权限
- 验证读集合是否仍然有效
- 更新版本号
- 刷新写集合或发布写入

### 3.4 代价在哪里

每次读写都不再是“裸 load/store”，而要走事务屏障（barrier）。
这也是 STM 开销大的主要原因之一。

---

## 4. STM 的主要痛点：软件屏障开销

### 4.1 为什么 STM 常被嫌慢

因为它常对每次共享访问都插入额外逻辑：

- 版本检查
- 读 / 写集合维护
- 锁状态判定
- 日志或缓冲更新

### 4.2 后果

即使完全没有冲突，单线程事务路径也可能显著慢于普通顺序代码。

### 4.3 为什么编译器优化很重要

很多事务屏障存在冗余，编译器可尝试：

- 合并相邻访问
- 消除重复 open-for-read / open-for-write
- 缩减不必要元数据维护

这说明事务系统能否实用，不只取决于运行时，也取决于编译器支持。

---

## 5. 硬件事务内存（HTM）：把事务元数据藏进缓存协议

### 5.0.1 HTM 缓存行标记位

除 MESI 状态位外增加：
- **R bit**：在事务的 read set 中
- **W bit**：在事务的 write set 中

四种冲突检测触发方式：
1. 收到对 W-word 的 shared request → 读-写冲突
2. 收到对 R-word 的 exclusive request → 写-读冲突
3. 收到对 W-word 的 exclusive request → 写-写冲突
4. Eager versioning 需要额外缓存写入用于 undo log

### 5.0.2 HTM 懒乐观实现的完整流程

从 CPU 增加 TM State 寄存器开始，逐步展示 Xbegin→Load→Store→Xcommit 的完整缓存行状态变化（R/W bit、V bit、Tag、Data 每一步的变化）。

关键步骤：
- Load 操作时标记 read set
- Store 操作标记 write set
- Commit 时的 "Fast two-phase commit"——验证并申请 exclusive 访问、批量 reset R/W 位
- Abort 时：收到远程 commit 的 coherence requests → 检测冲突 → invalidate write set → gang-reset → restore register checkpoint

### 5.0.3 HTM 性能对比

- HTM 比 STM 快 2-7 倍（Vacation 基准测试）
- 单线程开销在顺序执行 10% 以内
- 扩展性接近理想

### 5.0.4 硬件加速的三种类型

1. **硬件加速 STM (HASTM)**：在 STM 上加简单硬件原语加速瓶颈，但仍保留软件屏障
2. **硬件 TM (HTM)**：版本化和冲突检测直接由硬件完成，无软件屏障
3. **混合 TM (Hybrid)**：根据事务特性和资源在 HTM/STM 之间切换（如 Sun Rock）

### 5.0.5 Intel RTM 的关键限制

- Haswell 新增指令：`xbegin`（含 fallback address）、`xend`、`xabort`
- 在 L1 缓存中跟踪读写集
- 多种原因可导致自动 abort（读写集中缓存行被逐出等）
- **实现不保证进展**：必须提供 fallback 路径回退到锁
- Intel 优化手册第 12 章：增加事务成功概率的指南

### 5.1 HTM 的关键想法

利用缓存天然已经追踪：

- 读过哪些缓存行
- 写过哪些缓存行
- 其他核心是否访问同一行

于是可以：

- 用缓存行元数据近似事务读写集合
- 用一致性协议检测事务冲突

### 5.2 为什么这很快

- 读写集合记录由硬件隐式完成
- 冲突检测与缓存协议自然耦合
- 提交 / abort 路径可用微架构快速完成

### 5.3 但它的粒度通常是缓存行

这意味着：

- 会存在假冲突
- 一旦事务触及太多行，容量限制就会出现

---

## 6. HTM 的提交与回滚

### 6.1 提交

若使用 lazy 版本管理：

- 写入先保留在缓存中的事务状态下
- 提交时统一宣布生效并清理事务标记

### 6.2 回滚

若事务 abort：

- 丢弃写集合中尚未正式提交的内容
- 恢复寄存器检查点
- 清除事务跟踪位

### 6.3 为什么硬件能做得快

因为这些状态大多已经在片上，而且由硬件直接维护，不需像 STM 那样通过软件数据结构间接追踪。

---

## 7. HTM 的关键限制：不保证成功，也不总保证前进

### 7.1 常见 abort 原因

- 数据冲突
- 事务读写集合太大，超出缓存容量
- 中断、系统事件、页错误等异步因素
- 不支持的指令或系统调用

### 7.2 最重要的现实结论

HTM 常常只能作为“快速路径”。
一旦事务反复失败，系统必须：

- 回退到锁
- 或回退到 STM / 其他保底机制

### 7.3 为什么这很重要

否则程序可能在冲突或容量限制下无限 abort，失去进展保证。

---

## 8. Intel RTM 一类实际 HTM 的启示

现实工业 HTM 通常并不是“万能事务机”，而是：

- 尽量快
- 尽量适合中小事务
- 失败时必须有后备路径

因此编程时要认识到：

- HTM 更像一种投机优化机制
- 不是取代所有锁的绝对语义基础

---

## 9. 整门课最值得回收的统一主线

### 9.0.1 Transactional Memory 的生产力量化

"事务内存可以达到专家级细粒度锁编程 90% 的性能收益，但只需 10% 的开发时间。"

### 9.0.2 课程覆盖的多尺度视角

从单核到分布式集群的六个尺度：
1. 异构移动 SoC
2. 多核 CPU
3. 多核 GPU
4. CPU + GPU 组合
5. 集群
6. AI 加速器硬件

### 9.0.3 课程覆盖的抽象层次

数据并行思维（map/reduce/scan）、函数并行（cilk_spawn）、事务（atomic {}）、任务（ISPC task）、SPMD（ISPC gang/CUDA warp）

### 9.0.4 现代软件的惊人低效

> "现代软件相对于现代机器的峰值能力惊人地低效——大量性能被留在桌面上"

### 9.0.5 未来高性能计算的必然方向

"在可预见的未来，获取更高性能计算硬件的主要途径仍是**增加并行性与硬件专用化的结合**。"

硬件示例：NVIDIA GPU (32-wide SIMD, 2048 CUDA threads/SM, Tensor Cores)、Apple A11 异构 SoC、Intel CPU+GPU、Google TPU、AWS Trainium 等。

### 9.1 并行必须显式暴露

- 单核自动提速时代已经结束
- 性能增长依赖程序员或编译系统主动暴露并行结构

### 9.2 数据移动经常比计算更贵

- 缓存
- 局部性
- 块化
- 融合
- shared memory / SRAM / register 复用
- HBM 与集体通信

几乎每一讲都在不同层次重复这件事。

### 9.3 抽象决定优化上限

- ISPC / SPMD
- 数据并行原语
- CUDA 层次
- Halide 调度
- DSL 与 auto-scheduler
- 事务内存的声明式原子块

高层抽象不是“远离性能”，而是决定系统能否自动优化的前提。

### 9.4 正确性与性能是耦合的

- 锁实现影响一致性流量
- 一致性模型影响程序可见行为
- 同步粒度既决定正确性也决定速度
- 事务实现既有语义问题也有性能问题

---

## 10. 回头看：这门课希望你真正带走什么

不是记住每个协议状态名或每个 API，而是形成以下能力：

1. 看见并行机会
2. 用合适抽象表达并行
3. 识别数据移动成本
4. 在不同硬件层次安排局部性与同步
5. 对共享内存正确性保持严格敬畏
6. 理解系统、编译器、硬件是共同完成性能的

---

## 常见误区

1. **误区：STM 慢，HTM 快，所以 STM 没意义。**
   STM 提供更强的软件可实现性与通用性，HTM 则提供快路径，两者定位不同。
2. **误区：HTM 能彻底替代锁。**
   容量、冲突和异步 abort 使其几乎总需要后备方案。
3. **误区：事务内存只是并发理论话题。**
   它连接了硬件、语言、编译器与运行时设计。
4. **误区：课程后半段跟前半段脱节。**
   实际上一直在围绕并行、数据移动、抽象与正确性四条主线展开。

---

## 对应源码

| 文件 | 主题 | 重点 |
|---|---|---|
| `lecture18_part1.cpp` | STM 结构与流程 | 事务描述符、读写集合、版本验证 |
| `lecture18_part2.cpp` | HTM 机制 | 缓存中跟踪读写集合、冲突与 abort |
| `lecture18_part3.cpp` | 事务对比与课程总结 | STM / HTM 权衡、回退路径与总复盘 |

---

## 学完本讲应做到

- 能解释 STM 为什么需要大量元数据与软件屏障。
- 能说明 HTM 为什么快、又为什么不能单独保证前进。
- 能把事务内存和缓存协议、编译器优化联系起来理解。
- 能用整门课的统一主线回看所有主题，而不是把它们看成孤立知识点。

