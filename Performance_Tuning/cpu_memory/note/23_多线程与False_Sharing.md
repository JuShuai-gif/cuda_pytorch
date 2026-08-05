# 23 多线程与 False Sharing

> 对应 PDF：第 6.4 节 Multi-Thread Optimizations（PDFp65）、第 6.4.1 节 Concurrency Optimizations（PDFp65~67）、附录 A.3 Measure Cache Line Sharing Overhead（PDFp100~101）、图 6.10、图 6.11
> 本篇回答：多线程对缓存的三个挑战是什么？false sharing 到底发生了什么、代价多大？如何分组/分离变量避免它？TLS 何时有用？

## 1. 本章要解决的问题

- 多线程的缓存三挑战：并发、原子性、带宽。
- False Sharing：两个线程改不同变量为何互相拖累？
- 图 6.10/6.11：SMP vs 多核上的 false sharing 代价。
- 变量分类与放置：只读/读写分组、常写分离、TLS。
- 附录 A.3 测试程序怎么测量缓存行共享开销。

## 2. 前置知识

- note/09：MESI、RFO、一致性。
- note/19：缓存行、结构体布局。
- 线程同步基础（pthread、barrier、affinity）。

## 3. 核心概念

- **False Sharing（伪共享）**：两个线程修改**不同变量**，但它们在同一缓存行 → 每次写都触发 RFO，缓存行在核间来回倒手。
- **RFO（Request For Ownership）**：写前获取缓存行所有权。
- **Cache Line Contention（缓存行竞争）**：多核竞争同一缓存行。
- **Thread-Local Storage（TLS，线程局部存储）**：`__thread` 变量，每线程一份。
- **`__attribute__((section(".data.ro")))`**：把变量放入指定 section 分组。
- **`alignas`/`__attribute__((aligned))`**：按缓存行对齐，配合 padding 分离变量。
- **`__sync_add_and_fetch`**：gcc 原子自增 intrinsic。

## 4. 硬件工作流程

### 4.1 False Sharing 的场景

```text
struct { int a; int b; } s;   // a、b 在同一 64B 缓存行

线程0 反复写 s.a  ──► RFO 拿到行所有权 ──► 写 s.a
线程1 反复写 s.b  ──► RFO 从线程0 抢走行 ──► 写 s.b
                     （线程0 的行被作废 Invalid）

结果：每个写操作都伴随一次完整的缓存行所有权转移
```

- 两个变量逻辑独立、物理同住一行 → 相互干扰，叫"伪"共享（并非真的共享数据）。

### 4.2 三种变量的处理策略（PDFp66~67）

```text
① 只读/只初始化一次 → 常量 → S 态共享，无 RFO → 可分组（const → .rodata）
② 读写但被不同线程频繁写 → 各放一条缓存行（padding）→ 避免 RFO
③ 单线程专属 → TLS（__thread）→ 每线程独立副本
```

论文四条建议：

1. 至少分离"只读（初始化后）"与"读写"变量（可再加"读多写少"第三类）。
2. 把一起使用的读写变量放进一个结构体，确保它们物理相邻。
3. 常被不同线程写的变量，放到独立缓存行（尾部加 padding）。
4. 多线程各用各的独立变量 → 用 TLS。

### 4.3 图 6.10/6.11：代价数据

- 图 6.10（4×P4 SMP）：同缓存行 vs 各线程独立缓存行，开销 **390% / 734% / 1147%**（2/3/4 线程）。
- 图 6.11（Core 2 四核，共享 L2 对）：独立缓存行无扩展问题；同缓存行有轻微开销但不随核数增长。
- 结论：false sharing 在**多处理器（SMP）**上灾难性，多核共享缓存上较轻，但现代多路机器仍会踩坑。

## 5. PDF 核心观点

> 来源：PDF 第 65~67、100~101 页；对应章节 6.4、6.4.1、A.3、图 6.10、6.11。以下为概括。

1. **多线程三挑战**（PDFp65）：并发（多线程同跑的内存效应）、原子性（共享数据协调）、带宽（处理器/总线带宽有限）。
2. **并发优化的矛盾**（PDFp65）：一般缓存优化想把数据放一起减小 footprint；但多线程写会导致缓存行必须在各核 L1d 处于 E 态 → RFO 风暴。
3. **False Sharing 定义**（PDFp65）：各线程用不同位置（独立）但同缓存行，仍触发 RFO，写突然变贵。
4. **图 6.10 数据**（PDFp65）：4×P4，同缓存行 vs 独立行，开销 390%/734%/1147%。
5. **多核上较轻**（PDFp66，图 6.11）：Core 2 四核同缓存行有轻微开销但不随核数增长；但多处理器机器仍会严重，需在真 SMP 上测试。
6. **简单修复不可取**（PDFp66）：把每个变量放独立缓存行会大幅增大 footprint。
7. **变量分类**（PDFp66）：常量（const→.rodata）可共享；常写变量分离；单线程变量用 TLS。
8. **section 分组**（PDFp66~67）：`__attribute__((section(".data.ro")))` 把读写变量分组，保证物理相邻、中间没有常写变量。
9. **TLS**（PDFp67）：`__thread int bar` 每线程独立副本，无 false sharing；代价：线程创建时初始化开销、寻址更贵、DSO 中一个 TLS 变量会连带分配该对象全部 TLS 内存。
10. **结构体 + padding**（PDFp67）：把一起用的读写变量放结构体、按缓存行对齐、尾部 padding 填满行。
11. **附录 A.3**（PDFp100~101）：测试程序用 `__sync_add_and_fetch` 原子自增，或普通自增 + `asm volatile(""::"m"(*p))` 防止编译器把增量提出循环；线程用 affinity 钉到 0~3 核。

## 6. 通俗解释

False sharing 就像**合租一套房却各住一间的两人共用一个房门**：

> 变量 a 和 b 住在同一缓存行（同一间房）。线程 0 改 a、线程 1 改 b，逻辑上互不相干。
> 但物理上它们住一间房——谁要改，就得把整个房间从对方手里抢过来（RFO），
> 对方要改再抢回去。于是每写一次都要"搬家"，两个线程互相踢皮球，慢到爆炸（图 6.10：最高 1147%）。

为什么多核机器上没那么惨？

> 多核共享 L2/L3，抢房间的成本低一些（图 6.11 开销不随核数涨）。
> 但多路服务器（多物理 CPU）没有共享缓存，抢房间要跨互联线——又回到灾难级。

怎么解决？

> 给常写变量"一人一房"（padding/独立缓存行）、"分房"（section 分组）、
> 或者给每人发一套自己的房（TLS）。只读数据无所谓，大家共用一个房间没问题（S 态）。

## 7. 示例分析

### 7.1 为什么"不同变量"也会互相拖累

```cpp
struct { long a; long b; } shared;   // 同一缓存行
// 线程0: ++shared.a;   线程1: ++shared.b;
```

- 每线程每步：RFO 抢行 → 写 → 对方 RFO 抢行 → 写……
- 行在核间来回，任何一刻只有一个核能写，其余核全被阻塞。
- 论文图 6.10：2/3/4 线程分别慢 390%/734%/1147%。

### 7.2 修复：padding 分离

```cpp
struct alignas(64) Counter { long val; };   // 每计数器独立一行
std::vector<Counter> c(nthreads);
// 线程 t 写 c[t].val —— 各线程行不同，无 RFO 竞争
```

### 7.3 修复：TLS

```cpp
__thread long local_sum = 0;   // 每线程独立
// 最后合并
```

## 8. 未优化代码

对应 false sharing 的程序（两个线程写相邻变量）。

```cpp
// bad.cpp: 两个线程写同一结构体的不同字段（同缓存行）
#include <thread>

int main() {
    constexpr long N = 500'000'000;
    struct { long a; long b; } s{0, 0};
    std::thread t1([&] { for (long i = 0; i < N; ++i) ++s.a; });
    std::thread t2([&] { for (long i = 0; i < N; ++i) ++s.b; });
    t1.join(); t2.join();
    return s.a + s.b == 0;
}
```

## 9. 优化后代码

对应 padding 分离的程序。

```cpp
// good.cpp: 每计数器独立缓存行（alignas(64)）
#include <thread>

int main() {
    constexpr long N = 500'000'000;
    struct alignas(64) Counter { long val; };
    Counter c[2]{ {0}, {0} };
    std::thread t1([&] { for (long i = 0; i < N; ++i) ++c[0].val; });
    std::thread t2([&] { for (long i = 0; i < N; ++i) ++c[1].val; });
    t1.join(); t2.join();
    return c[0].val + c[1].val == 0;
}
```

> 完整实验见 src/13_false_sharing；注意 padding 大小应由 `coherency_line_size` 决定，不硬编码 64。

## 10. 为什么会更快

| 角度 | 同缓存行 | padding 分离 |
|---|---|---|
| RFO 次数 | 每写一次一次所有权转移 | 零（各写各的行） |
| 缓存行倒手 | 每操作一次 | 无 |
| 一致性协议消息 | 大量 | 无 |
| 线程扩展性 | 线程越多越慢（1147%） | 线性 |
| 可并行性 | 写被串行化 | 真并行 |

论文数据：4×P4 同缓存行最高慢 1147%（图 6.10）。

## 11. 如何验证

```bash
./build/13_false_sharing/false_sharing        # 同行 vs padding
./scripts/perf_stat.sh ./build/13_false_sharing/false_sharing
# 看线程数扫描时运行时间是否随线程数恶化
```

读取本机缓存行大小（决定 padding 用多少）：

```bash
cat /sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size
```

## 12. 实验结果应该怎么看

- 同缓存行版本：线程数从 1 增加到 4，耗时暴增（RFO 风暴）。
- padding 版本：耗时基本不随线程数恶化（带宽/调度噪声除外）。
- 若用 perf：同缓存行版的 cache-misses 或总线相关事件显著更高。
- 在共享缓存的多核上差异可能小于独立缓存的多路机器——解释时说明平台。

## 13. 常见误区

- **误区 1：两个线程改不同变量就没问题**。同缓存行就是 false sharing，照样互相拖累。
- **误区 2：padding 永远是 64**。应先读缓存行大小；不同平台可能不同。
- **误区 3：false sharing 只在 SMP 上存在**。多核共享缓存上较轻但存在；多路机器严重。
- **误区 4：所有变量都该独立缓存行**。会爆 footprint；只对"常被不同线程写"的变量这样做。
- **误区 5：TLS 是万能的**。有初始化开销、寻址更贵、DSO 内 TLS 连带分配。

## 14. 实践练习

1. 运行 src/13，对比同行/padding 在不同线程数下的耗时，画出曲线。
2. 用附录 A.3 的程序思路，自己实现一个测量缓存行共享开销的版本。
3. 讨论：为什么只读变量不需要特殊处理（S 态共享无 RFO）。
4. 用 `__thread` 改一个计数器版本，测量 TLS 的开销。
5. 解释论文图 6.11 中"多核开销不随核数增长"的原因。

## 15. 本章总结

- 多线程缓存三挑战：并发、原子性、带宽。
- False sharing：不同变量同缓存行，写触发 RFO 风暴；SMP 上最高慢 1147%。
- 处理策略：常量分组、常写变量独立缓存行、单线程变量用 TLS。
- 结构体 + alignas + padding 是分离常用手段；padding 大小按缓存行。
- 测试要在真实平台（尤其多路 SMP）上做，多核共享缓存会掩盖问题。

## 16. 对应代码

- src/13_false_sharing/（同缓存行 vs padding）
- src/14_atomic_contention/（衔接：真正共享数据用原子操作）
- src/15_thread_affinity/（绑定与线程拓扑）
