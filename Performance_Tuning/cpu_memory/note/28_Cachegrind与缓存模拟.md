# 28 Cachegrind 与缓存模拟

> 对应 PDF：第 7.2 节 Simulating CPU Caches（PDFp81~82）、第 7.3 节 Measuring Memory Usage（PDFp82~84）、图 7.5~7.7
> 本篇回答：Cachegrind 怎么模拟缓存？输出怎么读？改缓存参数能看什么？Massif 怎么用？模拟与真实硬件的差异？

## 1. 本章要解决的问题

- Cachegrind 模拟哪些缓存（L1i/L1d/L2），怎么指定参数？
- 汇总输出与 cg_annotate 按函数/行怎么读？
- 模拟器的局限（LRU、忽略上下文切换/系统调用）。
- Massif/memusage 测内存占用与分配模式。
- 分配模式（大量小分配/malloc 头）如何影响缓存与预取。

## 2. 前置知识

- note/26：工具总览、OProfile/perf、CPI。
- note/05：缓存层级、行大小、关联度。
- valgrind 框架概念。

## 3. 核心概念

- **Cachegrind**：valgrind 的缓存模拟工具。
- **cg_annotate**：展示 Cachegrind 结果的工具。
- **I1 / D1 / L2**：模拟的一级指令、一级数据、二级缓存。
- **Ir / Dr / Dw**：指令读、数据读、数据写引用数。
- **I1mr / D1mr / D1mw / L2mr / L2mw**：各缓存 miss 计数。
- **Massif**：堆内存随时间/调用点的模拟工具。
- **memusage**：glibc 轻量内存测量（真实运行）。
- **LRU（Least Recently Used）**：Cachegrind 模拟的逐出策略。

## 4. 硬件工作流程

### 4.1 Cachegrind 用法

```bash
valgrind --tool=cachegrind command arg             # 用宿主 CPU 缓存参数模拟
valgrind --tool=cachegrind --L2=8388608,8,64 cmd   # 自定义 L2: 大小,关联度,行大小
cg_annotate cachegrind.out.PID                     # 按函数查看
```

- 拦截所有内存访问，模拟 L1i/L1d/L2 缓存。
- 输出 cachegrind.out.PID 文件，cg_annotate 生成可读报告。

### 4.2 汇总输出（图 7.5，PDFp81）

```text
I refs: 152,653,497     I1 misses: 25,833    L2i misses: 2,475
D refs: 56,857,129 (35,838,721 rd + 21,018,408 wr)
D1 misses: 14,187       L2d misses: 7,701
L2 refs: 40,020         L2 misses: 10,176
```

- 总指令/数据引用、各级 miss 数与 miss rate。
- L2 访问可分指令/数据；数据缓存分读/写。

### 4.3 cg_annotate 按函数（图 7.6，PDFp82）

```text
Ir   I1mr I2mr  Dr     D1mr D2mr  Dw     D1mw D2mw  file:function
53M  9    8     9.5M   13   3     5.8M   14   0     ???:_IO_file_xsputn
36M  6267 114   11.2M  74   18    7.1M   22   0     ???:vfprintf
...
```

- 先看 L2 miss（I2mr/D2mr），再 L1。
- 需要调试信息（否则符号表不完整）。
- 可给源码文件 → 逐行标注 miss。

### 4.4 局限（PDFp81~82）

- 模拟 LRU 逐出：真实大关联度缓存未必用 LRU。
- 不考虑上下文切换/系统调用（会刷 L1、破坏 L2）→ 模拟 miss 低于实际。

### 4.5 Massif / memusage（PDFp82~84）

```bash
valgrind --tool=massif command arg                  # 堆占用随时间/调用点
memusage -p out.png command arg                     # glibc 轻量版，生成 PNG
```

- massif：记录分配调用点 + 时间戳 + 大小（图 7.7），输出 massif.PID.txt/.ps。
- memusage：真实运行、快；可输出分配大小直方图；`-n NAME` 指定被观测程序。
- 分配模式：同一位置大量小分配 + 平缓斜率 → 用 obstack/批量分配。

## 5. PDF 核心观点

> 来源：PDF 第 81~84 页；对应章节 7.2、7.3、图 7.5~7.7。以下为概括。

1. **Cachegrind 拦截所有内存访问并模拟 L1i/L1d/L2**（PDFp81）：缓存大小/行大小/关联度可由 --I1/--D1/--L2 指定。
2. **汇总输出**（PDFp81，图 7.5）：总引用、各级 miss 与 miss rate，可拆分指令/数据、读/写。
3. **cg_annotate 按函数/行**（PDFp82，图 7.6）：Ir/Dr/Dw 是总访问，后两列是 miss；先 L2 后 L1。
4. **模拟局限**（PDFp81~82）：LRU 假设、忽略上下文切换/系统调用 → 实际 miss 可能更高；适合学习内存行为。
5. **Massif**（PDFp82，图 7.7）：按分配调用点 + 时间追踪堆占用；栈占用可选（--stacks=no）；`--alloc-fn` 指定自定义分配函数。
6. **memusage**（PDFp83）：真实运行（非模拟）、快；分配大小直方图；`-n NAME` 指定程序。
7. **分配模式影响性能**（PDFp83~84）：链表逐节点分配不保证顺序布局；malloc 头/填充可占 50% 并砍半预取率。
8. **识别信号**（PDFp83）：同一位置大量小分配 + 平缓上升 → 该用 obstack/批量分配。

## 6. 通俗解释

Cachegrind 是一个**虚拟跑步机 + 模拟沙盘**：

> 你的程序在跑步机上跑（被 valgrind 模拟执行），旁边有个"缓存仿真器"记录每一步的缓存行为。
> 你可以改沙盘参数（把 L2 从 2MB 换 8MB、8 路换 4 路）看"如果硬件这样，会 miss 多少"——
> 这在真实机器上是做不到的（你手头没有各种缓存大小的 CPU）。

但它也有失真：

> 仿真器假设"最久没用的先滚蛋"（LRU），真实硬件未必如此；而且它不模拟"程序被切走/系统调用"
> 造成的缓存清空——所以它报的 miss 数通常会比真实低。

Massif 是**记账软件**：

> 记每笔堆内存从哪个调用点、什么时间、分配了多少。图里"同一个地方反复冒出小斜坡"，
> 就是大量小分配，该合并了（obstack）。

## 7. 示例分析

### 7.1 自定义缓存参数

```bash
# 模拟 8MB、8 路、64B 行的 L2
valgrind --tool=cachegrind --L2=8388608,8,64 ./program
```

- 用途：研究"如果缓存更大/关联度更高，miss 会降多少"（论文图 3.8 的数据来源）。
- 注意 --L2 必须出现在程序名之前。

### 7.2 读 cg_annotate

- 找 `I2mr`/`D2mr` 大的函数 → 代码/数据在 L2 层 miss 多。
- 给源码文件后逐行看 `D1mr` → 精确定位问题行。
- 优先优化 L2 miss（代价大），再 L1。

### 7.3 分配模式识别

- massif 图：某地址区间从 ~800ms 涨到 1800ms、斜率平缓 → 大量小分配（论文示例 0x4c0e7d5）。
- 解法：obstack 或自建池，把连续小对象分配进同一大块。

## 8. 未优化代码

对应"大量小分配 + 链表"的程序。

```cpp
// bad.cpp: 每节点单独 malloc
#include <cstdlib>

struct Node { int data; Node* next; };

int main() {
    constexpr int N = 1 << 20;
    Node* head = nullptr;
    for (int i = 0; i < N; ++i) {
        Node* n = (Node*)malloc(sizeof(Node));
        n->data = i;
        n->next = head;
        head = n;
    }
    long long sum = 0;
    for (Node* p = head; p; p = p->next) sum += p->data;
    return sum == 0;
}
```

## 9. 优化后代码

对应"连续节点池"的程序。

```cpp
// good.cpp: 连续节点池
#include <vector>

struct Node { int data; Node* next; };

int main() {
    constexpr int N = 1 << 20;
    std::vector<Node> pool(N);
    for (int i = 0; i < N; ++i) {
        pool[i].data = i;
        pool[i].next = (i + 1 < N) ? &pool[i + 1] : nullptr;
    }
    long long sum = 0;
    for (const Node* p = &pool[0]; p; p = p->next) sum += p->data;
    return sum == 0;
}
```

## 10. 为什么会更快

| 角度 | 单独 malloc | 节点池 |
|---|---|---|
| D1 miss | 高（布局乱） | 低（连续） |
| 预取 | 失效 | 有效 |
| malloc 头/填充 | 每节点 16B+ | 无 |
| TLB | 多页 | 少页 |

Cachegrind 会给出模拟 miss 数；真实 perf 验证；两者要区分（模拟可能偏低）。

## 11. 如何验证

```bash
./scripts/cachegrind.sh ./build/12_pointer_chasing/pointer_chasing
valgrind --tool=cachegrind --L2=8388608,8,64 ./build/12_pointer_chasing/pointer_chasing
cg_annotate cachegrind.out.$(pgrep -f pointer_chasing | head -1)

valgrind --tool=massif ./build/12_pointer_chasing/pointer_chasing
memusage -p /tmp/mem.png ./build/12_pointer_chasing/pointer_chasing
```

## 12. 实验结果应该怎么看

- 对比两版代码的 D1 miss 率与 L2 miss 率。
- 用 --L2 改参数观察 miss 变化（容量/关联度敏感度）。
- 用真实 perf 交叉验证（Cachegrind 是模拟，可能偏低）。
- massif 图识别分配模式。

## 13. 常见误区

- **误区 1：Cachegrind miss = 真实 miss**。LRU 假设 + 忽略上下文切换/系统调用 → 偏低。
- **误区 2：改 --L2 参数能模拟一切硬件**。只模拟缓存容量/关联度/行大小，不含硬件预取等逻辑。
- **误区 3：cg_annotate 不需要调试信息**。没有调试信息则函数符号不全。
- **误区 4：memusage 是模拟**。它真实运行，比 massif 快。
- **误区 5：分配头开销无关紧要**。可占 50%，砍半预取率。

## 14. 实践练习

1. 用 Cachegrind 跑"单独 malloc"与"节点池"，对比 D1 miss 率。
2. 用 --L2 改缓存大小，画出 miss 随容量的变化（复现论文图 3.8 思路）。
3. 用 cg_annotate 定位一个函数的热点行（需调试信息）。
4. 用 massif 与 memusage 对比同一程序，说明两者差异。
5. 解释为什么 Cachegrind 报的 miss 通常低于真实硬件。

## 15. 本章总结

- Cachegrind 用 valgrind 框架模拟 L1i/L1d/L2，可改参数、可逐函数/行。
- 汇总输出与 cg_annotate 是主要查看手段。
- 模拟局限：LRU、忽略上下文切换/系统调用 → 实际 miss 更高。
- Massif/memusage 测内存占用与分配模式。
- 大量小分配与 malloc 头会破坏局部性与预取；obstack/批量分配是解法。

## 16. 对应代码

- src/12_pointer_chasing/（链表 vs 节点池，Cachegrind 分析对象）
- scripts/cachegrind.sh（封装 valgrind --tool=cachegrind）
