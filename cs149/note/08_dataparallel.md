# CS149 第 8 讲：数据并行思维

**课程**：Stanford CS149，2025 年秋季

---

## 本讲核心问题

1. 为什么很多不规则并行问题可以重写成规则的数据并行问题？
2. map、reduce、scan 为什么被称为“数据并行三大基石”？
3. 为什么 scan 在并行算法中比初学者想象得重要得多？
4. gather/scatter、排序、分段扫描如何把复杂结构转为规则流水？

---

## 1. 数据并行模型的核心思想

### 1.1 从“工作线程要做什么”转向“序列要经历什么变换”

数据并行不是先描述线程行为，而是先描述：

- 一个序列如何变成另一个序列
- 一组元素如何被映射、过滤、归约、重排

### 1.2 为什么这种视角很强

因为它：

- 更接近高层算法语义
- 更利于编译器 / 运行时统一优化
- 更容易映射到 SIMD、GPU、向量库、并行 DSL

### 1.3 数据并行与任务并行的差别

- 任务并行强调不同任务可做不同工作。
- 数据并行强调很多元素执行同一种变换。

二者并不冲突，但数据并行特别适合大规模规则计算。

### 1.0.1 序列作为关键数据类型

多语言中的序列抽象：
- Scala: `List[T]`
- Python Pandas: DataFrames
- PyTorch/JAX: Tensors
- Haskell: `seq T`

关键区别：程序通过特定操作（而非直接元素访问）来操作序列。这使编译器/runtime 有更大优化空间。

### 1.0.2 Map 的函数签名与多语言实现

- Haskell: `map :: (a -> b) -> seq a -> seq b`
- C++: `std::transform(begin, end, out_begin, func)`
- JAX: `vmap(func)(data)`

---

## 2. Map：最直接的数据并行原语

### 2.1 定义

`map` 表示把一个无副作用函数应用到所有元素上：

```text
map f [x0, x1, x2, ...] -> [f(x0), f(x1), f(x2), ...]
```

### 2.2 为什么它天然并行

- 每个输出元素只依赖对应输入元素
- 元素之间没有写冲突
- 没有跨元素数据依赖

### 2.3 工程意义

一旦问题能写成 map：

- SIMD / GPU / 多线程都很容易利用
- 性能瓶颈很可能转向带宽和局部性

> 对应源码：`lecture8_part1.cpp`
> 内容：map、reduce、filter、基于 map + sort 的 histogram。

---

## 3. Reduce / Fold：从很多值合成一个值

### 3.1 定义

reduce 的目标是把序列压缩为一个结果，例如求和、求最大值、逻辑与。

### 3.2 为什么 reduce 不像 map 那么简单

因为它引入了跨元素组合。

并行 reduce 通常要求：

- 操作是**结合的（associative）**
- 最好还具有单位元（identity）

### 3.3 并行实现直觉

- 先让每个线程 / block / lane 做局部归约
- 再把局部结果做更高一层归约

### 3.4 这解释了很多性能现象

- reduce 常会出现树形同步结构
- 最终汇总步骤可能形成串行尾巴
- 局部归约能显著减少共享写冲突

### 3.0.1 Fold 的精确函数签名

`fold :: b -> ((b, a) -> b) -> seq a -> b`

- 初始值 `b` 必须是单位元
- 并行 fold 需要 combiner 函数 `comb :: (b, b) -> b`
- 若 `f :: (b, b) -> b` 本身是结合的二元操作，则不需要额外的 combiner

---

## 4. Scan：并行算法中的“隐藏主角”

### 4.1 什么是 scan

scan 又叫前缀和 / prefix sum。它输出的是累积结果序列，而不是单个标量。

例如：

```text
inclusive scan(+): [3,8,4,6] -> [3,11,15,21]
exclusive scan(+): [3,8,4,6] -> [0,3,11,15]
```

### 4.2 为什么 scan 如此重要

因为它是很多数据重排算法的基础：

- 过滤后的写出位置分配
- 压缩存储
- stream compaction
- radix sort
- 直方图边界计算
- 稀疏结构索引生成

### 4.3 并行 scan 的两个目标

- **span 小**：层数少，易并行
- **work-efficient**：总工作量不要比顺序多太多

### 4.0.1 Scan 的精确数学定义

- `scan_inclusive(⊕, A) = [a0, a0⊕a1, a0⊕a1⊕a2, ...]`
- `scan_exclusive(⊕, A) = [I, a0, a0⊕a1, a0⊕a1⊕a2, ...]`（I 是 ⊕ 的单位元）
- 当 ⊕ = + 时，inclusive scan 即为 prefix sum

### 4.0.2 Blelloch Work-Efficient Scan 伪代码

**Up-sweep 阶段**（构建部分和树）：
```
for d = 0 to (lg N) - 1:
  forall k = 0 to N-1 by 2^(d+1):
    a[k + 2^(d+1) - 1] = a[k + 2^d - 1] + a[k + 2^(d+1) - 1]
```

**Down-sweep 阶段**（分发部分和）：
```
a[N-1] = 0
for d = (lg N)-1 down to 0:
  forall k = 0 to N-1 by 2^(d+1):
    t = a[k + 2^d - 1]
    a[k + 2^d - 1] = a[k + 2^(d+1) - 1]
    a[k + 2^(d+1) - 1] = t + a[k + 2^(d+1) - 1]
```
Work = O(N), Span = O(log N)

### 4.0.3 两核 Scan 实现

每个核对一半数据做顺序 scan，然后再加 base (a0-7)。Work 仍为 O(N)，但常数仅为 1.5。需要考虑 NUMA 访问成本。

### 4.0.4 Warp-Level Scan 实现

```cuda
for (int i = 1; i < 32; i *= 2) {
    int n = __shfl_up_sync(0xFFFFFFFF, val, i);
    if (threadIdx.x % 32 >= i) val += n;
}
```
5 步循环（2^5=32），Work = N log N。在 SIMD 上下文中，work-efficient scan 反而更差——需要超过 2x 指令数，导致更低的 SIMD 利用率。

> 对应源码：`lecture8_part2.cpp`
> 内容：顺序 scan、朴素并行 scan、Blelloch work-efficient scan、SIMD 风格 scan、多核 scan。

---

## 5. 朴素 scan 与工作高效 scan

### 5.1 朴素并行 scan

- 每轮把步长翻倍
- span 是 `O(log N)`
- 但总工作量往往是 `O(N log N)`

### 5.2 Blelloch 扫描

通过两阶段：

1. **up-sweep / reduce phase**
2. **down-sweep phase**

可实现：

- work `O(N)`
- span `O(log N)`

### 5.3 为什么课程强调“不是所有层级都该用同一种 scan”

- 在很小的 SIMD / warp 范围内，朴素方法虽然工作量大一点，但控制更简单、利用率更高。
- 在大数组或 block 级别，work-efficient 版本更合适。

这体现了现代并行算法设计中的重要原则：

- **不同硬件层级用不同算法，整体最优。**

### 5.0.1 Block-Level Scan 完整实现

4 个步骤：
1. 每个 warp 做自己的 scan
2. 每个 warp 的 lane 31 写 partial result 到 shared memory
3. warp 0 在 shared memory 上做 base scan
4. 所有 warps 加 base

超出单 block 能力时需 3 个 kernel launch：
- Kernel 1: Block 0 scan
- Kernel 2: Block 1..N-1 Add
- Kernel 3: Block 0 Add

层次化实现匹配存储体系。

---

## 6. 分段扫描（Segmented Scan）

### 6.1 它解决什么问题

有时输入并不是一个整体序列，而是很多连续小段拼在一起。

分段扫描允许：

- 在每个 segment 内独立做 scan
- 但仍用一套规则并行框架处理整个数组

### 6.2 为什么它重要

它能把很多“不规则的列表的列表”问题转成规则数组问题：

- CSR 稀疏矩阵
- 可变长度邻接表
- 分组聚合
- 排序后按 key 聚合

### 6.3 本质意义

它是在规则数组之上恢复局部边界语义。

### 6.0.1 Segmented Scan 伪代码

Up-sweep 阶段额外传播 `flag`（OR 操作），Down-sweep 阶段维护 `flag_original` 副本以处理段边界重置。这使每个 segment 内部独立做 scan，但不破坏全局并行框架。

> 对应源码：`lecture8_part3.cpp`
> 内容：segmented scan、gather、scatter、基于排序与扫描的散射思路、稀疏矩阵乘法示例。

---

## 7. Gather 与 Scatter

### 7.1 Gather

```text
output[i] = input[index[i]]
```

特点：

- 是并行读
- 每个线程只写自己的输出位置
- 通常不存在写冲突

### 7.2 Scatter

```text
output[index[i]] = input[i]
```

特点：

- 是并行写
- 若多个 `i` 指向同一目标位置，就会冲突

### 7.3 为什么 scatter 更难

- 需要同步或冲突解决
- 可能需要原子操作
- 在 GPU 上常带来严重随机写与串行化

### 7.4 常见替代思路

- sort + segmented scan
- 先统计位置，再前缀和，再稳定写出

这就是把不规则写冲突转成规则多阶段管线的典型例子。

### 7.0.1 Gather/Scatter 的硬件指令历史

- AVX2（2013）：支持 gather，但不支持 SIMD scatter
- AVX512：有 scatter 指令
- GPU：有硬件 gather/scatter，但仍比连续 load/store 昂贵

---

## 8. 排序为何是数据并行中的万能锤子之一

很多看似需要锁或链表的数据组织问题，一旦转换为：

1. 生成 key
2. 按 key 排序
3. 找边界 / 聚合 / 扫描

就可能变成：

- 高度规则
- 易于 SIMD / GPU 实现
- 避免细粒度同步

### 8.1 典型案例：粒子分桶 / 网格构建

原本可能的写法：

- 每个粒子找到所属格子
- 直接 append 到共享链表
- 需要细粒度锁或原子操作

数据并行写法：

- map 出 cell id
- sort `(cell_id, particle_id)`
- 找每个 cell 的起止位置

### 8.2 为什么这经常更快

虽然多了几轮排序和扫描，但：

- 消除了高冲突同步
- 暴露出大规模规则并行
- 让硬件更容易吃满吞吐

### 8.0.1 稀疏矩阵乘法的数据并行方法

使用 CSR 格式（values, cols, row_starts）：
1. gather `x[cols[i]]`
2. map `values[i] * gathered[i]`
3. 从 row_starts 创建 flags vector
4. inclusive segmented-scan on (products, flags)
5. 取每段最后元素

### 8.0.2 Scatter 转 Sort 的特例

当 index 唯一且涵盖所有引用元素时，scatter = 一次 `sort input by index`（置换）。

非唯一 index 的 scatterOp：通过 sort + find starts + segmented scan 实现原子操作效果。

### 8.0.3 Group-by、Filter、Sort

- `groupBy :: Seq (key, T) -> Seq (key, Seq T)`
- Filter 可视为 map 后 compact（用 scan 确定输出位置）

### 8.0.4 粒子网格的 5 种方案

| 方案 | 方法 | 问题 |
|---|---|---|
| 1 | 全局锁 | 大量争用 |
| 2 | Per-cell 锁 | ~16x 减少争用 |
| 3 | Per-cell 并行 | 并行度不足 + 低效 |
| 4 | 部分结果 + 合并 | 需额外内存 |
| 5 | 数据并行（map→sort→find starts）| 消除细粒度同步但增加带宽 |

### 8.0.5 数据并行直方图

两阶段：`compute_bin`(map) + `sort` + `find_starts` + `bin_sizes`。`bin_sizes` 需处理空 bin 的边界情况。

### 8.0.6 具体数据并行系统

CUDA Thrust、Pandas Dataframe、JAX、Apache Spark/Hadoop

---

## 9. 稀疏矩阵和不规则并行也能被“规则化”

第 8 讲一个非常重要的思想是：

- 并行问题是否规则，不一定是问题本身决定的
- 往往取决于你选择的表示与计算顺序

像稀疏矩阵乘法、图算法、直方图、粒子分桶这类问题，都可以通过：

- 索引数组
- 排序
- gather/scatter
- segmented scan

转写为规则的数据并行流水。

这对后续 GPU、AI 编译器、DSL 设计都非常关键。

---

## 常见误区

1. **误区：数据并行只适用于规则数组运算。**
   许多不规则问题都能通过重表示转成数据并行。
2. **误区：scan 只是求前缀和的小技巧。**
   它是大量并行重排和压缩算法的核心积木。
3. **误区：sort 太贵，所以不该用。**
   在高冲突同步场景下，sort 往往比锁和原子更划算。
4. **误区：scatter 和 gather 本质一样。**
   gather 主要是并行读，scatter 主要是并行写，难度差别很大。

---

## 对应源码

| 文件 | 主题 | 重点 |
|---|---|---|
| `lecture8_part1.cpp` | map / reduce / filter | 数据并行基础原语 |
| `lecture8_part2.cpp` | scan 算法 | `O(N log N)` 与 `O(N)` work 的差别 |
| `lecture8_part3.cpp` | segmented scan / gather / scatter | 不规则问题规则化 |

---

## 学完本讲应做到

- 能用 map、reduce、scan 的视角重述一个并行问题。
- 能解释为什么 scan 是很多高级并行算法的基础设施。
- 能理解 sort + scan 为什么常用于替代细粒度同步。
- 能把“不规则并行”视为“表示选择问题”，而不是天生不可并行。

