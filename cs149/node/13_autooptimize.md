# CS149 第 13 讲：领域专用编程系统与自动性能优化

**PDF**：Lecture 13 — Domain-Specific Programming Systems and Automatic Performance Optimization

---

## 本讲核心问题

1. 为什么高性能编程长期处在“性能高但生产力低”的两难中？
2. DSL 如何同时提高抽象层次与优化空间？
3. Halide 为什么被视为“算法 / 调度分离”的经典案例？
4. 自动调度器与 LLM 在性能优化中各自适合做什么？

---

## 1. 生产力与性能的长期矛盾

### 1.1 传统高性能实现的问题

直接用：

- C++
- SIMD intrinsic
- CUDA
- 汇编式内核模板

虽然能逼近硬件峰值，但代价是：

- 编程复杂
- 可维护性差
- 对具体硬件耦合强
- 优化经验难迁移

### 1.2 为什么 DSL 有机会打破僵局

领域专用语言通过限制表达范围，换来：

- 更高层的语义信息
- 更小、更结构化的优化搜索空间
- 更容易自动做融合、块化、向量化、并行化

### 1.3 关键认知

不是“越通用越好”，而是：

- 在足够重要的领域，把表达限制到合适范围，反而更能同时提升效率与开发速度。

---

## 2. Halide 的核心思想：算法与调度分离

### 2.0.1 Halide Function 与 Expression 的精确定义

- **Halide Function**: 在 N 维域上定义的无限（但离散的）值集合
- **Halide Expression**: 无副作用的表达式，描述如何用其他函数的值来计算某个域点的函数值
- Halide 是**声明式**语言——它不定义迭代顺序、不定义域中的哪些值被存储，只定义计算所需的值。域点上的迭代是**隐式的**（无显式循环）。

### 2.0.2 Halide 的工业应用

- Google 手机摄像头处理管线（HDR+、人像模式等）
- Instagram、Adobe 等公司也在工业中使用
- Google HDR+ pipeline: 超过 2000 个 Halide functions

### 2.1 算法（What）

算法描述：

- 每个输出点该由哪些输入点经过什么函数得到
- 更像数学上的定义

### 2.2 调度（How）

调度描述：

- 按什么顺序计算
- 是否 tile
- 在哪一层并行
- 在哪一层向量化
- producer 在哪里计算、存多久

### 2.3 这为什么是革命性的

传统代码里，算法和执行顺序往往耦合得很死。Halide 把它们拆开后：

- 同一个算法可尝试很多调度
- 调度搜索空间变得结构化
- 自动优化器有机会替代手工枚举

---

## 3. 图像模糊案例：从朴素实现到系统化优化

### 3.0.1 图像模糊的精确工作量分析

| 实现方案 | 工作量（数学形式） | 实际值（3×3 模糊） |
|---|---|---|
| 直接二维模糊 | N² × W × H | 9 × W × H |
| 两遍分离模糊 | 2N × W × H | 6 × W × H |
| Chunked v1 (CHUNK_SIZE=1) | — | 12 × W × H（反而更差！） |
| Chunked v2 (CHUNK_SIZE=16) | — | (34/16)×3×W×H = 6.4×W×H |

CHUNK_SIZE=1 的问题：每行都重新计算输入和临时缓冲区，完全没有数据复用。

CHUNK_SIZE=16 趋近理想值 6×W×H。

### 3.0.2 两遍模糊的缓存行为分析

- Input 数据被立即复用 3 次（"从不重复加载所需数据"）
- tmp_buf 数据被复用 3 次，但中间间隔三行（"若缓存容量达到 3 行图像"）
- 关键洞察：**两遍实现中对 tmp_buf 的 loads/stores 是额外开销——这是实现方式引入的（artifactual），并非计算本质需求。算法的本质带宽需求仅是读入每个 input 元素、写出每个 output 元素。**

### 3.1 直接二维模糊

- 每个输出像素访问一个二维邻域
- 工作量与核大小平方相关
- 中间数据复用与缓存行为可能一般

### 3.2 分离卷积（separable filter）

若滤波核可分解成横向和纵向两次一维卷积：

- 原本 `N^2` 的工作量可降到 `2N`
- 算法层面已经获得重大收益

### 3.3 两遍实现的问题

虽然工作量减少，但如果：

- 横向结果整幅图写回临时缓冲
- 再完整读回做纵向处理

就会引入大量中间流量。

### 3.4 进一步优化方向

- 分块处理
- producer-consumer 融合
- 让临时结果只在片上 / 缓存短暂停留

这说明真正的高性能不只依赖算法复杂度，也依赖调度方式。

> 对应源码：`lecture13_part1.cpp`
> 内容：单遍模糊、两遍模糊、分块 / 融合版本、缓存友好程度对比。

---

## 4. Halide 调度原语的系统意义

### 4.0.1 完整 Halide 代码示例

```cpp
Var x, y;
Func blurx, blury, bright, out;

// Algorithm (WHAT)
blurx(x,y) = (input(x-1,y) + input(x,y) + input(x+1,y)) / 3;
blury(x,y) = (blurx(x,y-1) + blurx(x,y) + blurx(x,y+1)) / 3;
bright(x,y) = blury(x,y) * 1.2f;
out(x,y) = lookup(bright(x,y));

// Schedule (HOW)
blurx.compute_at(out, xi).vectorize(x, 8);
blury.compute_at(out, xi).vectorize(x, 8);
out.tile(x, y, xi, yi, 256, 32).vectorize(xi, 8).parallel(y);
```

### 4.0.2 三种 Halide 调度策略的 loop nest 对比

**策略 1: `compute_root()`（全量计算，无融合）**
```c
// 先计算整张 blurx 图
for (int y = 0; y < H; y++)
    for (int x = 0; x < W; x++)
        blurx[y][x] = ...;
// 再用 blurx 计算 blury 整张图
for (int y = 0; y < H; y++)
    for (int x = 0; x < W; x++)
        blury[y][x] = ...;
```
→ 需要分配整张中间结果图的存储空间

**策略 2: `compute_at(out, xi)`（tile 级融合）**
```c
for (int y_tile) for (int x_tile)
    // 只为当前 tile 计算 blurx
    for (int yi) for (int xi)
        blurx_tile[yi][xi] = ...;
    for (int yi) for (int xi)
        blury_tile[yi][xi] = ...;
```
→ 中间结果仅暂存于 tile 大小的片上空间

**策略 3: `compute_at(out, x)`（行级融合）**
```c
for (int y_tile) for (int x_tile)
    for (int yi) for (int xi)
        for (int x = x_tile; x < x_tile+tile; x++)
            blurx_row[x] = ...;  // 仅为当前行计算
        for (int x = ...)
            out[...] = blury_row[x];
```

### 4.0.3 Halide 的语言约束

三个关键约束使自动调度成为可能：
1. 在**规则 N 维域**上的计算
2. 仅支持**前馈流水线**（+ 规约和固定深度递归的特殊支持）
3. 所有**依赖可由编译器推断**

### 4.0.4 Halide 初始学术成果的定量数据

- Camera RAW 处理：463 行 ARM NEON 汇编 → Halide 代码量减少 2.75 倍且快 5%
- Bilateral filter：122 行 C++ → Halide 34 行算法 + 6 行调度 → CPU 快 5.9 倍、GPU 比手写 CUDA 快 2 倍

常见原语包括：

- `tile`
- `vectorize`
- `parallel`
- `compute_root`
- `compute_at`
- `reorder`

### 4.1 `tile`

- 把二维或多维迭代空间按块切分
- 主要目的是提升局部性、适配缓存 / shared memory / SRAM

### 4.2 `vectorize`

- 显式告诉编译器某一层循环适合 SIMD 化
- 把连续元素打包处理

### 4.3 `parallel`

- 指定哪一层循环适合多线程展开
- 常用于较粗粒度维度，如 tile 行、tile 列或批次维度

### 4.4 `compute_root` vs `compute_at`

- `compute_root`：先整块算完 producer，再让 consumer 用
- `compute_at`：在 consumer 的某一层循环中按需生成 producer

这两者直接决定：

- 中间结果生命周期
- 中间缓冲大小
- producer-consumer 复用关系
- 片上 / 缓存可容纳程度

> 对应源码：`lecture13_part2.cpp`
> 内容：Halide 风格 DAG、tile/vectorize/parallel/reorder、`compute_root` 与 `compute_at` 的模拟。

---

## 5. 自动调度器为什么可行

### 5.1 手工调度的难点

即使算法固定，调度空间也可能巨大：

- tile 大小怎么选
- 哪层并行
- 哪层向量化
- producer 放在哪个循环层级计算
- 是否融合、是否重排

### 5.2 Halide 的独特优势

由于调度原语是结构化的，系统可以：

- 把调度看成一系列有限选择
- 构建搜索空间
- 用 cost model 估计候选方案

### 5.3 自动化的本质

这不是“AI 魔法”，而是：

- 好的中间表示
- 好的搜索空间设计
- 好的代价模型

共同作用的结果。

### 5.0.1 自动调度器的现实动机

Google 有 80+ 程序员写 Halide，但**只有极少数人**被信任写调度（schedule）。编写高效调度是极其稀缺的技能——这正是自动调度器的驱动力。

### 5.0.2 调度搜索的贪婪算法

```
For each node N in the program DAG (从 DAG 末端开始):
    选择当前节点 N 的放置位置 (N.compute_at())
    选择 tile 尺寸
Repeat until entire DAG is scheduled
```

搜索方法：Greedy search → Beam search

### 5.0.3 AI 代价模型的具体实现

- 简单的 MLP，在数微秒内评估一个 schedule
- 在 166 秒内测试了 140 万个 schedule
- 输出 27 个系数，插入手工打造的代价模型
- 基于大量随机生成的 Halide 程序训练

### 5.0.4 自动调度器 vs 人类专家

在 Max filter、Non-local means denoising、Lens blur 等任务上，自动调度器可匹敌甚至超越人类 Halide 专家。部分结果来自 Mullapudi 2016，部分来自 Adams 2019。

---

## 6. 机器学习代价模型与搜索

### 6.1 为什么需要 cost model

真实测量每个候选 schedule 太慢，所以需要：

- 快速预测性能的模型
- 用于筛选最有希望的候选

### 6.2 成功的关键不只是模型本身

更关键的是：

- 表示是否足够结构化
- 特征是否反映局部性、并行度、向量化、工作量
- 搜索是否覆盖高质量区域

### 6.3 一个重要启示

自动优化之所以在某些 DSL 中效果特别好，不是因为模型神奇，而是因为：

- 语言设计本身就让优化问题“更容易被搜索”。

---

## 7. LLM 与自动优化：应该如何分工

### 7.1 LLM 擅长的部分

- 代码生成
- 解释高层优化思路
- 结合历史案例做启发式迁移
- 在 DSL 层面尝试候选 schedule 或 kernel 结构

### 7.2 LLM 不擅长单独完成的部分

- 纯靠文字直觉准确判断低层性能细节
- 在巨大连续参数空间中稳定找到最优值
- 直接生成大规模低级优化代码且一次命中最佳实现

### 7.3 更合理的组合方式

- DSL 提供高层结构化表示
- 自动调度器 / 代价模型提供系统搜索
- LLM 提供候选生成、规则总结、经验迁移和交互式优化

这也是现代性能工程越来越现实的方向。

### 7.0.1 LLM 反思式优化的完整流程

**第一步 prompt**：生成 CUDA 代码 → 执行 → 收集 profiling 数据（SM util 42%、DRAM util 89%、L2 cache hit rate 68%）

**第二步 prompt**：将 profiling 数据反馈给 LLM 进行反思和改进 → 重新生成代码 → 再次 profiling

### 7.0.2 KernelBench 基准

一个数百个 PyTorch kernel 的基准测试，目标是让 LLM agent 自动生成正确且快速的 CUDA kernels。

### 7.0.3 LLM 自我改进的四种方向

1. **基于经验微调 LLM**（fine-tune from optimization history）
2. **"实践问题"数据库**——存储从决策序列到结果的完整优化轨迹，"不只是存储解决方案，而是存储优化决策序列"
3. **基于经验的 prompt 优化**——"Prompt optimizer: 检查优化循环的轨迹，尝试总结为重要事实和原则"
4. **Halide 穷举搜索 + LLM agent 结合**——"优化成本极高，但能得到一些最好的结果"

### 7.0.4 DSL 对 LLM 代码生成的辅助

用 DSL（如 Triton、CUTLASS/CuTe、TileLang）让 LLM 组装高层原语而不是直接写 CUDA——减少正确性错误和幻觉。但要注意："DNNs 在较少使用的语言上写出正确代码仍有困难"。

### 7.0.5 本讲的未解之辩

- 公司每年在 AI 计算上花费数千万到数亿美元
- 核心辩论：真正使系统成功的价值来自 DSL 设计，还是 LLM agent？

---

## 8. 本讲最重要的方法论

真正可持续的高性能系统往往依赖三层设计同时成立：

1. **好表示**：程序语义足够清晰，暴露优化空间。
2. **好搜索空间**：合法调度组合是结构化的，不是杂乱无章的。 
3. **好优化器**：可以自动探索并评估候选方案。

Halide 成功的根本，就在于它把这三件事做成了一个闭环。

---

## 常见误区

1. **误区：DSL 会损失性能。**
   若 DSL 暴露了更多结构信息，反而可能比手写通用代码更容易达到高性能。
2. **误区：自动调度就是穷举。**
   真正有效的是结构化搜索 + 代价模型。
3. **误区：LLM 会直接取代性能工程。**
   更现实的是 LLM 与 DSL / profiler / auto-scheduler 协同。
4. **误区：优化主要发生在“写代码的人”脑子里。**
   表示与系统设计同样决定了可优化上限。

---

## 对应源码

| 文件 | 主题 | 重点 |
|---|---|---|
| `lecture13_part1.cpp` | 图像模糊演化 | 分离卷积、分块、融合、临时缓冲生命周期 |
| `lecture13_part2.cpp` | Halide 调度模拟 | tile、vectorize、parallel、compute_at/root |

---

## 学完本讲应做到

- 能解释为什么算法与调度分离是系统设计上的巨大进步。
- 能说明 Halide 调度原语各自在优化什么。
- 能理解自动调度器之所以有效，离不开好的中间表示。
- 能对 LLM、DSL 和 auto-scheduler 的合理分工形成清晰认识。

