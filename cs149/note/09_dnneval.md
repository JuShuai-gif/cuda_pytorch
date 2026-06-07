# CS149 第 9 讲：高效执行深度神经网络

**课程**：Stanford CS149，2025 年秋季

---

## 本讲核心问题

1. 为什么 DNN 推理与训练看起来“算很多”，却常常仍然受限于数据移动？
2. GEMM 为什么成为几乎所有深度学习系统的核心内核？
3. 卷积、融合、块化与 Flash Attention 之间有什么共同方法论？
4. 如何用算术强度和 roofline 视角分析 DNN kernel？

---

## 1. 把神经网络看成计算图与线性代数组合

### 1.1 神经元与层的基本抽象

一个神经元通常执行：

- 加权求和
- 加偏置
- 过激活函数

很多层都能视为：

- 大规模矩阵乘法
- 逐元素运算
- 局部邻域聚合
- 归一化与归约

### 1.2 典型层的计算形态

- **全连接层**：矩阵乘法 + 激活
- **卷积层**：局部窗口滑动 + 权重共享
- **池化层**：局部归约
- **Softmax**：归约 + 指数 + 归一化
- **Attention**：两个矩阵乘法 + 归一化 + 再乘法

### 1.3 课程要你看到的本质

DNN 并不是一种全新计算宇宙，而是：

- 经典并行模式在现代规模上的极致体现
- 算力、带宽、局部性、融合、块化共同作用的代表场景

### 1.0.1 Inception/GoogLeNet 架构

经典的多分支网络架构，包含 DepthConcat、多种卷积核（1×1, 3×3, 5×5, 7×7）、MaxPool、AveragePool、FC、Softmax 等组件。Inception-ResNet-A 模块使用残差设计（多个分支融合后接 ReLU）。

### 1.0.2 MobileNet 架构

使用 Depthwise Separable Conv（Conv dw + Conv 1×1）交替模式，从 224×224×3 输入逐步降采样到 1×1×1000 分类输出。不同层次维度差异很大，给库实现者带来巨大挑战。

### 1.0.3 神经元的精确定义

一个有 n 个输入、n+1 个参数（weights + bias）的单元：
$$output = f(\sum_{i=1}^{n} x_i \cdot w_i + b)$$

- ReLU: $f(x) = \max(0, x)$
- Sigmoid: $f(x) = \frac{1}{1+e^{-x}}$（可作为二分类器）

### 1.0.4 全连接层 vs 稀疏（局部）连接层

- **全连接层**：每个输出连接到所有输入
- **局部连接层**：每个输出仅连接少量相邻输入
- 卷积层 = 局部连接 + 权重共享（同一层所有单元使用相同参数）

---

## 2. Roofline

### 2.1 算术强度的定义

$$
Arithmetic\ Intensity = rac{FLOPs}{Bytes\ Transferred}
$$

### 2.2 两种受限区间

- **带宽受限**：每搬一点数据只做很少计算，性能上限由内存带宽决定。
- **算力受限**：数据复用高，能让算子长期忙碌，性能上限由峰值 FLOPS 决定。

### 2.3 为什么硬件越强，程序越容易显得带宽受限

因为：

- 峰值 FLOPS 提升速度通常比片外带宽更快。
- 同一个低算术强度 kernel，在更强机器上更容易“喂不饱算力”。

### 2.4 一个关键结论

很多 DNN 优化的核心目的，其实不是减少 FLOPs，而是：

- 提高算术强度
- 增加片上复用
- 减少中间结果回写

---

## 3. 循环融合：提高算术强度的直接手段

### 3.1 为什么融合有效

如果多个操作分开执行：

1. 中间结果写回 DRAM
2. 下一个算子再读回中间结果

这会产生大量额外流量。

### 3.2 典型例子

把：

- `tmp = A + B`
- `tmp2 = tmp * C`
- `E = tmp2 + D`

融合成一个循环后：

- 减少中间数组写回与读回
- 相同输入数据一次加载后做更多运算
- 算术强度显著提升

### 3.3 DNN 中的意义

- Conv + Bias + ReLU
- Conv + Scale + Residual Add
- Attention 中多个阶段的块内融合

这些本质上都是“尽量在数据还在片上时多做一点事”。

---

## 4. GEMM：深度学习系统的核心内核

### 4.1 为什么 GEMM 如此重要

矩阵乘法是：

- 全连接层核心
- 卷积常见 lower / implicit 展开后的核心
- Attention 中 `QK^T` 与 `PV` 的核心
- 许多归一化和投影模块背后的基础积木

### 4.2 朴素三重循环为什么不够

朴素 GEMM 的问题通常不在公式，而在数据访问顺序：

- A、B、C 的复用没有被充分利用
- 缓存命中率差
- SIMD 与寄存器复用不充分

### 4.3 多层次块化（hierarchical blocking）

现代高性能 GEMM 常按多个层级组织：

- L3 级块
- L2 级块
- L1 级块
- Register tile / micro-kernel

核心思想是：

- 让一个被加载到更快存储层的数据块，在被驱逐前尽可能多次参与计算。

### 4.4 SIMD / 向量化的多种组织方式

不同实现可能选择：

- 广播 A、向量读取 B
- 转置 B 以利于连续访问
- 用外积式微内核提升寄存器复用

> 对应源码：`lecture9_part1.cpp`
> 内容：朴素 GEMM、单层块化、分层块化、面向 SIMD 的组织方式、微内核思路。

---

## 5. 卷积实现：从直接形式到 GEMM 化

### 5.1 直接卷积

直接实现卷积时：

- 外层遍历输出位置
- 内层遍历卷积核窗口与通道

优点：

- 逻辑直观
- 不需要额外重排大矩阵

缺点：

- 实现高性能较难
- 对缓存与向量化友好性依赖具体布局

### 5.2 im2col / 卷积转 GEMM

思路：

- 把输入滑窗展开为矩阵
- 把卷积转成一次或若干次大矩阵乘法

优点：

- 直接复用成熟高性能 GEMM 库
- 算法路径统一

缺点：

- 需要额外存储展开矩阵
- 会增加 DRAM 流量

### 5.3 implicit GEMM

现代库更常做 implicit GEMM：

- 不在片外显式构造整个 im2col 矩阵
- 只在片上按块构造需要的子块
- 结合共享内存 / SRAM / Tensor Core 做高效乘法

这实际上兼顾了：

- GEMM 的成熟高吞吐执行方式
- 避免 im2col 的巨大中间存储开销

### 5.0.1 直接卷积的 7 层嵌套循环

```c
for (int n = 0; n < BATCH_SIZE; n++)
for (int h = 0; h < OUTPUT_H; h++)
for (int w = 0; w < OUTPUT_W; w++)
for (int k = 0; k < NUM_FILTERS; k++)
for (int c = 0; c < INPUT_DEPTH; c++)
for (int r = 0; r < FILTER_H; r++)
for (int s = 0; s < FILTER_W; s++)
    output[n][h][w][k] += input[n][h+r][w+s][c] * weights[k][c][r][s];
```

复用关系：
- filter weights 被所有输出位置复用
- input values 被多个 filter 跨 filter 维度复用

### 5.0.2 im2col 矩阵构造细节

3×3 卷积展开为 9 列矩阵，需要 0-padding 处理边界。存储开销为 O(K²)（对 K×K 核）。多通道时每通道独立展开再拼接。

### 5.0.3 GEMM 的 NVIDIA 符号约定

- R×S = 滤波器空间支持（卷积核大小）
- C = 输入通道数
- K = 滤波器数量（输出通道数）
- N = batch size
- P×Q = 输出空间大小

### 5.0.4 Implicit GEMM 优化

不显式构造完整 im2col 矩阵，只在 GPU 片上 shared memory 内按块物化子矩阵。用 CUTLASS 的 tuned shared-memory GEMM 执行子块乘法。不增加片外存储和 DRAM 流量。

### 5.0.5 CUTLASS 与 Triton

- **CUTLASS**：NVIDIA 开源库，提供 shared-memory GEMM、warp-level GEMM、iterator for block loading 等构建块
- **Triton**：语言级支持 tile load/store 到 shared memory，执行数据并行操作
- **Thunderkittens**：CUDA tile-based 编程原语库，支持 async load/store of tiles

> 对应源码：`lecture9_part2.cpp`
> 内容：直接卷积、im2col 思想、多通道卷积、池化与算术强度分析。

---

## 6. DNN 融合：减少中间结果搬运

### 6.1 Conv + Bias + ReLU

这是最经典的融合场景：

- 卷积输出一旦产生，立即加偏置并激活
- 避免将裸卷积结果落回 DRAM 再读出

### 6.2 Conv + Pooling

如果池化窗口与卷积输出块之间可局部配合：

- 可以在片上做局部聚合
- 只把池化结果写出

### 6.3 Softmax

若分多阶段分别读写中间结果，流量很大。
高效实现会尽量：

- 行内分块
- 保持局部最大值与局部和
- 尽量在寄存器或片上缓冲中完成归一化前后各步骤

### 6.4 融合的统一原则

**只要中间值不会被其他地方复用，就尽量别把它写回片外内存。**

### 6.0.1 Batch Size 对 GPU 利用率的影响

- N=1, P=Q=64：仅 524K 输出 ≈ 2MB
- N=32, P=Q=256：256M 输出 ≈ 1GB
- 大 batch 提供更多并行工作以填充 GPU

### 6.0.2 Conv + Scale + Bias 融合代码

```c
// 融合前：分开的三次循环经过 DRAM
// 融合后：直接在卷积循环内完成
for (int n=0; n<N; n++)
for (int h=0; h<H; h++)
for (int w=0; w<W; w++)
for (int k=0; k<K; k++) {
    float tmp = 0;
    for (...) tmp += ...; // 卷积计算
    output[k] = tmp * scale[k] + bias[k]; // 融合的 scale/bias
    output[k] = max(0.0f, output[k]);     // 融合的 ReLU
}
```

### 6.0.3 Softmax 融合的内存流量对比

- Naive 版：读 5MN+2M 元素、写 3MN+2M 元素
- Fused 版：仅读 MN 元素、写 MN 元素（单行 working set 放入片上存储的条件下）

---

## 7. Flash Attention

### 7.1 朴素 attention 的问题

标准 attention 常写成：

1. `S = QK^T`
2. `P = softmax(S)`
3. `O = PV`

如果显式存整个 `N x N` 分数矩阵：

- 内存占用巨大
- 读写流量巨大
- 长序列时根本不可接受

### 7.2 Flash Attention 的核心思想

- 按块处理 Q、K、V
- 不显式物化完整 `S`
- 在处理块时维护 softmax 所需的局部最大值和归一化因子
- 直接把中间概率乘 V 并累计到输出

### 7.3 为什么它是课程里的明星案例

因为它把本门课所有思想都串起来了：

- 块化
- 融合
- 提高算术强度
- 减少中间写回
- 让片上存储承担主要中间状态

### 7.4 重要认识

Flash Attention 的价值不只是“某个具体算法更快”，而是说明：

- 重新组织计算顺序，往往能从根本上改变内存复杂度与性能上限。

### 7.0.1 Transformer Attention 的精确公式

Q、K、V 均为 N×d 矩阵：
- S = QK^T（N×N 矩阵）
- P = softmax(S)（行归一化）
- O = PV

N 可达数千，朴素实现需 O(N²) 空间。

### 7.0.2 Softmax 分块计算的数学推导

对于分块 $x = [x^{(1)}, x^{(2)}]$：

$$m(x) = \max(m(x^{(1)}), m(x^{(2)}))$$

$$l(x) = e^{m(x^{(1)})-m(x)} \cdot l(x^{(1)}) + e^{m(x^{(2)})-m(x)} \cdot l(x^{(2)})$$

$$f(x) = [e^{m(x^{(1)})-m(x)} \cdot f(x^{(1)}), e^{m(x^{(2)})-m(x)} \cdot f(x^{(2)})]$$

### 7.0.3 Fused Attention 伪代码

```
for each block j of K,V:
  for each block i of Q,O:
    Load Qi, KTj, Vj, Oi into shared memory
    Compute Sij = Qi × KTj
    Compute Mij, Pij, lij (行归一化)
    Multiply Pij × Vj
    Accumulate into Oi with rescaling
```

永不物化 N² 矩阵，高算术强度（读 3 blocks、2 次矩阵乘法 + 行归约）。

### 7.0.4 DNN 优化三类方法总结

1. **更好的算法/模型设计**：深度、宽度、stride 等超参数搜索
2. **软件优化**：loop blocking/tiling、fusion（人工 + 自动）
3. **近似/压缩**：低精度量化（16-bit/8-bit 已普及，正在推进 4-bit，极端 1-bit）

### 7.0.5 GPU 为次优平台的场景

通用 GPU 仍有指令/控制开销。引出下讲专用硬件加速：Google TPU3、Huawei Kirin NPU、Apple Neural Engine、GraphCore IPU、Cerebras WSE、SambaNova SN10 等。

---

## 8. DNN 高性能实现的统一套路

无论是 GEMM、卷积还是 attention，优秀实现几乎总在做这几件事：

1. **块化**：让数据块适配片上层次。
2. **融合**：减少中间结果回写。
3. **向量化 / Tensor 化**：让算子一次做更多工作。
4. **层次化复用**：寄存器、共享内存、缓存逐层复用。
5. **重排布局**：让访存更连续、更可合并。

---

## 常见误区

1. **误区：DNN 很“算术密集”，所以主要受算力限制。**
   许多真实 kernel 实际仍受数据搬运影响很大。
2. **误区：卷积转 GEMM 一定最好。**
   若 im2col 中间展开太大，可能反而浪费带宽。
3. **误区：融合只是减少 kernel launch。**
   更重要的是减少中间结果片外流量。
4. **误区：Flash Attention 只是工程技巧。**
   它体现的是系统性重排计算顺序的思想。

---

## 对应源码

| 文件 | 主题 | 重点 |
|---|---|---|
| `lecture9_part1.cpp` | GEMM 优化 | 多层块化、向量化、寄存器微内核 |
| `lecture9_part2.cpp` | 卷积实现 | 直接卷积、im2col、池化、融合与 AI 分析 |

---

## 学完本讲应做到

- 能用 roofline 语言分析一个 DNN kernel。
- 能解释为什么 GEMM 是 DNN 系统中的基础计算内核。
- 能说清楚卷积、融合和 Flash Attention 的共同方法论。
- 能认识到“重新安排数据流”常比“减少一点 FLOPs”更重要。

