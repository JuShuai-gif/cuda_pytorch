# Paper-01: Attention Is All You Need — Transformer 原始论文导读

> Vaswani et al., 2017. "Attention Is All You Need." NeurIPS 2017.

---

## 1. 解决什么问题

在 Transformer 出现之前，序列建模的主流方法是 RNN、LSTM 和 GRU。这些架构有一个根本性的瓶颈：**序列计算是顺序依赖的**。LSTM 在处理第 t 个 token 时，必须先完成前 t-1 个 token 的计算，这意味着无法在时间维度上并行化。

这种顺序性的代价在工程上体现得尤为明显。训练一个机器翻译模型时，一个 50 词的句子就需要 50 个串行的时间步。在 GPU 上，这等于每次只用一个 SM（Streaming Multiprocessor），其余计算单元全部空闲——这是对硬件的极度浪费。更糟糕的是 RNN/LSTM 在处理长序列时存在"遗忘"问题：虽然 LSTM 引入了门控机制（遗忘门、输入门、输出门）来缓解梯度消失，但当序列长度超过大约 50 个 token 时，早期的信息仍然很难有效传递到后面。

卷积架构（如 ByteNet、ConvS2S）尝试用 CNN 解决并行化问题，但 CNN 的感受野是固定的，长距离依赖需要堆叠很多层，计算复杂度 O(n log n) 也不够理想。

Transformer 要解决的核心问题是：**如何在保持对全局依赖建模能力的同时，实现完全并行化的序列处理**。

---

## 2. 核心创新

Transformer 的数学核心是 Scaled Dot-Product Attention：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

其中 Q（Query）、K（Key）、V（Value）都是输入矩阵的线性投影：

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

关键在于以下几点设计：

**Scaled Dot-Product 的数学直觉**：Q 和 K 做点积得到的是"相关性分数"矩阵，衡量序列中每对位置之间的关联强度。除以 $\sqrt{d_k}$ 是因为当 $d_k$（key 维度）很大时，点积的方差会增大到 $d_k$，导致 softmax 进入梯度极小区间。除以 $\sqrt{d_k}$ 将方差控制为 1。

**Multi-Head Attention**：将 Q、K、V 分割成 h 个 head，每个 head 独立执行 attention：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W_O$$

其中 $\text{head}_i = \text{Attention}(QW_Q^i, KW_K^i, VW_V^i)$。

这不是简单的"算多遍"，而是让不同的 head 关注不同的子空间——一个 head 可能关注语法结构，另一个可能关注语义相似性。

**Positional Encoding**：由于 attention 对位置完全不敏感（输入顺序打乱后输出也相应打乱），需要注入位置信息：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

选择正弦函数并非任意——它使得 PE(pos+k) 可以被表示为 PE(pos) 的线性函数，理论上模型可以学习相对位置关系。

---

## 3. 为什么有效

从信息论角度看，attention 机制为每个 token 提供了一个"全局视野"——不需要像 RNN 那样把信息压缩到一个隐状态中，而是可以直接访问序列中的每一个 token。

从优化角度看，Transformer 的梯度路径极短且均质。在 LSTM 中，timestep 0 的信息要经过 t 次变换才能到达 timestep t，梯度需要穿过 t 个非线性层。而在 self-attention 中，任意两个位置的交互路径长度都是 O(1)——只是一个 attention 矩阵的元素，梯度直接回传。

从表示能力的角度看，Multi-Head Attention 提供了一种"动态网络"：attention 权重是输入数据决定的，等价于每个输入实例拥有不同的连接权重模式。这与 CNN 的固定卷积核形成鲜明对比。

---

## 4. GPU/硬件角度解释

这可能是 Transformer 最被低估的部分：**为什么 attention 在 GPU 上比 RNN 更快**。

RNN/LSTM 是 **recurrent computation**——每个时间步依赖前一个时间步，是一个天然的串行瓶颈。GPU 虽然有数千个核心，但由于依赖链，同一时刻只能做当前时间步的计算。500 长度的序列 = 500 个串行步骤。这是典型的 **latency bound**（延迟受限）。

而 self-attention 的核心操作是矩阵乘法 $QK^T$ 和 softmax 后的再乘 $V$。矩阵乘法是 GPU 上优化程度最高的操作——cuBLAS 库将大矩阵乘法充分向量化，将所有 SM 全部利用起来。$QK^T$ 的计算本质上是 $Q$ 的每一行与 $K$ 的每一行做内积，这可以完全并行。这是典型的 **compute bound**——计算密集，GPU 可以满负荷运转。

还有一个关键区别：**arithmetic intensity**（运算强度，FLOP/byte）。矩阵乘法的 arithmetic intensity 是 O(n)（n 是矩阵维度），而 RNN 的门控计算是 O(1)（每个参数只做一个乘加）。在高 arithmetic intensity 的操作中，数据搬运（从 HBM 到 SRAM）的时间被计算时间完全覆盖，利用率极高。

不过这也有代价：$QK^T$ 的显存占用是 O(n²)，对于长序列，attention matrix (n×n) 会比输入 (n×d) 大得多。这也是为什么后续会出现 FlashAttention——专门解决 attention 的 memory bottleneck。

---

## 5. 工业意义

Transformer 的影响远超学术界：

1. **BERT（2018）**：直接用 Transformer Encoder 做预训练，开启了预训练-微调范式，将 NLP 的 SOTA 全面刷新。

2. **GPT 系列**：用 Transformer Decoder 做自回归生成，从 GPT-1 到 GPT-4，参数从 1.17 亿扩展到万亿级别，证明了 Transformer 的 scaling 能力。

3. **跨模态应用**：Vision Transformer (ViT) 将图像编码为 patch 序列，证明 Transformer 在视觉任务上同样有效。后续的 CLIP、DALL-E、Sora 本质上都是 Transformer 架构。

4. **硬件生态**：Transformer 催生了专门的硬件优化——FP8/CUTLASS 中的 attention kernel、FlashAttention、TPU 中专门为 attention 优化的矩阵乘法单元。

可以说，Transformer 是过去十年 AI 领域最重要的架构创新，它证明了"scale matters"，而 scale 的前提是架构本身能够高效利用硬件。

---

## 6. 如何复现

关键实现细节（往往是论文中没有明说但工程上致命的）：

1. **Input Embedding Scaling**：原论文将 embedding 权重乘以 $\sqrt{d_{model}}$。这是因为 positional encoding 的值在 [-1, 1] 区间，而 embedding 的初始值通常较小（~0.02），直接相加会让位置信号淹没语义信号。

2. **Layer Normalization 位置**：原始论文用的是 Post-LN（在残差连接之后），但后续研究发现 Pre-LN（在 attention/FFN 之前）训练更稳定。现在几乎所有实现都用 Pre-LN。

3. **Dropout 位置**：在每个 sub-layer 的输出上（在残差加法之前），以及 attention weights 上，还有 embedding 上。一共三处 dropout，这往往被初学者忽略。

4. **Label Smoothing**：原始论文用了 0.1 的 label smoothing，虽然看似细节，但对最终 BLEU 分数影响显著。

5. **Beam Search**：推理时用 beam size = 4，加 length penalty α = 0.6 来控制生成长度。

一个纯 NumPy 的最小复现代码约 300 行；一个完整的训练 pipeline（包括混合精度）约 1000 行。核心 attention 实现的错误通常出在两个地方：mask 的方向（上三角 vs 下三角）和 softmax 的数值稳定性（减去最大值）。

---

## 7. 面试要点

**必问题**：

1. **为什么是 scaled dot-product，而不是普通的 dot-product？**
   答：当 $d_k$ 增大时，点积方差增大到 $d_k$，softmax 饱和导致梯度消失。除以 $\sqrt{d_k}$ 可将方差控制为 1。

2. **Multi-Head Attention 为什么比 single-head 好？**
   答：不同 head 关注不同子空间（语法、语义、共指等），相当于集成学习。同时分割 head 降低每个 head 的 $d_k$，减少计算量。

3. **为什么 Transformer 需要 Positional Encoding？**
   答：Self-attention 对位置无感知——"我 吃 苹果"和"苹果 吃 我"在 attention 矩阵中完全相同。必须注入位置信息。

4. **Transformer 的复杂度是多少？**
   答：Self-attention 的复杂度是 O(n²·d)（n 是序列长度，d 是模型维度）。当 n >> d 时是 O(n²)，当 d >> n 时是 O(n²·d)。这也是长序列处理的瓶颈。

5. **Encoder 和 Decoder 的 self-attention 有什么区别？**
   答：Decoder 的 self-attention 是 masked（causal）的，遮住了未来的 token；Encoder-Decoder attention 中 Q 来自 decoder，K 和 V 来自 encoder。

6. **为什么 Transformer 的 decoder 可以并行训练，但必须串行推理？**
   答：训练时用 teacher forcing，一次性输入整个目标序列，用 mask 防止看到未来。推理时是逐 token 自回归生成，必须串行。
