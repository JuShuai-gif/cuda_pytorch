# Paper-03: Llama / Llama 2 / Llama 3 — 开源大模型的 Scaling 实践

> Touvron et al., 2023. "LLaMA: Open and Efficient Foundation Language Models."
> Touvron et al., 2023. "Llama 2: Open Foundation and Fine-Tuned Chat Models."
> Meta AI, 2024. "The Llama 3 Herd of Models."

---

## 1. 解决什么问题

在 Llama 系列出现之前，开源 LLM 生态中有一个尴尬的局面：GPT-3（175B）及其后继模型（GPT-3.5、GPT-4）在几乎所有 benchmark 上碾压了小参数量模型，但这些模型是闭源的。学界和中小企业的普遍信念是"参数越大越好"——但真的是这样吗？

Chinchilla 的 scaling law 指出，**给定 compute budget，最优策略应当让训练 token 数和模型参数同步增长**（即 D ≈ 20N）。这意味着 Open AI 的 GPT-3（175B 参数，300B tokens 训练）实际上严重 under-trained——如果按 compute-optimal 的标准，GPT-3 应该用约 3.5T tokens 来训练，而不是 300B。

Llama 系列的核心问题是：**如果你不能/不想训练 500B 参数的模型，用同样的 compute budget，你能用更小的模型加更多的数据超越大模型吗？**

答案是：**可以，而且确实做到了**。

---

## 2. 核心创新

Llama 系列并非单个论文，而是持续演进的架构栈。以下是三个版本的关键创新：

### Llama 1（2023.02）——核心架构定型

**Pre-norm 而非 Post-norm**：在 Transformer 原论文中，LayerNorm 在残差连接的加法之后（x + Sublayer(x) → LayerNorm）。Llama 将其前置（LayerNorm(x) → x + Sublayer(x)）。Pre-norm 的梯度路径更稳定：残差连接旁路不经过任何归一化，梯度可以"无损"回传。

**RMSNorm 替代 LayerNorm**：

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2 + \epsilon}} \odot \gamma$$

对比标准 LayerNorm：
$$\text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \odot \gamma + \beta$$

区别在于去掉了均值中心化（subtract mean），只保留了缩放（rescale by RMS）。这并非近似——实验表明均值中心化在实践中并不必要，去掉后节省了约 7% 的计算量。为什么？因为 LayerNorm 需要两趟遍历（算均值 + 算方差），RMSNorm 只需要一次。

**SwiGLU 激活函数** 替代 ReLU 在 FFN 中：

$$\text{SwiGLU}(x) = (xW_1 \odot \text{Swish}(xW_2))W_3$$

其中 Swish(x) = x · σ(x)。SwiGLU 的门控结构允许一部分通道"关闭"（接近 0），给网络提供了类似 LSTM 门控的动态选择能力。实验表明在相同参数量下 SwiGLU 比 ReLU 好约 1-2 个 perplexity 点。

**RoPE（Rotary Position Embedding）**：

$$\begin{pmatrix} q_0 \\ q_1 \end{pmatrix} \cdot R(m\theta) = \begin{pmatrix} q_0\cos(m\theta) - q_1\sin(m\theta) \\ q_0\sin(m\theta) + q_1\cos(m\theta) \end{pmatrix}$$

RoPE 的核心思想：将 Q 和 K 向量按相邻维度配对，对每对进行旋转（角度由位置决定）。这样做有两个优雅的性质：① Q·K 的值只依赖于相对位置（不依赖绝对位置）；② 可以直接外推到比训练时更长的序列（只需做频率插值）。

### Llama 2（2023.07）——训练和改进

**GQA（Grouped Query Attention）**：MHA（Multi-Head Attention）中每个 head 有独立的 Q、K、V。GQA 将多个 Q head 共享一组 K、V head。例如 32 个 Q head 共享 8 个 K/V head。这在推理时大幅减少了 K、V 的显存占用（KV cache 缩小 4 倍），而性能几乎不下降。

**RLHF 训练 pipeline**：Llama 2 的 RLHF 包含 5 轮迭代，每轮包括：人类偏好数据收集 → 奖励模型训练 → PPO 微调 → 拒绝采样 → 再训练。这是开源社区第一次看到完整的工业级 RLHF 流程。

### Llama 3（2024.04）——极致 scaling

**Tokenizer 升级**：vocabulary 从 32k 扩展到 128k（使用 tiktoken 的 BPE），支持更好的多语言编码效率。

**8B 和 70B 两个版本**：8B 版本用 ~15T tokens 训练（远超过 Chinchilla-optimal 的 160B），70B 版本用 ~15T tokens 训练（远超 Chinchilla-optimal 的 1.4T）。这是 Meta 认为的"practical optimal"——在给定 inference budget 下，用更小的模型 + 更多数据 > 更大的模型 + 更少数据。

---

## 3. 为什么有效

Llama 系列的成功可以归结为几个核心设计原则：

1. **架构简化 ≠ 性能损失**：Pre-norm、RMSNorm、SwiGLU、RoPE——这些改变每一项都看似微小，但叠加后的效果是显著的。更重要的是，它们让训练 **更稳定**。训练稳定性对 scaling 来说比峰值性能更重要——一个架构如果在 10B 参数下崩溃了，那么它再怎么有潜力也没用。

2. **compute-optimal 的实践修正**：Chinchilla 说 D=20N 最优，但那是针对训练阶段。Llama 3 的训练策略是"在目标推理成本下最大化模型能力"。一个 70B 模型用 15T tokens 训练，推理成本远低于 405B 模型，但能力相当。

3. **数据和规模是王道**：Llama 3 用了约 15T tokens 的高质量数据，经过复杂的过滤 pipeline（去重、质量分类器、启发式规则、模型辅助过滤）。架构的改进只是 20% 的贡献，数据和规模贡献了 80%。

---

## 4. GPU/硬件角度解释

Llama 的架构选择在硬件层面上非常精细：

**GQA 的 KV Cache 节省**：设 batch=32, seq_len=4096, heads=32, d_head=128, layers=32。
- MHA 的 KV cache：32 × 2 × 32 × 4096 × 128 × 2 bytes = 约 67 GB（FP16）
- GQA (8 KV heads) 的 KV cache：32 × 2 × 8 × 4096 × 128 × 2 bytes = 约 17 GB

这对推理吞吐至关重要——KV cache 主要存放在 HBM 中，更小的 cache 意味着更大的 batch size，更高的 GPU 利用率。

**RMSNorm 的计算简化**：LayerNorm 的均值计算需要在每个元素上做 write-read-write（读→累加求均值→再读减去均值→再读求方差），至少需要 3 次全局访存。RMSNorm 只需 2 次。在大 batch 训练中，norm 操作的内存带宽对端到端速度影响显著。

**RoPE 的 fuse 优化**：RoPE 的旋转变换可以 fused 到 attention 的 Q、K 计算中，不需要额外的 kernel launch。在 CUDA 层面，这节省了 kernel launch overhead（~10μs）和一次 global memory roundtrip。

---

## 5. 工业意义

Llama 是开源 LLM 的"Linux 时刻"：

1. **打破了"闭源才强"的叙事**：Llama 2-70B 在多个 benchmark 上接近 GPT-3.5-turbo，Llama 3-70B 超过了 GPT-3.5-turbo 和 Claude Sonnet。

2. **定义了现代 LLM 架构标准**：Pre-norm + RMSNorm + SwiGLU + RoPE + GQA 几乎成为了所有后续开源模型（Mistral、Qwen、Yi、DeepSeek）的默认选择。

3. **证明了数据质量的杠杆效应**：Llama 1 只用 1-1.4T tokens 就达到了比 Chinchilla-optimal 模型更好的效果，关键在于更高比例的高质量数据。

4. **推动了 RLHF 开源生态**：Llama 2 的 RLHF 论文详细描述了 data collection pipeline、reward model training、PPO implementation，让整个社区都能复现并对齐大模型。

5. **生态飞轮**：Hugging Face、LangChain、Ollama、llama.cpp 等工具链迅速围绕 Llama 构建了完整的推理/微调生态。

---

## 6. 如何复现

关键实现细节：

1. **RMSNorm + ε 值**：通常 ε=1e-6（FP32）或 1e-5（FP16）。在混合精度训练中，norm 计算通常在 FP32 精度下进行（FP16 的数值范围不够）。

2. **RoPE 的 base frequency**：Llama 默认是 10000，对应正弦位置编码的 1/10000^(2i/d)。为了支持长上下文（>4K），需要调整 base frequency 或用 NTK-aware/ YaRN 插值方法。

3. **SwiGLU 导致 FFN 参数量膨胀**：标准 FFN 的参数量是 2 · d_model · d_ff。SwiGLU 需要 3 个投影矩阵（W₁, W₂, W₃），参数量是 3 · d_model · d_ff。为了保持整体参数量不变，通常将 d_ff 乘以 2/3（即从 4d 减到 8/3 d）。

4. **训练超参**：AdamW optimizer, β₁=0.9, β₂=0.95（比常用 0.999 更激进）, weight decay=0.1, grad clip=1.0, cosine LR schedule 降到 max LR 的 10%。

5. **数据混合**：Llama 1 的代码数据占比很小但效果显著——代码能力的提升显著提高了推理（reasoning）能力。后续所有 LLM 训练都包含了可观的代码数据比例。

6. **Sequence packing**：为了最大化 GPU 利用率，将短样本拼接成完整的 sequence，用 attention mask 隔开，避免 padding 浪费。

---

## 7. 面试要点

**必问题**：

1. **Pre-norm vs Post-norm 的区别，为什么 Pre-norm 更稳定？**
   答：Pre-norm 的残差路径不经过 normalization，梯度可以直接回传，避免了梯度消失/爆炸。Post-norm 中每个 sublayer 的输出都要被归一化后再加残差，tan 多层后 norm 的梯度会累积导致不稳定。

2. **RMSNorm 和 LayerNorm 的区别？为什么去掉 mean centering 不影响性能？**
   答：RMSNorm 只做 re-scaling（除以 RMS），不做 re-centering（减去均值）。实践中均值的信息被后续的线性层的 bias 或残差连接补偿了，因此 re-centering 冗余。

3. **RoPE 的核心公式和相比于 sinusoidal positional encoding 的优势？**
   答：RoPE 通过旋转变换将绝对位置嵌入到 Q 和 K 的每一对相邻维度上。优势：① 自然编码了相对位置（Qᵢ·Kⱼ 只依赖于 i-j）；② 可以通过调整 base frequency 实现长度外推。

4. **GQA 为什么能减少推理开销？对训练有影响吗？**
   答：GQA 减少了 KV cache 的大小（头数从 N 减到 N/K 组），推理时显存和带宽压力降低，batch size 可以更大。训练时 GQA 从 MHA 初始化（复制 K、V 权重），所以训练开销与 MHA 几乎相同。

5. **为什么 Chinchilla 预测的 compute-optimal 和 Llama 3 的实际训练 tokens 差距这么大？**
   答：Chinchilla 只优化训练 cost。Llama 3 还考虑了推理 cost——小模型推理更快更便宜。如果推理阶段会服务数百万请求，那么增加训练 cost 来代替更大的推理 cost 是划算的。这被称为"inference-aware scaling"。

6. **SwiGLU 的三层权重结构是怎样的，相比于普通 FFN 参数量如何？**
   答：普通 FFN 是上投影 + 激活 + 下投影（2 个矩阵）。SwiGLU 是门控投影 + 上投影（经 Swish 门控）+ 下投影（3 个矩阵）。参数量是 3/2 倍，所以需调整中间维度保持公平比较。
