# Paper 16: LongLoRA — Efficient Fine-tuning of Long-Context Large Language Models (Chen et al., ICLR 2024)

> 论文全称：**LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models**
> 发表会议：ICLR 2024
> 作者：Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, Jiaya Jia（CUHK / MIT HAN Lab）

---

## 1. 论文解决什么问题

预训练 LLM 的上下文窗口通常有限（如 LLaMA-7B 只有 2k token）。虽然可以通过继续预训练来扩展上下文窗口，但**全注意力计算复杂度为 $O(N^2)$**，当上下文从 2k 扩展到 100k 时，计算成本飙升 2500 倍，需要大量的 GPU 资源。本文提出 LongLoRA，结合 **Shifted Sparse Attention（S²-Attn）** 与 **LoRA** 参数高效微调，将 LLaMA-7B 的上下文从 2k 扩展到 100k，训练成本仅为全注意力的 **1/3**。

---

## 2. 核心方法

### Shifted Sparse Attention（S²-Attn）

将输入序列分成等长的组（groups），使注意力计算仅在局部组内进行。为了防止信息孤岛，在**一半注意力头中应用移位**：

1. **分组（Grouping）**：将长度为 $N$ 的序列分为 $G$ 组，每组大小 $N/G$
2. **移位（Shifting）**：奇数索引的注意力头中，将分组边界移动 $N/(2G)$ 个 token
3. 分组+移位后，每个 token 的可见范围覆盖了局部邻域 + 跨组边界 token

这近似了全注意力的覆盖范围，但计算复杂度降为 $O(N^2 / G)$。

### LoRA 用于上下文扩展

标准的 LoRA 仅对 Query 和 Value 投影矩阵添加低秩适配器。LongLoRA 发现：
- 对于上下文窗口扩展任务，**嵌入层（embedding layer）和归一化层（normalization layer）也需要训练**
- 这些层参数量很小（<1% 总参数），但放开训练后显著提升长上下文能力

### 训练策略

- **渐进式扩展**：从 $\sim$4k 开始，逐步增加到 8k、16k、32k、64k、100k
- **位置编码扩展**：使用 Position Interpolation (PI) 将 RoPE 的旋转频率线性缩放

### 效率对比

| 方法 | 训练 FLOPs（100k 上下文） | 可训练参数量 |
|------|--------------------------|-------------|
| Full Fine-tune + Full Attn | 6× baseline | 100% |
| LoRA + Full Attn | 4× baseline | 1-2% |
| LongLoRA (S²-Attn + LoRA) | **2× baseline** | <1% |

---

## 3. 关键公式

### 标准全注意力 vs S²-Attn 计算复杂度

全注意力：
$$FLOPs_{\text{full}} = O(N^2 d)$$

S²-Attn（组大小为 $M = N/G$）：
$$FLOPs_{S^2} = O\left(G \cdot \left(\frac{N}{G}\right)^2 d\right) = O(N^2 d / G)$$

### Shifted 分组的注意力范围

未移位的注意力头：token $i$ 只能看到组 $\lfloor iG/N \rfloor$ 内的 token

移位后的注意力头：token $i$ 可以看到跨越两个相邻组边界的 token：
$$\text{AttnRange}_{\text{shifted}}(i) = \left[\left\lfloor \frac{iG}{N} \right\rfloor \cdot \frac{N}{G} - \frac{N}{2G}, \left\lfloor \frac{iG}{N} \right\rfloor \cdot \frac{N}{G} + \frac{N}{G} + \frac{N}{2G}\right)$$

### Position Interpolation（位置编码缩放）

RoPE 原始旋转角度：
$$\theta_i = 10000^{-2i/d}$$

缩放后的 RoPE（缩短因子 $\alpha = L_{\text{new}} / L_{\text{old}}$）：
$$\theta_i = (10000 \cdot \alpha)^{-2i/d}$$

### LoRA 低秩分解

$$W' = W + \Delta W = W + B A, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}$$

其中 $r \ll d$，通常 $r = 8$ 或 $r = 16$。

---

## 4. 实验结论

| 模型 | 上下文长度 | 方法 | PG19 PPL | Proof-Pile PPL | 训练时间（相对） |
|------|-----------|------|----------|----------------|-----------------|
| LLaMA-2-7B | 4k | Full FT + Full Attn | 7.04 | 3.28 | 1× (baseline) |
| LLaMA-2-7B | 4k | LongLoRA | **7.11** | **3.34** | **0.33×** |
| LLaMA-2-7B | 32k | LongLoRA | **6.85** | **3.08** | 0.33× vs 全注意力 |
| LLaMA-2-7B | 100k | LongLoRA | **6.67** | **2.82** | 0.33× vs 全注意力 |

- LongLoRA 在 100k 上下文下 PPL 优于全注意力方法（因为训练稳定），且训练时间仅为 1/3
- S²-Attn 在 $G=4$（即每 4 个 token 一组）时取得最佳精度-效率平衡
- 嵌入层和归一化层放开训练带来 **0.3-0.5 PPL 的额外提升**
- LongLoRA-70B 在 LongChat 多轮对话评测中达到与 GPT-3.5-Turbo-16k 接近的性能
- 使用 S²-Attn 训练后，推理时可切回标准全注意力（因为模型已学会处理长上下文），无需修改推理代码

---

## 5. 工业价值

- **显著降低长上下文 LLM 的训练门槛**：原来需要 32× A100 训练 100k 上下文，LongLoRA 仅需 8× A100
- **兼容现有推理框架**：推理时使用标准全注意力，可直接对接 vLLM、TensorRT-LLM 等
- **LoRA 生态**：LongLoRA 适配器仅 ~100MB（vs 全模型 ~13GB），易于分享和热插拔
- **实际应用**：已被多个开源长上下文模型（如 LongAlpaca、YaRN-Llama）的训练流程采纳

---

## 6. 与课程 Lecture 的关系

- **Lecture 15（Long-Context LLM）**：本文是课程中长上下文 LLM 的核心论文，示范了如何用参数高效微调（LoRA）+ 稀疏注意力（S²-Attn）两把利器实现高效上下文扩展
- **Lecture 4-6（Quantization / PTQ / QAT）**：与低精度训练有概念上的联系——LongLoRA 减少训练参数，量化减少参数精度，两者方向不同但目标一致
- **Lecture 7（NAS）**：S²-Attn 中的分组大小 $G$ 可以视为一个搜索维度——需要在精度和效率之间寻找 Pareto 最优

---

## 7. 我应该如何复现

1. **环境准备**：PyTorch 2.0+，`transformers`、`peft`（用于 LoRA）、`datasets`
2. **加载基座模型**：`meta-llama/Llama-2-7b-hf`
3. **实现 S²-Attn**：
   - 修改自注意力模块的前向传播
   - 使用 `torch.chunk` 将序列分成 $G$ 组
   - 对奇数/偶数头交替应用 shift（`torch.roll` 沿序列维度滚动 $N/(2G)$ 个位置）
   - 每组内独立计算注意力
4. **配置 LoRA**：
   - 使用 `peft.LoraConfig`，设置 `r=16, alpha=32`，target_modules 包含 `["q_proj", "v_proj"]`
   - **重要**：额外设置 `modules_to_save=["embed_tokens", "norm"]` 以训练嵌入层和归一化层
5. **位置编码扩展**：
   - 修改 RoPE 的 `base` 参数（如 10000 → 1000000）实现 Position Interpolation
6. **训练**：
   - 数据集：RedPajama（长文档语料），或 PG19 + Books3
   - 使用 DeepSpeed ZeRO-2 进行分布式训练
   - 在 8× A100 (80GB) 上约需 1000 步达到 100k 上下文
7. **验证**：在 PG19（PPL）、LongBench（检索 + 推理任务）上评估长上下文能力
