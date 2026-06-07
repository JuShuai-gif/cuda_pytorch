# Paper 06: LoRA (Hu et al., ICLR 2022)

> 论文全称：**LoRA: Low-Rank Adaptation of Large Language Models**
> 发表会议：ICLR 2022
> 作者：Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen（Microsoft）

---

## 1. 论文解决什么问题

全参数微调（full fine-tuning）大语言模型面临两个核心瓶颈：
- **显存爆炸**：以 GPT-3 175B 为例，全参数微调需要 ~350GB 显存（模型参数 × 2 for optimizer states），远超单卡能力
- **部署和管理成本高**：每个下游任务都需要保存一份完整的模型副本（175B × 每个任务），存储和切换成本极高

**LoRA** 提出：预训练模型的权重参数包含了丰富的通用知识，下游任务的微调**只需要在低秩子空间中学习增量更新**。通过将更新矩阵参数化为两个小矩阵的乘积，将可训练参数量减少 10,000 倍。

---

## 2. 核心方法

### 低秩分解假设
核心假设：微调期间的权重更新 $\Delta W$ 位于一个低秩子空间中：

$$\text{rank}(\Delta W) \ll \min(d_{\text{in}}, d_{\text{out}})$$

### LoRA 参数化
对于预训练权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$（冻结，不更新），LoRA 引入两个低秩矩阵：

$$W = W_0 + \Delta W = W_0 + BA$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$。

- $A$ 用随机高斯初始化，$B$ 用零初始化（保证训练开始时 $\Delta W = 0$）
- 前向传播：$h = W_0 x + \alpha \cdot BA x$（其中 $\alpha$ 是缩放超参数）
- 只有 $A$ 和 $B$ 接收梯度更新，$W_0$ 完全冻结

### 参数量分析
原始矩阵参数量：$d \times k$
LoRA 参数量：$r \times (d + k)$

典型设置：$d = 4096, k = 4096, r = 16$
- 原始：$4096 \times 4096 = 16.7M$ 参数
- LoRA：$16 \times (4096 + 4096) = 131K$ 参数
- 压缩比：**128× ↓**

### 应用到 Transformer
LoRA 通常只添加到 $W_Q$ 和 $W_V$（query 和 value 投影矩阵），也可以加 $W_K$（key）：

| 添加位置 | 参数量 | 推荐度 |
|----------|--------|--------|
| 仅 $W_Q$ | 最少 | ▲ 有限的适应能力 |
| $W_Q + W_V$ | 适中 | ★ 推荐（论文默认） |
| $W_Q + W_K + W_V + W_O$ | 最多 | ▲▲ 最好的微调效果 |

---

## 3. 关键公式

### LoRA 前向传播
输入 $x$，输出 $h$：

$$h = W_0 x + \frac{\alpha}{r} \cdot BA x$$

其中 $\alpha/r$ 这一定标保证了选择不同 $r$ 时训练超参数的一致性（Alpha 保持常数，实际缩放由 $r$ 补偿）。

### 梯度分析
LoRA 的梯度更新：

$$\frac{\partial \mathcal{L}}{\partial A} = \frac{\alpha}{r} \cdot B^T \frac{\partial \mathcal{L}}{\partial h} x^T$$
$$\frac{\partial \mathcal{L}}{\partial B} = \frac{\alpha}{r} \cdot \frac{\partial \mathcal{L}}{\partial h} (Ax)^T$$

$W_0$ 不接收梯度（冻结）。

### 多任务合并
推理时，不同任务的 LoRA 权重可以与基础模型 $W_0$ 合并：

$$W_{\text{task}_i} = W_0 + B_i A_i$$

切换任务时，只需矩阵加法（$B_i A_i$），无需替换整个模型。

---

## 4. 实验结论

### GPT-3 175B 下游任务（LoRA, r=4, 仅 $W_Q+W_V$）

| 方法 | 可训练参数 | WikiSQL | MNLI-m | SAMSum | E2E NLG |
|------|-----------|---------|--------|--------|---------|
| GPT-3 (Few-Shot) | 0 | 73.4 | 81.9 | 45.3 | 54.6 |
| Fine-Tune (Adapter) | 40.1M | 72.5 | 89.0 | 53.2 | 69.3 |
| Fine-Tune (Full) | 175B | 73.2 | 89.5 | 53.1 | 69.2 |
| **LoRA** | **4.7M** | **73.4** | **89.9** | **53.8** | **70.4** |

- **关键结论**：LoRA 用全微调 1/37,000 的参数量，实现了相当甚至更好的效果
- LoRA 不仅节省训练显存，在推理时也可以 merge 回原模型权重，**推理速度零开销**
- 不同 rank 的影响：`r=4` 已经足够，`r=8` 的增益很小

### 与其他 PEFT 方法对比

| 方法 | 推理额外开销 | 训练显存 | 多任务切换成本 |
|------|-------------|---------|---------------|
| Full Fine-tuning | 0 | 最高 | 每任务一份完整模型 |
| Adapter Layers | 有（额外层延迟）| 低 | 低 |
| Prefix Tuning | 有（prompt 长度） | 低 | 低 |
| **LoRA** | **0（可 merge）** | **低** | **极低** |

---

## 5. 工业价值

- **微调范式改变**：LoRA 已成为 LLM 微调的事实标准，被 Hugging Face PEFT、Microsoft DeepSpeed、NVIDIA NeMo 等主流库原生支持
- **降低微调门槛**：使用 LoRA 可以从消费级 GPU（RTX 3090/4090）微调 13B 级别模型
- **开源社区基础**：Hugging Face 上有数以万计的 LoRA 适配器，覆盖几乎所有主流开源模型
- **商业落地**：被 Anthropic（Claude）、Stability AI、Databricks 等公司用于实际生产系统
- **多 LoRA 服务**：一些推理服务支持"热插拔"多个 LoRA 适配器

---

## 6. 与课程 Lecture 的关系

- **Lecture 11 (LLM Efficiency)**：LoRA 是 LLM 高效训练的关键技术，是 Lecture 11 的重点讨论论文
- **Lecture 12 (PEFT)**：如果课程有参数高效微调专题，LoRA 是核心论文
- **与蒸馏/量化的关系**：LoRA 可以与量化（QLoRA）、蒸馏等方法结合，形成更高效的训练方案
- **基础数学原理**：低秩分解思想与课程中 SVD 分解、矩阵近似等基础数学概念紧密相关

---

## 7. 我应该如何复现

1. **环境准备**：
   ```bash
   pip install transformers peft datasets accelerate bitsandbytes
   ```

2. **加载基础模型**：加载 LLaMA-7B 或更小的模型（opt-125m 用于快速实验）：
   ```python
   from transformers import AutoModelForCausalLM
   model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
   ```

3. **配置 LoRA**：
   ```python
   from peft import LoraConfig, get_peft_model
   config = LoraConfig(
       r=8,                          # 低秩维度
       lora_alpha=32,               # LoRA 缩放系数
       target_modules=["q_proj", "v_proj"],  # 目标层
       lora_dropout=0.1,
       bias="none",
   )
   model = get_peft_model(model, config)
   ```

4. **训练**：用标准 Trainer 微调，learning_rate 设为 3e-4，仅 3-5 个 epoch 即可

5. **merge 权重**（可选）：
   ```python
   model = model.merge_and_unload()
   ```
   将 LoRA 权重合并到基础权重中，推理时无额外开销

6. **快速验证**：用 `facebook/opt-125m` + 简单的文本分类任务（IMDb/AG News）在 CPU 上验证 LoRA 的有效性

7. **进阶实验**：
   - 测试不同 rank（r=1, 2, 4, 8, 16, 32, 64）对精度的影响
   - 对比 `target_modules` 不同组合（仅 Q, Q+V, Q+K+V+O）
   - 对比 alpha 参数对训练稳定性的影响

