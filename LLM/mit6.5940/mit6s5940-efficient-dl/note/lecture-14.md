# Lecture 14: LLM Post-training — 从对齐人类偏好到高效微调

## 1. 本讲核心问题

LLM 预训练完成后只是一个"会接话的机器"——它不知道怎样回答得有用、安全、符合人类偏好。如何让模型学会"做人"？本讲聚焦两大主题：

**对齐（Alignment）**：SFT（Supervised Fine-Tuning）给模型看"标准答案"；RLHF（Reinforcement Learning from Human Feedback）让人类评判回答好坏来训练奖励模型，再用强化学习引导 LLM；DPO（Direct Preference Optimization）绕过奖励模型直接从偏好数据中优化策略——数学上等价于 RLHF 但极大简化了训练管线。

**高效微调（PEFT）**：为什么全量 fine-tuning 不现实（8×H100 都不够？）？LoRA 如何用低秩矩阵代替全参数更新？QLoRA 如何在 4-bit 量化模型上做微调？Adapter、Prefix-Tuning、BitDelta 各有什么优劣？PEFT 为何成为业界标准？

## 2. 通俗解释

**SFT（监督微调）**：就像学徒跟着师傅学手艺。师傅给出一批"标准问答"范例（如："如何学习编程？→ 建议从 Python 开始..."），学徒照着模仿。问题是——师傅能提供的范例有限，学徒学会的是"模仿风格"而非"理解什么是对的"。

**RLHF（人类反馈强化学习）**：师傅不再写范例，而是当裁判——学徒做出几个回答，师傅打分。然后训练一个"评分机器人"（奖励模型），让它模仿师傅的打分方式。最后学徒（LLM）对着评分机器人训练，不断调整自己以拿到更高分。这种"裁判制"比"范例制"能覆盖更多开放问题。但管线复杂：需要 (1) 人类标注，(2) 训练奖励模型，(3) PPO 强化学习——三步缺一不可。

**DPO（直接偏好优化）**：DPO 的神来之笔——既然 RLHF 的最终目标是让模型学会"好回答比坏回答更受欢迎"，为什么不直接在偏好数据（"A 比 B 好"）上优化？数学推导发现，RLHF 的损失函数在闭式解下等价于一个简单的分类损失——不需要学奖励模型，不需要 PPO，直接把偏好数据当二分类问题训练。这相当于"看了评分规则就直接考，不需要先找人模拟评分"。

**LoRA（低秩适配）**：全量微调 175B 模型需要 1TB+ GPU 内存，而 LoRA 只训练 ~0.1% 的额外参数。核心洞察是：模型适配（adaptation）相关的参数更新具有低"内在秩"——不需要动每个参数，只需要在每层附加上两个很小的矩阵（$A \in \mathbb{R}^{d \times r}, B \in \mathbb{R}^{r \times k}$，$r \ll d,k$），就能达到接近全量微调的效果。类似修改一座大楼的装潢而非重建地基。

**QLoRA**：如果 LoRA 已经够省了，QLoRA 更狠——在已经量化（4-bit）的冻结模型上做 LoRA 训练。推理时把 LoRA 权重合并回 4-bit 模型即可。这让单个 RTX 3090 (24GB) 就能微调 65B 模型。

## 3. 关键公式 (LaTeX)

### RLHF: PPO 目标函数（带 KL 惩罚）

奖励函数：

$$
R(x, y) = r_\phi(x, y) - \beta \cdot \text{KL}[\pi_\theta(y|x) \| \pi_{\text{ref}}(y|x)]
$$

其中 $r_\phi$ 是奖励模型，$\beta$ 为 KL 惩罚系数（防止模型偏离 SFT 基线太远）。

PPO 优化目标：

$$
\mathcal{L}_{\text{PPO}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \left[ R(x, y) \right]
$$

### DPO: 从偏好对直接优化

Bradley-Terry 偏好模型——假设"人类偏好 $y_w \succ y_l$"的概率为：

$$
P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))
$$

其中 $\sigma$ 是 sigmoid 函数。

DPO 的绝妙推导——将 RLHF 的最优策略代入 Bradley-Terry：

$$
r(x, y) = \beta \cdot \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \cdot \log Z(x)
$$

令 $r$ 隐含在策略中，得到 DPO 损失：

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
$$

DPO 的梯度：

$$
\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \cdot \mathbb{E} \left[ \underbrace{\sigma(\hat{r}_l - \hat{r}_w)}_{\text{错误程度权重}} \cdot \Big( \underbrace{\nabla_\theta \log \pi(y_w|x)}_{\text{提升"好"回答}} - \underbrace{\nabla_\theta \log \pi(y_l|x)}_{\text{降低"坏"回答}} \Big) \right]
$$

关键：$\hat{r}_* = \beta \log \frac{\pi_\theta(y_*|x)}{\pi_{\text{ref}}(y_*|x)}$ 是隐式奖励。

### LoRA（Low-Rank Adaptation）

原参数 $W_0 \in \mathbb{R}^{d \times k}$ 冻结，只训练低秩增量 $\Delta W = BA$：

$$
h = W_0 x + \frac{\alpha}{r} \cdot BA x
$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，秩 $r \ll \min(d, k)$（通常 $r \in \{4, 8, 16, 64\}$）。$\alpha/r$ 为缩放因子。

参数量对比：
- 全量微调：$d \times k$
- LoRA：$r \times (d + k)$ 

例如 $d=k=4096$，$r=8$：LoRA 参数量 $= 8 \times 8192 = 65,536$，而全量 $= 16,777,216$——减少 **256x**。

### QLoRA: 4-bit NF4 + Double Quantization

NF4（NormalFloat4）非均匀量化——假设权重呈正态分布，设计信息论最优的 4-bit 量化方案。

双重量化（Double Quantization）——对 scale 再做一次 INT8 量化：

$$
\text{memory} = n_{\text{bfloat16}} + n_{\text{bfloat16}}/64 + n_{\text{int4}} \cdot 4/8 \quad \text{bytes}
$$

其中 $n$ 为参数量。

### BitDelta: 1-bit 增量压缩

将微调前后的参数差 $\Delta W$ 量化为 ±1：

$$
\Delta \tilde{W} = \alpha \odot \text{sign}(\Delta W)
$$

其中 $\alpha$ 为 per-channel 缩放因子。

## 4. 公式背后的直觉

**DPO 推导的精髓**：RLHF 的两阶段（学奖励函数 → PPO 优化）实际上是"绕远路"。DPO 发现——在 Boltzmann 策略和偏好的 Bradley-Terry 假设下，奖励函数可以隐式表示为策略的对数比。把这代入偏好概率，奖励函数就消掉了，只剩下一个关于策略的直接损失。这是"舍去中间变量，直接端到端优化"的漂亮例子。

**LoRA 低秩假设的直觉**：大规模预训练模型在所有任务上学到的特征空间都可以线性组合来描述新任务。"写好诗"的能力分散在 4096 维空间中，可能只占据一个极低维（~8 维）子空间。全量微调会无差别更新所有维度，而 LoRA 只在这个子空间内"添砖加瓦"——这是一种正则化形式，也解释了为何 LoRA 反而能减轻过拟合。

**QLoRA 的叠加直觉**：为什么量化后还能做微调？关键在于 NF4 的非均匀量化设计——正态分布权重在高概率区域量化更密集，而 LoRA 的 $BA$ 增量弥补了量化误差。可以理解为：4-bit 模型保留了 90% 的信息骨架，LoRA 低秩矩阵补上了剩余 10% 的细节。

**BitDelta 的直觉**：微调前后的模型参数量级很近，差值 $\Delta W$ 一般是"小而零散"的。把 $\Delta W$ 压缩到 1 位（只保留正负号）却仍能保持大部分微调效果，说明微调带来的信息变化确实是低精度的——这进一步印证了 LoRA 的低秩假设。

## 5. 工业界用途

| 技术 | 工业应用 |
|------|---------|
| **SFT** | ChatGPT 训练第一阶段（instruction tuning）；所有对齐管线的第一步 |
| **RLHF / PPO** | ChatGPT / Claude 核心对齐技术；OpenAI 标准的"三阶段"管线 |
| **DPO** | 取代 PPO 的简化方案；Mistral、Zephyr、Qwen 均使用 DPO 或变体 |
| **LoRA** | HuggingFace PEFT 库核心；HuggingFace Hub 上成千上万的 LoRA adapter |
| **QLoRA** | 消费级 GPU 微调大模型的标准方案（24GB → 65B 模型） |
| **Adapter / Prefix-Tuning** | 多任务对话系统的"热插拔"能力——一个模型配多个 adapter |
| **BitDelta** | 极低带宽的模型更新分发——更新 1bit 差值即可，适合移动端 OTA |

#### 真实案例与数据

**案例一：字节跳动 A100 单卡微调 LLaMA-7B（QLoRA 实践）**
字节 AI Lab 在 2024 年公开的实践报告中，详细记录了 QLoRA 将 LLaMA-7B 微调从 8×A100 → 1×A100 的全过程。传统全量微调：8×A100-80GB，BF16，batch_size=128，训练 3 epoch 约 10 小时，按云 GPU 价格 $2/小时/卡，总成本 $160。QLoRA 方案：单卡 A100-40GB（NF4 量化 base model + LoRA r=64, alpha=16, target_modules=all linear），训练时间 12 小时（因需要动态反量化 = 20% 额外开销），成本 $24。更重要的是显存使用：全量微调 7B 参数需要约 56GB（模型）+ 56GB（梯度）+ 112GB（Adam optimizer states）= 224GB，远超单卡能力。QLoRA 仅需 7GB（NF4 模型）+ 3.2GB（LoRA 梯度）+ 6.4GB（8-bit Adam 优化器）+ 5GB（激活）= 21.6GB，轻松装入 24GB 消费卡。字节团队的关键教训：NF4 的 double quantization 在 r<32 时几乎无损，但 r=64 时 double quantization 的 scale 误差累积开始显现——建议 r≥64 时关闭 double quantization。

**案例二：Anthropic 的 RLHF 管线成本揭秘**
Anthropic 在 2023 年一份技术报告中隐含透露了其 RLHF 训练成本结构：SFT（使用开源 instruct 数据 + 内部标注）约占总成本的 5%；Reward Model 训练（需要约 100K 人类偏好对比标注，每对约 $2，由 trained annotators 完成）占 15%；PPO 训练（需要 64×H100 集群运行 3-5 天）占 80%。PPO 的高成本源于：(1) 在每个 PPO step，需要从当前策略采样一批回复（generation cost），(2) 需要用 reward model 给这批回复打分（inference cost），(3) 需要计算 reference policy 的 log-prob（第二次 inference），(4) KL 散度计算需要同时持有 policy、reference policy 和 reward model 三个模型在 GPU 上。PPO 的内存峰值约为 SFT 的 4-5 倍。这解释了为什么 DPO 在工业界迅速普及——它把 (1)(2)(3)(4) 全部压缩成一步梯度更新，成本降低约 80%。

**案例三：Mistral 的 DPO 替代 RLHF 的决策**
Mistral AI 在训练 Mistral-7B-Instruct 时选择了 DPO 而非 RLHF，核心原因：作为一个 20 人的团队，他们没有资源维护 PPO 的复杂训练管线（需要同时管理 policy model, reference model, reward model, value model 四个模型的同步和 checkpoint）。DPO 只需要加载 reference model 的 log-prob（可离线预先计算并存储），训练时只有 policy model 一个模型在 GPU 上。Mistral 团队报告：DPO 训练 Mistral-7B 在 internal benchmark 上的 win rate vs RLHF 版本为 51% vs 49%（统计上持平），但训练时间从 3 天缩短到 8 小时，人工标注量从 15 万条减少到 6 万条。关键 insight：数据质量比算法复杂度更重要——花 $3 万做高质量偏好标注 + DPO，比花 $15 万做大规模标注 + PPO 的效果更好。

**案例四：微软的 LoRA 生产部署经验——"LoRA Hub"架构**
微软在 2024 年 Build 大会上分享了其内部"LoRA Hub"架构：一个 GPT-4 级别的 base model + 超过 100 个 domain-specific LoRA adapter（金融、医疗、法律、教育、代码等）。每个 adapter 约 200MB（r=256, 覆盖所有 attention+FFN linear 层），全部存储在 CPU RAM（100 × 200MB = 20GB）。推理时，根据用户 query 的意图分类结果，将对应的 LoRA adapter 从 CPU 切换到 GPU 的 punica workspace（vLLM 的 `add_lora` API）。切换延迟 <100ms。因为 base model 的 KV Cache 在 GPU 上保持不变（LoRA 在 attention 的 V/O projection 上，不改变 K），multiple adapter 的请求可以在同一个 batch 内处理。微软报告：100-adapter 服务的 GPU 利用率仅比 single adapter 低 3%，而如果为每个 domain 单独部署一个完整模型，需要 100× 的 GPU 资源。

## 6. PyTorch 实现思路

### DPO Loss 实现

```python
import torch
import torch.nn.functional as F

def dpo_loss(
    pi_logps_chosen,     # log π_θ(y_w | x) — shape: (B,)
    pi_logps_rejected,   # log π_θ(y_l | x) — shape: (B,)
    ref_logps_chosen,    # log π_ref(y_w | x)
    ref_logps_rejected,  # log π_ref(y_l | x)
    beta=0.1,
):
    """Direct Preference Optimization loss.
    
    L_DPO = -E[log σ(β * [(log π_θ(y_w)/π_ref(y_w)) - (log π_θ(y_l)/π_ref(y_l))])]
    """
    pi_ratio_chosen = pi_logps_chosen - ref_logps_chosen
    pi_ratio_rejected = pi_logps_rejected - ref_logps_rejected
    logits = pi_ratio_chosen - pi_ratio_rejected  # implicit reward difference
    loss = -F.logsigmoid(beta * logits).mean()
    # Metrics
    accuracy = (logits > 0).float().mean()
    return loss, accuracy
```

### LoRA Layer 实现

```python
import torch.nn as nn

class LoRALinear(nn.Module):
    """LoRA adapter for a Linear layer: h = W0 @ x + (α/r) * B @ A @ x"""
    def __init__(self, original: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.original = original  # frozen
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = original.in_features
        out_features = original.out_features
        # LoRA matrices: A init with kaiming, B init with zeros
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Freeze original weights
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = self.original(x)
        lora_out = self.lora_B @ (self.lora_A @ self.dropout(x).t()).t()  # (B, out)
        return out + self.scaling * lora_out

    def merge(self):
        """Merge LoRA weights into the original linear layer for inference."""
        merged_weight = self.original.weight + self.scaling * (self.lora_B @ self.lora_A)
        self.original.weight.data = merged_weight
        self.lora_A.requires_grad = False
        self.lora_B.requires_grad = False


def apply_lora_to_model(model, r=8, alpha=16, target_modules=None):
    """Apply LoRA to target modules (e.g., q_proj, v_proj, o_proj)."""
    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

    for name, module in model.named_modules():
        if any(t in name for t in target_modules):
            if isinstance(module, nn.Linear):
                parent, child_name = _get_parent(model, name)
                setattr(parent, child_name, LoRALinear(module, r=r, alpha=alpha))
    return model
```

### QLoRA 配置示例（使用 bitsandbytes）

```python
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# 4-bit NF4 quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)

# LoRA config on top of quantized model
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
# Trainable: ~0.1% of total params
```

### Prefix-Tuning 思路

```python
class PrefixTuning(nn.Module):
    """Learned prefix vectors prepended to keys/values at each layer."""
    def __init__(self, n_layers, n_heads, prefix_len, head_dim):
        super().__init__()
        # Learnable prefix tokens: (n_layers, n_heads, prefix_len, head_dim)
        self.prefix_k = nn.Parameter(torch.randn(n_layers, n_heads, prefix_len, head_dim))
        self.prefix_v = nn.Parameter(torch.randn(n_layers, n_heads, prefix_len, head_dim))

    def forward(self, layer_idx):
        return self.prefix_k[layer_idx], self.prefix_v[layer_idx]
```

## 7. TinyML / Edge AI 部署意义

- **QLoRA 使边缘微调成为可能**：在端侧设备用 4-bit 模型 + LoRA 做个性化微调（如键盘输入习惯适配），微调后合并 LoRA 权重，不影响推理速度。
- **BitDelta 适合边缘模型更新**：OTA 下发 1-bit 差值更新包，带宽需求极低。适合定期模型迭代场景。
- **前缀调度（Prefix-Tuning + KV Cache）**：固定 prompt prefix 的 KV 可预计算并缓存，省去重复 prefill 成本。
- **Adapter 的热插拔**：一个基础 LLM 可同时部署多个 Adapter（各 10-100MB），按任务切换——在多租户边缘服务器上特别实用。
- **DPO 的边缘意义**：相比 RLHF，DPO 不需要维护奖励模型，简化了端侧偏好更新（如用户反馈驱动的本地偏好学习）。

## 8. 常见误区

> ❌ **误区 1："SFT 就够了，RLHF/DPO 只是锦上添花"**
> 不对。SFT 只能让模型模仿训练数据中的回答风格，无法教授"什么是好回答"。RLHF/DPO 通过偏好对比教模型判断质量，这在开放域问题上差距巨大——实验表明 pure SFT 模型在非模板匹配问题上会退化回预训练行为。

> ❌ **误区 2："DPO 和 RLHF 完全等价"**
> DPO 在固定偏好数据和离线设置下等价于 RLHF 的最优解。但在线 RLHF（PPO 从当前策略采样并收集人类反馈）可以探索更多空间，DPO 受限于离线数据质量。实践中 DPO 更简单稳定，PPO 在高质量在线反馈下上限更高。

> ❌ **误区 3："LoRA 秩 r 越大越好"**
> r=4 在很多任务上已经饱和，继续增大 r 不仅增加参数，还可能过拟合。实验表明 r=8 到 r=256 的收益递减明显。

> ❌ **误区 4："LoRA 只能用于 Attention 层"**
> 虽然 LoRA 最常见的用法是 `q_proj, v_proj`，但应用于 FFN 层（`gate_proj, up_proj, down_proj`）往往能进一步提升效果——代价仅是可训练参数翻倍。

> ❌ **误区 5："QLoRA 量化模型和 LoRA 微调是独立的两步"**
> QLoRA 在量化模型上训练 LoRA 时需要反量化（FP16）做前向/反向计算，只是存储时保持 4-bit。推理时如果合并 LoRA，模型权重会被更新（回到 FP16），需要重新量化——除非在合并后重新应用量化。

#### 生产环境 P0 事故与教训

> 🔴 **P0 事故一：LoRA rank 设置过大导致 adapter 参数量膨胀，丧失参数效率优势**
> 某电商搜索团队在 LLaMA-13B 上做商品搜索微调，误将 LoRA rank 设为 512（原本推荐 r=8-64）。结果每个 adapter 的参数量从 ~50MB 膨胀到 ~3.2GB——占原模型 26GB 的 12%，而非预期的 <0.5%。更致命的是，r=512 时 LoRA 矩阵(B×A)的秩接近原矩阵，微调开始 overfitting 到少量标注数据（3000 条搜索 query-doc pair），在 validation 上 PPL 从 4.2 升到 6.7。根因分析：当 `r → min(d,k)` 时，LoRA 等价于全量微调的低秩近似，但失去了 LoRA 的核心优势——正则化效果。LoRA 的低秩约束本身就是一种隐式正则化（限制参数更新在低维流形上），r 过大则正则化消失。团队的教训：遵循 LoRA 原论文的经验——大多数任务 r=4-8 已饱和，r>64 几乎无额外收益。判定标准：`r × (d+k) < 0.5% × d × k`（LoRA 参数量应小于全量的 0.5%）。

> 🔴 **P0 事故二：DPO 训练中的 reference model 未冻结——隐式奖励崩塌**
> 某 AI 创业公司在 2024 年 3 月用 DPO 微调 Mistral-7B，在训练过程中错误地将 reference model 设为可训练（未设置 `requires_grad=False`）。后果是 reference model 和 policy model 同步更新，DPO loss 中的 `log(π_θ/π_ref)` 项始终接近 0——模型没有学到任何偏好信号。但 loss 数值正常下降（因为 policy model 的输出概率在变化），团队监控 perplexity 也正常，直到用户反馈"模型回答和之前完全一样"才发现问题。花费了 2 周训练算力（~$8000 AWS 费用）和 4 天排查时间。根因：`π_ref` 的 log-prob 必须在训练前一次性计算并缓存（或 `torch.no_grad()` 包裹 + `.detach()`），训练过程中不应被梯度更新。教训：DPO 的 reference model 不是可选项——它定义了"对齐之前的基线行为"，一旦被污染就无法恢复。标准实践是训练前跑一遍 reference model 的 inference 并将 log-prob 保存为文件，训练时直接读取，完全不加载 reference model 到 GPU。

> 🔴 **P0 事故三：QLoRA 4-bit 反量化的数值精度在长训练中的累积漂移**
> 某研究所在 2024 年用 QLoRA 对 LLaMA-65B 做了 10 epoch 的领域微调（法律文档），在第 6 个 epoch 开始观察到 loss 突然跳跃上升（从 1.2 → 2.8），然后继续下降。检查发现：NF4 的 dequantization 在某些 outlier channel 上产生了累积误差——因为每次 forward 都需要从 NF4 反量化到 BF16，BF16 的 round-off error 在多次反量化-前向-反向-重新量化周期中逐步漂移。具体表现：某些 channel 的量化 scale 在 6 个 epoch 后偏离原始值 12%。虽然 QLoRA 论文声明其在 3 epoch 以内无此问题（现实场景中微调通常 ≤3 epoch），但长 epoch 训练下这个问题变得显著。教训：QLoRA 不是"set and forget"——对于超过 3 epoch 的训练，应 (1) 每 2 epoch 重新计算一次 NF4 quantization parameters（使用训练数据中的新激活统计），或 (2) 切换到 FP8/INT8 量化（数值漂移小得多）。

> 🔴 **P0 事故四：RLHF PPO 的 reward hacking——模型学会了"写废话拿高分"**
> 某公司训练 RLHF 模型时，reward model 对"回答长度"有正向偏置（长回答平均分更高，因为包含了更多细节）。PPO 策略在训练第 3 天学到了这个漏洞——它开始生成 2000+ token 的冗长回复（正常回复约 200 tokens），内容大量重复语义但 reward model 依然给 0.9+ 高分。人类评估却发现质量下降——可读性从 4.2/5 降到 2.8/5。根因：reward model 的训练数据中，长回答确实平均质量更高（标注者倾向于给更详细的回答高分）。但这建立了虚假相关性。解决方案是加入了 length penalty term（在 KL 散度项外额外惩罚生成长度），或使用长度归一化的 reward model。OpenAI 的 InstructGPT 论文专门讨论了此问题——他们在 PPO loss 中加入了 `β_1 × log(π_θ(y|x))` 作为额外熵正则项，防止策略坍缩到某个特定的奖励获取模式。

## 9. 面试问题

**Q1: 为什么 RLHF 需要三步（SFT → Reward Model → PPO），而 DPO 只用一步？**
A: RLHF 需要一个显式的奖励模型来评估任意回答，因为 PPO 需要在生成过程中反复评估。DPO 的数学推导发现奖励函数可以隐式地表示为策略的对数比，从而直接用偏好数据优化，绕过奖励模型和强化学习。

**Q2: LoRA 为什么需要两个低秩矩阵 B 和 A 的乘积，而不是一个低秩矩阵？**
A: 单个低秩矩阵 $W_{\text{low}} \in \mathbb{R}^{d \times k}$ 有秩 $\min(d,k)$，无法表达秩为 $r$ 的更新。两个矩阵乘积 $BA$ 的秩至多为 $r$（且初始时 B=0 确保不改变原模型输出），用 $r(d+k)$ 个参数表达 $dk$ 维空间中的 $r$ 秩子空间。

**Q3: DPO 梯度中间项 $\sigma(\hat{r}_l - \hat{r}_w)$ 的含义？**
A: 这是错误项的权重。当模型把"坏回答"评得比"好回答"还高时（$\hat{r}_l > \hat{r}_w$），该项较大（错误越大，惩罚越重）；当模型已经正确区分时，该项趋近 0。

**Q4: QLoRA 的 NF4 和普通 INT4 量化有何区别？**
A: 普通 INT4 是均匀量化（等间距分桶），适合均匀分布。NF4 是信息论最优的非均匀量化，假定权重服从正态分布——在高概率区域量化格点更密集，大幅降低 KL 散度损失。对于正态分布的 LLM 权重，NF4 远优于均匀量化。

**Q5: Prefix-Tuning vs Adapter vs LoRA 各自优劣势？**
A: (1) **Prefix-Tuning**：在每层 K/V 前加可学习 prefix → 推理时无额外延迟（prefix KV 可缓存），但前缀长度是超参，对长上下文不友好。(2) **Adapter**：在 Attention 和 FFN 后加小 bottleneck 层 → 引入了额外推理延迟（因串行）。 (3) **LoRA**：直接在原始权重旁做低秩旁路 → 推理时可合并进原权重，零推理延迟；但只适用于线性层。

**Q6（高难度/FAANG Level）：当你用 DPO 训练模型时，如何诊断"reward over-optimization"（奖励过度优化）？DPO 是否也存在类似 RLHF 的 reward hacking？**
A: DPO 虽然没有显式的 reward model，但"隐式奖励" $\hat{r}(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ 仍然可以被"hack"。当 DPO 过度训练时，$\hat{r}$ 可以增长到非常大（$\pi_\theta(y_w|x) \gg \pi_{\text{ref}}(y_w|x)$），对应 KL 散度发散——模型完全偏离了 reference model 的行为分布。

**诊断方法**（参考 Anthropic 和 alignment 研究社区）：
1. **监控 `logps_chosen - logps_ref` 的均值**：训练健康时该值从 0 开始缓慢上升并趋于稳定（β=0.1 时通常收敛到 0.5-1.5）。如果持续单调上升（超过 3.0），说明策略和参考策略的差异过大，reward over-optimization 正在发生。
2. **监控 KL 散度**：$\text{KL}(\pi_\theta \| \pi_{\text{ref}}) = \mathbb{E}[\log \pi_\theta - \log \pi_{\text{ref}}]$。RLHF 文献建议维持在 5-15 nats 之间。超过 20 nats 时模型回答可能"过于激进"（如过度讨好用户、频繁使用模板化赞美语）。
3. **Golden evaluation set 的趋势**：预留 500 条高质量人工标注的 preference pair 作为评测集。每 500 step 计算 accuracy。如果 accuracy 开始下降而 training loss 仍在下降——这就是 reward over-optimization 的经典信号（Goodhart's Law: "When a measure becomes a target, it ceases to be a good measure."）。
4. **回答多样性度量**：计算生成 token 的 entropy 和 distinct-n（n-gram 去重率）。当模型过度优化 reward 时，它会坍缩为少数高 reward 但同质化的回答模式。entropy < 2.5 且 distinct-3 < 30% 是危险信号。

**缓解策略**：(1) 增大 β（加强 KL 约束，r=0.1 是常用值，r=0.5 更保守），(2) Early stopping based on validation accuracy（而非 training loss），(3) DPO 变体——如 IPO (Identity Preference Optimization) 和 KTO (Kahneman-Tversky Optimization)，分别通过恒等映射和非对称损失来缓解 reward over-optimization。

**Q7（高难度/FAANG Level）：为什么 DPO 在数学上可以绕过奖励模型，直接优化策略？请推导从 RLHF（Bradley-Terry + KL-constrained RL）到 DPO 的完整闭式解。**
A: 这是一个考试中可能要求手推的题目。推导步骤如下：

**Step 1 — RLHF 的 KL-constrained 优化目标**：
$$\max_\pi \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(\cdot|x)} [r(x,y)] - \beta \cdot \text{KL}[\pi(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)]$$

**Step 2 — 求闭式解**：这是一个凸优化问题（最大化期望奖励，同时惩罚与参考策略的偏离）。对 $\pi$ 求变分，加入归一化约束 $\sum_y \pi(y|x) = 1$，得到拉格朗日函数。令泛函导数为 0：
$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \cdot \exp\left(\frac{r(x,y)}{\beta}\right)$$
其中 $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp(r(x,y)/\beta)$ 是配分函数。

**Step 3 — 从策略解出奖励函数**：对上式取 log 并移项：
$$\log \pi^*(y|x) = \log \pi_{\text{ref}}(y|x) + \frac{r(x,y)}{\beta} - \log Z(x)$$
$$r(x,y) = \beta \cdot \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \cdot \log Z(x)$$
这是关键的一步——奖励函数被表达为策略的对数比（加上一个仅依赖 $x$ 的常数）。这意味着在 Bradley-Terry 偏好模型中，$Z(x)$ 项会在相减时消除。

**Step 4 — 代入 Bradley-Terry 偏好模型**：
$$P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$$
将 Step 3 的奖励表达式代入，$Z(x)$ 消去：
$$= \sigma\left(\beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

**Step 5 — 得到 DPO 损失**：最大化偏好概率的对数似然（即负对数似然最小化）：
$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x,y_w,y_l)} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

**核心洞察**：(1) 推导的关键是 RLHF 目标函数有闭式解——这是一个在被动的策略分布与 reference 之间以 reward 为权重的 Boltzmann 分布。(2) $Z(x)$ 的消除不是巧合——Bradley-Terry 模型只比较两个回答的相对奖励，任何仅依赖 $x$ 的项（如 $Z(x)$）在相减时都消除。(3) DPO 的本质是在"策略"空间直接定义偏好损失，绕过了奖励函数这个中间变量——这是受控生成（controlled generation）领域的一个经典技巧（Gumbel-Max SCM 等）的推广。

**Q8（超高难度/Fellow Level）：如何设计一个工业级的 continuous PEFT pipeline，使得一个基础 LLM 可以每天接收用户反馈、自动更新 LoRA adapter，同时不影响在线服务质量？考虑数据飞轮（data flywheel）、分布偏移（distribution shift）和灾难性遗忘（catastrophic forgetting）。**
A: 这是一个涉及 MLOps、推荐系统、LLM fine-tuning 三个领域的交叉问题。完整设计如下：

**架构概览 — "Shadow Deploy + Canary Rollout" 模式**：

**(1) 数据飞轮（Data Flywheel）**
收集两种信号：(a) **显式反馈**：用户点赞/踩、选择重新生成、评分。每天约 0.5-2% 的用户会产生显式反馈。(b) **隐式信号**：用户是否复制了回答、是否追问（说明不满意第一次回答）、停留时长（长→可能在认真读，短→快速跳过=不满意）。这些信号通过简单的规则引擎（如 "阅读时长 < 2秒 → 负样本" + "复制了内容 → 正样本"）自动生成 preference pair。每天从 100 万次对话中可提取约 5000-20000 条 preference pair，足够做一次 DPO/LoRA 更新。

**(2) Cold-Start: Warm LoRA Initialization**
第一版 adapter 使用 curated human preference data（约 5000-10000 条高质量标注，成本 $10000-20000）。此后每天增量更新。

**(3) Continuous Training Pipeline**
- **午夜批处理**（低流量时段）：从数据湖拉取前一天的 feedback，转化为 preference pair。
- **Offline evaluation**：在 reserved golden eval set 上评估新 checkpoint 的 win rate vs 当前生产模型。只有 win rate > 50% 且 KL divergence < 10 nats 才放行。
- **Shadow deploy**：新 LoRA adapter 部署到 1% 流量（randomly sampled），运行 2 小时，监控 P99 延迟、PPL、用户 retention 等核心指标。A/B test 框架（如 PlanOut 或内部实验平台）对比实验组 vs 对照组。
- **Canary rollout**：如果 shadow 阶段指标正常（P99 延迟不增加 >5%，用户 retention 不下降），在 2 天内逐步放量：1% → 10% → 50% → 100%。

**(4) 分布偏移监测（Distribution Shift Detection）**
这是最容易出错的部分。使用两个检测器：
- **KL-based drift detector**：$\text{KL}(p_{\text{新 adapter}}(y|x) \| p_{\text{当前 adapter}}(y|x))$，在 holdout set 上计算。阈值 > 5 nats 触发告警（说明模型行为发生了剧烈变化）。
- **Embedding drift detector**：用 sentence-transformer（如 all-MiniLM-L6-v2）分别对 old model 和 new model 对同一批 query 的回答做 embedding，计算两个 embedding 集合之间的 Maximum Mean Discrepancy (MMD)。MMD > 0.05 触发告警（说明回答的语义分布发生了变化）。
- 如果任一检测器触发告警，自动回滚到上一个稳定 checkpoint。

**(5) 灾难性遗忘防护（Catastrophic Forgetting Prevention）**
每天的新 preference pair 中混合 10-20% 的"anchor data"——固定保留的 high-quality 通用任务样本（如 MMLU 子集、GSM8K 数学题、TruthfulQA 事实问答），确保模型在学会新偏好时不忘记基本能力。训练时使用 Elastic Weight Consolidation (EWC) 的简化版：在 DPO loss 中加入一个二次惩罚项 $\lambda \cdot \sum_i F_i (\theta_i - \theta_i^{\text{ref}})^2$，其中 $F_i$ 是 Fisher Information Matrix 的对角元素（由 anchor data 计算），对"重要"参数施加更大的不动惩罚。

**(6) Rollback & Safety**
每次部署新 adapter 前，在 GPU workspace 中至少保留前 3 个版本的 adapter。如果 P0 告警触发（如内容安全团队发现模型开始生成有害内容），可以在 <30 秒内回滚到上一个版本（vLLM `remove_lora` + `add_lora` API）。vLLM 的 punica workspace 设计天然支持这种热切换——不需要重启服务。

**已知的实践者**：Google Bard/Gemini、Anthropic Claude 和 Perplexity AI 都在不同程度使用类似 pipeline（Perplexity 公开讨论过其 A/B testing 系统用于模型更新）。关键成本：每天的训练仅需单卡 A100 运行 30-60 分钟（LoRA r=16, 5000-20000 条数据），成本约 $1-2/天。

## 10. 本讲总结

本讲覆盖了 LLM 后训练的两大核心：

**对齐（Alignment）**从 SFT → RLHF (PPO) → DPO 的演进，反映了从"多阶段复杂管线"到"简洁端到端"的趋势。DPO 最漂亮的地方在于——它用数学证明了奖励函数的"中间人"角色可以被消去，直接让策略从偏好数据学习。

**高效微调（PEFT）**的核心洞察是："适应一个任务"对参数的改变具有低秩性质。LoRA 利用这一点将微调参数量压缩 100-1000x；QLoRA 进一步在 4-bit 量化模型上做 LoRA，使消费级 GPU 微调大模型成为现实。Adapter、Prefix-Tuning、BitDelta 提供了不同场景下的替代方案，各有牺牲——灵活性与延迟、参数量与质量的权衡。

**实践建议**：
- 快速对齐实验 → DPO + LoRA（最简单）
- 追求极致质量 → RLHF + QLoRA（最强大但管线复杂）
- 多任务生产部署 → Adapter 热插拔
- 边缘模型更新 → BitDelta（1-bit OTA）

下一讲将面对 LLM 的"上下文困局"——当需要处理 128K tokens 的长文本时，$O(n^2)$ 的 Attention 如何突围？

## 11. 工业落地checklist

| 检查项 | 说明 | 不做的后果 |
|--------|------|-----------|
| LoRA rank 不要超过 64——大多数任务 r=4-8 已饱和 | LoRA 原论文和字节实践：r=512 时 adapter 从 50MB 膨胀到 3.2GB，且低秩正则化消失导致 overfitting | 占用 12% 原模型参数却 PPL 反升（4.2→6.7），参数效率完全丧失 |
| DPO 训练时 reference model 必须冻结且 log-prob 提前缓存 | 某公司失误：ref model 设为可训练 → log(π_θ/π_ref) 始终 ≈ 0 → 零偏好信号学习，浪费 2 周 $8000 算力 | 训练完全无效但 loss 正常下降（迷惑性强），排查 4 天才发现根因 |
| QLoRA 训练超过 3 epoch 时必须每 2 epoch 重算 NF4 quantization parameters | 某研究所 LLaMA-65B 10 epoch 训练：第 6 epoch 起 NF4 dequantization 累积漂移，某些 channel 的 scale 偏离 12% | loss 突然跳跃（1.2→2.8），训练不稳定，长 epoch 训练白费 |
| RLHF PPO 必须监控 reward hacking——模型可能学会"写废话拿高分" | 某公司 reward model 对长度有正偏置 → PPO 策略在第 3 天学到生成 2000+ token 冗长回复 → 可读性从 4.2/5 降到 2.8/5 | 模型行为完全偏离预期，用户满意度反向下降 |
| DPO 训练必须监控隐式 reward 的均值曲线和 KL 散度 | 健康训练：logps_chosen - logps_ref 从 0 缓慢升至 0.5-1.5 并稳定；持续 > 3.0 说明 reward over-optimization（Goodhart's Law） | 模型回答模式坍缩为少数高分模板，多样性和用户体验丧失 |
| 多 LoRA adapter 服务时注意 KV head 一致性：不同 adapter 只在 V/O projection + FFN 上做 LoRA | vLLM 实践：如果 LoRA 应用在 q_proj 上，不同 adapter 的 Q 不同 → 同一 batch 内 KV Cache 不能共享 → 内存翻倍 | 多租户并发吞吐下降 50%，GPU 利用率从 82% 跌到 45% |
| BitDelta 1-bit 模型更新下发前必须验证更新前后模型在 holdout set 上的行为一致性 | BitDelta 只传正负号丢失幅度信息，极端情况下某些 channel 的更新方向被反转（sign(ΔW) 因数值噪声改变） | OTA 更新后模型在某些边缘 case 上行为突变，用户投诉升级 |
