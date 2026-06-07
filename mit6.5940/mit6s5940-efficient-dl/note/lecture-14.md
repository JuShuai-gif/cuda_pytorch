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
