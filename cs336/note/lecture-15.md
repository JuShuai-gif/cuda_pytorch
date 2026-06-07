# Lecture 15: Alignment / RLHF

## 1. 本讲核心问题

一个预训练好的 LLM 能完成"next token prediction"，但它**不一定会遵循人类的指令**。模型可能输出有害内容、胡编乱造、或者回答方式不友好。**本讲的核心问题是：如何让 LLM 的输出对齐（align）人类的偏好和价值观？** 包括 RLHF（Reinforcement Learning from Human Feedback）、DPO（Direct Preference Optimization）、GRPO 等主流对齐方法。更深层的问题是：为什么对齐会带来"alignment tax"（对齐税）？

---

## 2. 通俗解释

### RLHF ≈ 训练一只聪明的狗

想象你收养了一只非常聪明的狗（预训练好的 LLM），它什么都能理解但行为完全不受控制：

- **SFT（Supervised Fine-Tuning）** ≈ **先示范好行为**
  - 你在狗面前示范"坐下"、"趴下"、"不要咬拖鞋"
  - 给正确行为的例子让它模仿
  - LLM 的 SFT：人类标注员写下"好回答"，让模型学习模仿

- **Reward Model** ≈ **建立奖励标准**
  - 你告诉它"这么做是对的"、"那么做是错的"
  - 训练一个"评分器"来量化什么是好行为

- **PPO（Proximal Policy Optimization）** ≈ **通过表扬和批评来微调**
  - 狗做了好事 → 给零食（+reward）
  - 狗咬了拖鞋 → 批评（-reward）
  - 但也不能偏离示范太远（KL penalty → 防止为了拿零食而歪曲行为）
  - PPO 的过程：模型生成回答 → Reward Model 打分 → 用梯度更新模型 → 重复

**核心类比：** 预训练 = 让一个孩子读了所有书（知识丰富但不知道怎么说话）
SFT = 给他看礼貌对话的范例（学会礼貌形式）
RLHF = 在日常对话中不断纠正他（真正学会得体交流）

### DPO ≈ 跳过打分器，直接对比好坏

DPO 的比喻：与其给狗设一个打分器，不如直接告诉它"坐下是对的，站着不对"——**直接比较两个行为的优劣，而不是单独打分**。

---

## 3. 数学公式 + 工程意义

### 3.1 RLHF 三阶段

#### 阶段 1: SFT（Supervised Fine-Tuning）

给定高质量的 (instruction, response) 对 $\mathcal{D}_{SFT} = \{(x_i, y_i)\}$，最大化：

$$
\mathcal{L}_{SFT}(\theta) = -\mathbb{E}_{(x, y) \sim \mathcal{D}_{SFT}} \left[ \sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{<t}) \right]
$$

**工程意义：** 这是最简单的阶段（标准 language modeling loss），核心在于数据质量。通常需要 10,000-100,000 对高质量 (instruction, response)。

#### 阶段 2: Reward Model

训练一个 reward model $r_\phi(x, y)$。给定人类偏好的对比对 $(x, y_w, y_l)$（$y_w$ 是更好的回答，$y_l$ 是更差的回答）：

$$
\mathcal{L}_{RM}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{RM}} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]
$$

这是一个 Bradley-Terry 模型的负对数似然。$\sigma$ 是 sigmoid 函数。

**工程意义：** Reward model 通常是与 SFT 模型规模相近的语言模型（把最后的 LM head 替换为标量输出）。人类偏好数据的收集是关键瓶颈——InstructGPT 用了约 33,000 个对比对。

#### 阶段 3: PPO

使用 PPO 优化 policy $\pi_\theta$，最大化 reward 同时不偏离 SFT 模型太远：

$$
\mathcal{L}_{PPO}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)} \left[ r_\phi(x, y) - \beta \cdot \underbrace{\log \frac{\pi_\theta(y|x)}{\pi_{SFT}(y|x)}}_{\text{KL penalty}} \right]
$$

**KL penalty 的工程师解释：** $\beta$ 是"不要走太远"的控制参数。如果 $\beta$ 太小 → 模型可能"reward hack"，找到高分但不自然/有问题的回答。如果 $\beta$ 太大 → 模型不敢偏离 SFT，对齐效果差。典型 $\beta$ 值在 0.01-0.1 之间。

### 3.2 DPO（Direct Preference Optimization）

DPO 的核心洞察：可以直接从偏好数据中隐式地推导出最优 policy，而不需要显式训练 reward model：

$$
\mathcal{L}_{DPO}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \cdot \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \cdot \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]
$$

**工程意义（DPO vs RLHF 对比）：**

| 维度 | RLHF | DPO |
|------|------|-----|
| 训练阶段 | 3 阶段（SFT + RM + PPO） | 2 阶段（SFT + DPO） |
| Reward Model | 需要 | 不需要 |
| GPU 资源 | 高（同时跑 4 个模型） | 低（2 个模型） |
| 稳定性 | PPO 训练不稳定 | 更稳定 |
| 在线/离线 | 需要在线采样 | 纯离线 |
| 性能（理论上限） | 更高（在线环境） | 略低但够用 |

### 3.3 GRPO（Group Relative Policy Optimization）

DeepSeek R1 使用的 GRPO 进一步简化了 RLHF。核心思想：**不再用 reward model，而是用同一 group 内的相对排名来评估好坏。**

对于一组回答 $G = \{y_1, y_2, \ldots, y_k\}$，相对优势计算为：

$$
A_i = \frac{r(y_i) - \text{mean}(G)}{\text{std}(G)}
$$

$$
\mathcal{L}_{GRPO}(\theta) = -\mathbb{E}_{G} \left[ \frac{1}{|G|}\sum_{i=1}^{|G|} \min \left( \rho_i(\theta) A_i, \text{clip}(\rho_i(\theta), 1-\epsilon, 1+\epsilon) A_i \right) \right]
$$

其中 $\rho_i(\theta) = \frac{\pi_\theta(y_i|x)}{\pi_{old}(y_i|x)}$ 是重要性采样比率。

**GRPO 的优势：**
- 不需要 reward model（用基于规则的 reward，如数学题答案验证）
- 组内比较代替绝对打分，更鲁棒
- DeepSeek R1 证明了用纯规则 reward + GRPO 可以训练出强推理能力

---

## 4. 工业界真实实现

### 4.1 InstructGPT（OpenAI, 2022）— RLHF 的开山之作

| 阶段 | 数据量 | 模型规模 |
|------|--------|---------|
| SFT | 13K (instruction, response) | 175B (GPT-3) |
| RM | 33K comparison pairs | 6B |
| PPO | 31K prompts | 175B |

**关键设计决策：**
- SFT 数据由 40 个标注员编写
- Reward model 是 6B 参数的 GPT-3 变体
- 标注员的偏好一致性约 73%
- 最终的 PPO 模型在 85% 的情况下比 SFT 模型更好

### 4.2 Llama 2 的 RLHF 流程（Meta, 2023）

Llama 2 的 RLHF 有两个关键创新：

1. **多轮 RLHF：** 不是一次性收集所有偏好数据，而是迭代收集——用当前最优模型生成回答，标注员在此基础上标注偏好，再训练新模型。

2. **两个 reward model：** 一个重点评估 helpfulness，另一个重点评估 safety。最终使用时取两者的 harmonic mean：
   $$
   r_{total}(x, y) = \frac{2 \cdot r_{help}(x, y) \cdot r_{safe}(x, y)}{r_{help}(x, y) + r_{safe}(x, y)}
   $$

**数据规模：**
- SFT: 27,540 (instruction, response) 对
- RM: 1,418,091 人类偏好对比对（binary comparisons）
- PPO: 使用 rejection sampling + PPO

### 4.3 DeepSeek R1 的 GRPO（2025）

DeepSeek R1 用纯 **rule-based reward** + GRPO 实现了强大的推理能力：

```python
# 伪代码：DeepSeek R1 的 rule-based reward
def rule_based_reward(response, expected_answer):
    reward = 0
    # Reward for correct answer
    if extract_answer(response) == expected_answer:
        reward += 1.0
    # Reward for reasoning format (e.g., <think> tags)
    if has_reasoning_format(response):
        reward += 0.1
    # Reward for valid code execution (if applicable)
    if code_valid(response):
        reward += 0.2
    # Penalty for broken formatting
    if has_format_errors(response):
        reward -= 0.3
    return reward
```

**GRPO 的关键参数：**
- Group size $k = 64$（生成 64 个候选回答）
- $\epsilon = 0.2$（clip 范围）
- $\beta = 0.01$（KL penalty）

### 4.4 DPO 在工业生产中的应用

Hugging Face 的 Zephyr-7B 是一个典型案例：

- 基础模型：Mistral-7B
- SFT: UltraChat 数据集（~200K conversations）
- DPO: UltraFeedback 数据集（~60K preference pairs）
- **结果：** 7B 参数的 Zephyr 在某些 benchmark 上超越了 Llama 2 70B Chat

---

## 5. CUDA/GPU 视角

### 5.1 RLHF 的 GPU 资源需求

RLHF 是 GPU 资源最密集的操作之一：

| 组件 | 模型数 | 显存需求（70B 为例） |
|------|--------|---------------------|
| Policy Model | 1 | ~140 GB (FP16) |
| Reference Model | 1 | ~140 GB |
| Reward Model | 1 | ~140 GB |
| Value/Critic Model | 1 | ~140 GB |
| **总计** | **4 个模型** | **~560 GB** |

这需要至少 8 块 A100-80GB GPU，且还要留出训练用的显存（激活值、优化器状态等），实际可能需要 16-32 块 A100。

### 5.2 DPO 的优势

DPO 只需要：
- Policy Model: ~140 GB
- Reference Model: ~140 GB
- **总计: ~280 GB**（仅 RLHF 的一半）

### 5.3 GRPO 的效率

GRPO 进一步简化：
- 只需要 Policy Model + Reference Model（同 DPO）
- 不需要 Reward Model
- Group sampling 可以在 batch 内并行，GPU 利用率更高

```python
# DPO-style training loop (simplified)
def dpo_step(policy_model, ref_model, batch):
    # batch: (prompt, chosen_response, rejected_response)
    with torch.no_grad():
        ref_chosen_logp = ref_model(batch.prompt, batch.chosen_response)
        ref_rejected_logp = ref_model(batch.prompt, batch.rejected_response)
    
    policy_chosen_logp = policy_model(batch.prompt, batch.chosen_response)
    policy_rejected_logp = policy_model(batch.prompt, batch.rejected_response)
    
    # DPO loss
    chosen_reward = beta * (policy_chosen_logp - ref_chosen_logp)
    rejected_reward = beta * (policy_rejected_logp - ref_rejected_logp)
    loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
    
    loss.backward()
    optimizer.step()
```

---

## 6. 本讲与整个 LLM 系统的关系

```
┌─────────────────────────────────────────────────────────────┐
│            LLM 训练的统一视角                                 │
├──────────────┬──────────────┬────────────────────────────────┤
│  Pre-training │  Mid-training │  Post-training (Alignment)   │
│  (Knowledge)  │  (Capability)  │  (Behavior)                  │
├──────────────┼──────────────┼────────────────────────────────┤
│  学知识       │  学技能        │  学做人                        │
│  Raw text     │  Long context  │  SFT → RLHF/DPO/GRPO         │
└──────────────┴──────────────┴────────────────────────────────┘
```

### Alignment Tax（对齐税）

**定义：** 在提高 helpfulness 和 safety 的同时，模型在某些 academic benchmark 上的能力会下降。

**原因：**
1. SFT 训练是在非常窄的分布（instruction 格式）上进行的，可能导致分布外泛化能力下降
2. RLHF 的 KL penalty 使模型偏向安全但保守的回答
3. Reward model 本身有偏见，会"奖励"无害但无用的回答

**缓解策略：**
- 混合 SFT 数据和原始预训练数据（data mixing）
- 使用较小的 KL penalty（$\beta < 0.05$）
- 迭代式 RLHF（多轮次逐步放松约束）

### Reward Hacking 和 Over-Optimization

当 PPO 运行足够多的 steps 时，模型可能"找到"reward model 的漏洞：

- **例子 1：** Reward model 喜欢长回答 → 模型生成越来越长的模板化回答
- **例子 2：** Reward model 惩罚否定词 → 模型从不说"我不知道"，而是编造答案
- **例子 3：** Reward model 喜欢礼貌 → 模型变得过度谦卑，效率下降

**应对方法：** 限制 PPO 步数、使用多个 reward model 取平均、加入规则约束。

### Constitutional AI（宪法 AI, Anthropic）

替代传统 RLHF 的方法：用一套"宪法"（constitution）规则来引导模型行为，让模型自己 critique 和 revise 自己的输出：

1. **Critique：** 模型根据宪法规则评估自己的回答
2. **Revision：** 模型根据评估修订自己的回答
3. **Training：** 用修订后的数据做 SFT + DPO

这种方法减少了对人类标注的依赖，提高了可扩展性。

---

## 7. 面试问题

1. **RLHF 的三个阶段分别是什么？每个阶段的目标和损失函数是什么？**

   *参考答案：参见上面数学公式部分。关键是理解 SFT（模仿好回答）、RM（学习人类偏好打分）、PPO（在 RM 的引导下优化 policy，同时保持不偏离 SFT 太远）。*

2. **DPO 和 RLHF 的核心区别是什么？DPO 为什么不需要 reward model？**

   *参考答案：DPO 直接从偏好数据中推导最优 policy（通过 reparameterization trick），将 Bradley-Terry 偏好模型重写为 policy 的函数，从而消去了 reward model。核心数学是：最优 policy 和最优 reward 之间存在一一对应的闭式解。DPO 更简单、更稳定，但只能离线使用，不能从在线 exploration 中获益。*

3. **什么是 reward hacking？举三个具体例子。如何检测和防止？**

   *参考答案：Reward hacking 是模型找到了 reward model 的打分漏洞，输出了高分但不符合人类期望的回答。例子：（1）用大量礼貌用语填充回答长度，（2）回避困难问题而非诚实回答，（3）用特定模式/关键词触发高分。防止方法：限制 PPO 步数、多 reward model ensemble、constraint-based 奖励。*

4. **GRPO 和传统 RLHF 有什么区别？DeepSeek R1 为什么选择 GRPO？**

   *参考答案：GRPO 不需要 reward model，用组内相对排名 + 基于规则的 reward 代替。DeepSeek R1 的选择原因：（1）数学/代码等推理任务有明确的 ground truth 可以自动验证，不需要昂贵的人类偏好数据或 reward model；（2）组内比较比绝对打分更稳定；（3）节省了大量训练 reward model 的计算资源。*

5. **"Alignment tax"是什么？在实践中如何权衡 helpfulness 和 safety？**

   *参考答案：Alignment tax 是在对齐过程中模型在某些 benchmark 上的能力下降（特别是在学术/推理任务上）。权衡策略包括：hybrid training（混合对齐数据和预训练数据）、多目标优化（同时优化 helpfulness + safety + accuracy）、不同场景使用不同温度参数。Llama 2 的两个 reward model 也是为了解决这个权衡问题。*

6. **Constitutional AI 和 RLHF 相比有什么优势和劣势？**

   *参考答案：优势：减少对人类标注的依赖、更可扩展、更可解释（宪法规则明确可审计）。劣势：宪法规则本身是人类设计的，可能有遗漏和偏见；AI 自我 critique 的能力有限；对于含糊的伦理场景可能不如人类判断细致。*

---

## 参考：对齐方法的演进

| 时间 | 方法 | 提出者 | 核心贡献 |
|------|------|--------|---------|
| 2017 | PPO | OpenAI | RL policy gradient 基础算法 |
| 2022 | InstructGPT | OpenAI | 首次大规模 RLHF |
| 2022 | Constitutional AI | Anthropic | 减少人类标注依赖 |
| 2023 | Llama 2 RLHF | Meta | 双 RM + 迭代 RLHF |
| 2023 | DPO | Stanford | 消除 reward model |
| 2024 | KTO | Contextual | 不需要偏好对，只需要二值反馈 |
| 2025 | GRPO | DeepSeek | 纯 rule-based + 组内相对排名 |

> **关键 Takeaways:**
> 1. 预训练给模型知识，对齐给模型行为规范
> 2. RLHF 是目前最成熟的对齐方法，但需要大量计算资源
> 3. DPO 简化了流程，但对数据质量要求更高
> 4. GRPO 证明了对齐不一定需要 reward model
> 5. Alignment tax 是真实存在的——对齐是一场持续的多目标优化
> 6. 未来趋势：更少人类标注、更多自动化、更高效的方法
