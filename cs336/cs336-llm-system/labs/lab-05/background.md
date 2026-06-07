# 背景知识：Data Pipeline 与 DPO / RLHF

## 1. LLM 训练数据 Pipeline

### 1.1 全流程概览

```
Raw Web Crawl (CommonCrawl)
    │
    ▼
[1] Text Extraction        ← 从 HTML/WARC 中提取纯文本
    │
    ▼
[2] Language Detection     ← FastText / CLD3 识别语言
    │
    ▼
[3] Quality Filtering      ← 长度、重复率、困惑度等过滤
    │
    ▼
[4] De-duplication         ← MinHash / SimHash 近似去重
    │
    ▼
[5] PII Removal            ← 移除邮箱、电话号码等敏感信息
    │
    ▼
[6] Data Mixing            ← 按比例混合多个数据源
    │
    ▼
[7] Tokenization           ← BPE / SentencePiece
    │
    ▼
[8] Packing / Batching     ← 将样本打包成训练序列
```

### 1.2 数据过滤规则

#### 基于规则的过滤

| 规则                  | 实现                         | 目的                       |
| --------------------- | ---------------------------- | -------------------------- |
| 最小长度              | `len(text) > min_chars`      | 去除过短的无意义文本       |
| 最大长度              | `len(text) < max_chars`      | 防止训练序列过长           |
| 单词重复率            | `max_word_freq / total_words` | 去除 spam / 机器生成文本   |
| 特殊字符比例          | `count(special) / len(text)` | 去除乱码                   |
| 平均词长              | `total_chars / total_words`  | 去除异常短的"词"           |
| 段落数                | `len(paragraphs) > min_para` | 确保文本有结构             |

#### 基于模型的过滤 (FineWeb-style)

| 方法                  | 原理                                    |
| --------------------- | --------------------------------------- |
| 教育价值分类器        | 用 small LM 打分（如 KenLM perplexity） |
| 重复 n-gram 比例      | 检测机器生成文本（通常重复率高）        |
| 文档质量分数          | 综合多个 heuristics 的加权分数          |

### 1.3 去重方法

| 方法           | 原理                                   | 特点                     |
| -------------- | -------------------------------------- | ------------------------ |
| **Exact dedup** | SHA256 hash 精确匹配                   | 准确但只能找到完全相同的 |
| **MinHash**    | 对 n-gram 集合做 min-hash 签名         | 可找近似重复，适合大规模 |
| **SimHash**    | 对文档做 locality-sensitive hash       | Google 使用，支持海明距离 |
| **URL dedup**  | 相同 URL 只保留一个                    | 简单有效                 |

**MinHash 原理简述：**

1. 将文档的 n-gram 集合映射为多个 hash 签名
2. 使用多个 hash function，取最小值作为签名
3. 将签名分成 bands，相同 band 内签名相同的文档视为近似重复

### 1.4 数据混合 (Data Mixing)

不同数据源按比例混合：

$$\text{mix\_weight}_i = \frac{\text{sample\_count}_i}{\text{total\_samples}}$$

常见的数据源配比（以 LLaMA 为例）：

| 数据源        | 比例   |
| ------------- | ------ |
| CommonCrawl   | 67.0%  |
| C4            | 15.0%  |
| GitHub        | 4.5%   |
| Wikipedia     | 4.5%   |
| Books         | 4.5%   |
| ArXiv         | 2.5%   |
| StackExchange | 2.0%   |

---

## 2. Alignment: RLHF

### 2.1 RLHF 三步流程

```
Step 1: Supervised Fine-Tuning (SFT)
  预训练 LLM → fine-tune on high-quality instruction-response pairs
  目的：让模型学会遵循指令格式

Step 2: Reward Model Training
  对同一 prompt 采样多个 response，人工标注偏好
  训练一个 reward model r_φ(x, y) 来预测人类偏好
  通常使用 Bradley-Terry model:
    P(y_w ≻ y_l | x) = σ(r(x, y_w) - r(x, y_l))

Step 3: PPO Fine-Tuning
  使用 reward model 通过 PPO 优化策略：
    max E[r_φ(x, y)] - β KL(π_θ || π_ref)
  其中 KL 惩罚项防止模型偏离太远（保持通用能力）
```

### 2.2 RLHF 的问题

1. **复杂**：需要训练和维护单独的 reward model
2. **不稳定**：PPO 训练对超参数敏感
3. **Reward hacking**：模型可能学会欺骗 reward model

---

## 3. Alignment: DPO (Direct Preference Optimization)

### 3.1 核心思路

DPO 的核心洞察：**Reward model 可以隐式地用策略模型表示**。

在 RLHF 的最优策略下（constrained optimization 的 closed-form 解）：

$$r(x, y) = \beta \log\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

将 reward 代入 Bradley-Terry preference model，$Z(x)$ 项抵消，得到：

$$\mathcal{L}_{\text{DPO}} = -\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

### 3.2 数学推导

从 RLHF 的优化目标出发：

$$\max_\theta \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta}\left[r(x, y)\right] - \beta \cdot \text{KL}\left(\pi_\theta \| \pi_{\text{ref}}\right)$$

最优策略的 closed-form：

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)$$

反解出 reward：

$$r(x, y) = \beta \log\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

代入 BT preference model：

$$P(y_w \succ y_l | x) = \sigma\left(r(x, y_w) - r(x, y_l)\right)$$

$Z(x)$ 抵消！直接得到 DPO loss。

### 3.3 DPO vs RLHF 对比

| 维度           | RLHF                     | DPO                       |
| -------------- | ------------------------ | ------------------------- |
| 步骤           | 3 步 (SFT+RM+PPO)        | 2 步 (SFT+DPO)            |
| 需要 reward model | 是                       | 否                        |
| 训练稳定性     | PPO 不稳定               | 稳定（标准 CE loss）       |
| 计算开销       | 高（4个模型在内存中）     | 低（2个模型）              |
| 数据效率       | 较高（reward 可以泛化）  | 需大量偏好数据             |
| 在线/离线      | 可以在线采样              | 通常是离线                 |
| 理论保证       | 较弱                     | 更强（closed-form 对应）   |

### 3.4 GRPO (DeepSeek)

GRPO (Group Relative Policy Optimization) 是 DeepSeek 提出的变体：

- 对同一 prompt 采样 group of responses
- 使用组内相对排名 (relative ranking) 而非绝对 reward
- 不需要 reward model，也不需要 reference model
- 更简单、更高效

$$\mathcal{L}_{\text{GRPO}} = -\mathbb{E}\left[\sum_i \frac{\exp(A_i / \tau)}{\sum_j \exp(A_j / \tau)} \cdot \mathbb{I}[y_i \text{ is best}]\right]$$

其中 $A_i$ 是 response $y_i$ 在 group 内的 advantage。

---

## 4. 核心公式速查

| 公式                           | 含义                        |
| ------------------------------ | --------------------------- |
| $\mathcal{L}_{\text{DPO}}$     | DPO loss function           |
| $\sigma(\cdot)$                | sigmoid function            |
| $\beta$                        | DPO temperature             |
| $\pi_{\text{ref}}$             | Reference (frozen) model    |
| $y_w \succ y_l$                | Preference: $y_w$ better than $y_l$ |
| $\text{KL}(\pi\|\pi_{\text{ref}})$ | KL divergence               |
