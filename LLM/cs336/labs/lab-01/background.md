# 背景知识：BPE、Cross-Entropy Loss 与 Perplexity

## 1. Byte-Pair Encoding (BPE)

### 1.1 为什么需要 Tokenization？

语言模型无法直接处理原始文本字符，需要将文本转换为模型可处理的数字序列（token IDs）。好的 tokenization 方案需要平衡：

- **vocab size**：太小 → 每个 token 承载信息少，序列变长；太大 → embedding 矩阵过大
- **覆盖率**：必须能处理任意输入（包括从未见过的字符）
- **效率**：常见词应被编码为少量 token

### 1.2 BPE 算法原理

BPE (Byte-Pair Encoding) 是一种 data-driven 的子词分割算法，核心思想是：

> 从字符级别开始，反复将最常共现的 token pair 合并为一个新 token。

**训练流程：**

```
输入: 训练语料 corpus, 目标 vocab size V
输出: merge 规则列表

1. 将语料中的每个词表示为字符序列，末尾加特殊符号 </w>
   "hello" → "h e l l o </w>"

2. 统计所有相邻 token pair 的出现频率

3. 找到频率最高的 pair，将其合并
   ("e", "l") → "el"

4. 将新合并的 token 加入 vocab

5. 重复步骤 2-4，直到 vocab size 达到 V

6. 保存所有 merge 规则（按顺序）
```

**编码 (Encoding) 流程：**

```
输入: 文本字符串
输出: token ID 序列

1. 将文本拆分为字符序列
2. 按 merge 规则的顺序，依次尝试合并相邻的 token pair
3. 将最终的 token 映射到对应的 ID
```

**解码 (Decoding) 流程：**

```
输入: token ID 序列
输出: 文本字符串

1. 将每个 token ID 映射回 token 字符串
2. 将所有 token 连接起来
3. 移除 </w> 标记，恢复原始文本
```

### 1.3 BPE 的数学形式

设 vocab 为 $V = \{v_1, v_2, ..., v_n\}$，语料为 $C$。

BPE 的目标是学习一个 merge 操作序列 $\mathcal{M} = (m_1, m_2, ..., m_k)$，使得：

$$m_t = \arg\max_{(a,b) \in V_t \times V_t} \text{freq}(a,b)$$

其中 $\text{freq}(a,b)$ 是 token pair $(a,b)$ 在语料中的出现次数。

### 1.4 GPT 系列使用的 Tokenization

| 模型     | Tokenization | Vocab Size | 特点                           |
| -------- | ------------ | ---------- | ------------------------------ |
| GPT-1    | BPE          | ~40,000    | 基础 BPE                       |
| GPT-2    | BPE + 改进   | 50,257     | 禁止跨字符类别的 merge         |
| GPT-3/4  | BPE (tiktoken) | 100,000+ | 使用 tiktoken 库，高效的 Rust 实现 |

---

## 2. Cross-Entropy Loss

### 2.1 信息论基础

在语言模型中，我们希望模型输出的概率分布 $p_\theta(y|x)$ 尽可能接近真实分布 $p(y|x)$。

**Cross-Entropy** 衡量两个分布之间的"距离"：

$$H(p, q) = -\sum_{i} p(i) \log q(i)$$

### 2.2 语言模型中的 Cross-Entropy

对于语言建模任务，给定上下文 $x_{<t}$，模型预测下一个 token 的概率分布 $p_\theta(\cdot | x_{<t})$。

设真实的下一个 token 为 $x_t$（one-hot 分布），则 cross-entropy loss 为：

$$\mathcal{L}_{CE} = -\log p_\theta(x_t | x_{<t})$$

对于长度为 $N$ 的序列，平均 loss 为：

$$\mathcal{L} = -\frac{1}{N} \sum_{t=1}^{N} \log p_\theta(x_t | x_{<t})$$

### 2.3 与 Softmax 的关系

模型输出的 logits $z \in \mathbb{R}^{|V|}$ 经过 softmax 转换为概率：

$$p_\theta(x_t | x_{<t}) = \frac{\exp(z_{x_t})}{\sum_{j} \exp(z_j)}$$

因此：

$$\mathcal{L}_{CE} = -z_{x_t} + \log\sum_{j} \exp(z_j)$$

这就是 PyTorch 中 `F.cross_entropy` 的实现（log_softmax + nll_loss）。

### 2.4 数值稳定性

直接实现 softmax 容易溢出。实践中的标准做法是 log-sum-exp trick：

$$\log\sum_{j} \exp(z_j) = m + \log\sum_{j} \exp(z_j - m)$$

其中 $m = \max_j z_j$。

---

## 3. Perplexity

### 3.1 定义

Perplexity (困惑度) 是语言模型最常用的评价指标，定义为：

$$\text{PPL} = \exp\left(-\frac{1}{N} \sum_{t=1}^{N} \log p_\theta(x_t | x_{<t})\right) = \exp(\mathcal{L})$$

### 3.2 直观理解

- Perplexity 可以理解为：模型在每个位置需要从多少个等概率候选中选择
- PPL = 1：完美模型（每步都确定性地预测正确）
- PPL = |V|：随机猜（均匀分布）
- PPL 越低越好

### 3.3 为什么使用 Perplexity？

1. **可比性**：不同 vocab size 的模型可以用 PPL 比较（理论上）
2. **直观性**：PPL 比 raw loss 更容易理解
3. **单调性**：loss 降低 ↔ PPL 降低

### 3.4 FlashAttention 对 PPL 的影响

FlashAttention 在数学上与标准 attention 等价，因此：

- 正确实现的 FlashAttention **不会改变** PPL
- 如果 PPL 不一致，说明实现有 bug

这是验证实现正确性的重要方法！

---

## 4. 核心公式速查

| 公式              | 表达式                                                            | 含义              |
| ----------------- | ----------------------------------------------------------------- | ----------------- |
| BPE merge 选择    | $m_t = \arg\max_{(a,b)} \text{freq}(a,b)$                         | 选最常见 pair     |
| Cross-entropy     | $\mathcal{L} = -\frac{1}{N}\sum \log p_\theta(x_t\|x_{<t})$      | 平均负对数似然    |
| Softmax           | $p_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}$                        | logits → 概率     |
| Log-sum-exp       | $\text{LSE}(z) = \max(z) + \log\sum\exp(z - \max(z))$            | 数值稳定的求和    |
| Perplexity        | $\text{PPL} = \exp(\mathcal{L})$                                  | 模型困惑度        |
