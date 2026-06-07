# Lecture 12: Transformers and LLM — Attention 机制与架构全景

## 1. 本讲核心问题

大型语言模型（LLM）如 GPT-4、LLaMA-3 为何能达到数十亿甚至数千亿参数？Transformers 架构的 Self-Attention 机制到底在做什么？为什么 LLM 在推理阶段如此耗费计算和内存——Attention 的时间/空间复杂度为何是 $O(n^2)$，KV Cache 为何爆炸式增长？Multi-Head Attention、Multi-Query Attention（MQA）、Grouped-Query Attention（GQA）这些变体分别解决什么问题？RoPE 旋转位置编码为何成为现代 LLM 的标准选择？Scaling Laws 告诉我们什么？

## 2. 通俗解释

可以把 Self-Attention 想象成一场高效的会议：每个词（token）都在听其他所有词说话，然后决定自己应该说什么。比如句子"The cat sat on the mat"，当模型处理"sat"时，它会重点关注"The cat"（谁在坐）和"on the mat"（坐在哪里），给相关词更高的"注意力分数"。

**Multi-Head Attention** 就像同时开多场并行会议，每个"头"关注不同的语言特征——一个头关注语法结构，另一个关注语义关联，还有一个关注位置关系。

**KV Cache** 的动机很简单：当你逐词生成回答时（自回归生成），已经算过的前文不需要重新算。就像你写文章时，不需要每分钟重读第一段——你已经记住它了。KV Cache 就是把这些"记忆"暂存起来，避免重复计算。这就是为什么 GPU 显存如此紧张——每生成一个新 token，KV Cache 都在增长。

**RoPE（旋转位置编码）** 解决了"位置"如何告诉模型的难题。传统方法给每个位置一个编号，但 RoPE 更巧妙地用旋转来编码相对位置，让模型天然地理解"两个词之间隔了多远"。

## 3. 关键公式 (LaTeX)

### Scaled Dot-Product Attention

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

其中 $Q, K, V \in \mathbb{R}^{n \times d_k}$。

### Multi-Head Attention (MHA)

$$
\begin{aligned}
\text{MHA}(X) &= \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O \\[4pt]
\text{head}_i &= \text{Attention}(XW^Q_i, XW^K_i, XW^V_i)
\end{aligned}
$$

每个 head 有独立的投影矩阵 $W^Q_i, W^K_i, W^V_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$。

### Multi-Query Attention (MQA)

MQA 中，所有 head 共享 K 和 V 的投影矩阵，只有 Q 保持独立：

$$
\text{head}_i = \text{Attention}(XW^Q_i, XW^K, XW^V), \quad i = 1, \dots, h
$$

其中 $W^K, W^V \in \mathbb{R}^{d_{\text{model}} \times d_k}$ 被所有 head 共享。

### Grouped-Query Attention (GQA)

GQA 在 MHA 和 MQA 之间折中：将 $h$ 个 head 分成 $g$ 组，每组共享一套 K/V：

$$
\text{head}_i = \text{Attention}(XW^Q_i, XW^K_{\lfloor i/g \rfloor}, XW^V_{\lfloor i/g \rfloor})
$$

### RoPE（Rotary Position Embedding）对 Q, K 的旋转变换

$$
\tilde{q}_m = q_m \cdot e^{im\theta} = \begin{pmatrix} q_m^{(1)} \\ q_m^{(2)} \end{pmatrix} \odot \begin{pmatrix} \cos m\theta \\ \sin m\theta \end{pmatrix} + \begin{pmatrix} -q_m^{(2)} \\ q_m^{(1)} \end{pmatrix} \odot \begin{pmatrix} \sin m\theta \\ \cos m\theta \end{pmatrix}
$$

更简洁的二维旋转形式：

$$
f_{\{q,k\}}(x_m, m) = R_{\Theta, m}^d \cdot W_{\{q,k\}} \cdot x_m
$$

其中 $R_{\Theta, m}^d$ 是 block-diagonal 旋转矩阵，频率 $\theta_j = 10000^{-2j/d}$。

RoPE 满足关键性质——两个 token 之间的 attention score 仅依赖于相对位置 $m-n$：

$$
(R_{\Theta, m}^d q)^T (R_{\Theta, n}^d k) = q^T R_{\Theta, n-m}^d k
$$

### 前馈网络 (FFN / MLP)

Transformer 中每个 attention 层后的 FFN：

$$
\text{FFN}(x) = \sigma(xW_1 + b_1)W_2 + b_2
$$

GPT 中常用 GELU 激活，LLaMA 使用 SwiGLU：

$$
\text{SwiGLU}(x) = (xW_1 \odot \text{SiLU}(xW_2)) W_3
$$

### Scaling Laws (Chinchilla / Kaplan)

Kaplan et al. (2020):

$$
L(N, D) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D}
$$

Chinchilla 优化分配：**$D \approx 20 \times N$**（训练 tokens 数约为参数量 20 倍时最优）。

## 4. 公式背后的直觉

**Scaled Dot-Product Attention**：为什么除以 $\sqrt{d_k}$？当 $d_k$ 很大时，$QK^T$ 的内积值会变大，softmax 的梯度会进入饱和区（梯度消失）。除以 $\sqrt{d_k}$ 可以使方差保持为 1，让 softmax 保持在有效的梯度区间。

**MHA vs MQA vs GQA 的核心直觉**：MHA 的 K 和 V 投影矩阵数量为 $2h$，每个 head 独立存储，导致 KV Cache 总大小 $= 2h \cdot d_k \cdot L$（$L$ 为序列长度）。MQA 将所有 head 的 K/V 压缩为一套共享矩阵，KV Cache 减少到 $2d_k \cdot L$，但可能损失模型质量。GQA 是折中方案——按组共享 K/V，在质量与效率间平衡。

**RoPE 的直觉**：位置嵌入的核心需求是让模型能感知 token 之间的距离。用绝对位置编号不够好——因为"第 5 个词和第 100 个词距离 95"，纯绝对编号表示这个关系需要复杂学习。RoPE 通过旋转矩阵实现内积只依赖相对位置的性质：把 Q 和 K 分别按位置旋转，点积后位置信息自然化为相对差，不需要额外学习。

**KV Cache 增长量级**：对于 $L$ 层、$h$ 个 head、hidden size $d$、序列长度 $n$、batch size $b$ 的模型：

$$
\text{KV Cache size} = 2 \cdot b \cdot L \cdot n \cdot d \quad (\text{bytes for FP16})
$$

对于 LLaMA-70B（80 层，hidden 8192，8K context），单 batch 的 KV Cache 约 $2 \times 1 \times 80 \times 8192 \times 8192 \times 2 \text{ bytes} \approx 20 \text{ GB}$。

**为什么 LLM 如此"吃算力"**：Self-Attention 的计算量是 $O(n^2 d)$，FFN 的计算量是 $O(n d^2)$。当 $n$ 很大时，Attention 主导；当 $d$ 很大（大模型），FFN 也极其昂贵。Scaling Laws 告诉我们：更大的模型需要按 $D \approx 20N$ 配比数据，这意味着训练 GPT-4 级别模型需要数万亿 tokens——训练成本呈超线性增长。

## 5. 工业界用途

| 组件 | 工业应用 |
|------|---------|
| **Self-Attention** | 所有现代 NLP 基础；代码生成（Copilot）、机器翻译、对话系统 |
| **MQA / GQA** | Google PaLM（MQA）、LLaMA-2 70B/LLaMA-3（GQA）、服务部署时 KV Cache 节省可达 8x |
| **RoPE** | LLaMA 全系列、Qwen、Mistral、Gemma——几乎成为新模型的事实标准 |
| **KV Cache** | vLLM / TensorRT-LLM 推理框架核心优化目标 |
| **Scaling Laws** | 指导数据中心级训练预算分配——多少 GPU、多少数据、多大规模 |
| **GPT 架构** | ChatGPT、GPT-4、Copilot、各类企业级对话应用 |
| **LLaMA 架构** | 开源生态核心：LLaMA-3、Mistral、Qwen、DeepSeek 均基于此架构改进 |

## 6. PyTorch 实现思路

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """Standard MHA with optional KV cache support."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, kv_cache=None):
        B, T, D = x.shape
        q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B, h, T, d_k)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # KV Cache concatenation (inference only)
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        new_kv_cache = (k, v)

        # Scaled dot-product attention
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, h, T, T_past+T)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = attn_weights @ v  # (B, h, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out), new_kv_cache


class GroupedQueryAttention(nn.Module):
    """GQA: n_kv_heads < n_heads, heads share K/V in groups."""
    def __init__(self, d_model, n_heads, n_kv_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, self.d_k * n_kv_heads, bias=False)
        self.W_v = nn.Linear(d_model, self.d_k * n_kv_heads, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)

        # Expand KV heads to match Q heads: repeat_interleave
        k = k.repeat_interleave(self.n_groups, dim=1)  # (B, n_heads, T, d_k)
        v = v.repeat_interleave(self.n_groups, dim=1)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


class RoPE(nn.Module):
    """Rotary Position Embedding applied to Q and K."""
    def __init__(self, d_k, max_seq_len=4096, theta=10000.0):
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        # Precompute frequencies
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, freqs)  # (max_seq_len, d_k/2)
        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())

    def forward(self, x, offset=0):
        """x shape: (B, n_heads, T, d_k)"""
        T = x.shape[2]
        cos = self.cos_cached[offset:offset+T, :].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[offset:offset+T, :].unsqueeze(0).unsqueeze(0)
        x1, x2 = x.chunk(2, dim=-1)
        x_rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + x_rotated * sin


class FeedForward(nn.Module):
    """SwiGLU FFN as used in LLaMA."""
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or int(8/3 * d_model)
        self.W1 = nn.Linear(d_model, d_ff, bias=False)
        self.W2 = nn.Linear(d_model, d_ff, bias=False)  # gate
        self.W3 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.W3(F.silu(self.W1(x)) * self.W2(x))
```

**高效实现要点**：生产环境推荐使用 `torch.nn.functional.scaled_dot_product_attention`（PyTorch 2.0+），它内部调用 FlashAttention kernel。使用 `SDPBackend` 可以自动选择最优实现。

```python
# PyTorch 2.0+ 高效 attention（自动使用 FlashAttention）
out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.1, is_causal=True)
```

## 7. TinyML / Edge AI 部署意义

对于边缘设备（手机、IoT、嵌入式）：

- **MQA/GQA 直接减少 KV Cache 内存**：从 MHA 的 $2h \cdot d_k \cdot L$ 降至 GQA 的 $2g \cdot d_k \cdot L$，可在端侧内存受限时运行更长上下文。
- **RoPE 无额外参数**：与 learned position embedding 不同，RoPE 无需存储位置嵌入表，节省参数量。
- **SwiGLU 相比 GELU**：激活效率高，在 INT8 量化时有更好的数值稳定性。
- **KV Cache 量化（如 KV8/KV4）**：进一步压缩 K/V 精度，是端侧 LLM（如 llama.cpp 的 GGUF 格式）的核心技术。
- **实际部署**：即便使用 GQA + 4-bit 量化 + 4K context，7B 模型仍需约 4-6GB 内存——仍远超普通 MCU（<1MB SRAM），但对手机 NPU 可行。

## 8. 常见误区

> ❌ **误区 1："Attention 取代了 RNN，所以 Transformer 一定更快"**
> 不对。Attention 的 $O(n^2)$ 复杂度意味着长序列反而比线性复杂度的 RNN 慢得多。Transformer 的优势在于并行训练（所有位置同时计算），而非推理速度。

> ❌ **误区 2："KV Cache 越大越好——应该尽可能缓存"**
> KV Cache 是内存换时间的典型。当 context 超过 128K 时，单 batch 的 KV Cache 可达数百 GB，超过单 GPU 显存，需要 offloading 或多 GPU 拆分。

> ❌ **误区 3："RoPE 就是简单地把位置编号加到向量上"**
> RoPE 不做加法，做的是旋转。关键区别在于它保证 attention score 仅依赖相对位置，这在训练长文本和外推到训练未见长度时至关重要。

> ❌ **误区 4："GQA 比 MHA 显著差"**
> 实验表明 GQA（如 8 KV heads + 32 Q heads）几乎无损甚至持平 MHA。LLaMA-3 70B 用 8 KV heads（vs 64 Q heads）效果极好。性能退化仅在极度压缩（如 1 KV head = MQA）时才明显。

> ❌ **误区 5："Scaling Laws 告诉我们模型越大越好"**
> Chinchilla 定律恰恰说明，在固定计算预算下，盲目堆参数量而不增加数据量是次优的。GPT-3 是典型的"over-parameterized, under-trained"案例。

## 9. 面试问题

**Q1: 为什么 Self-Attention 需要除以 $\sqrt{d_k}$？**
A: 防止大维度内积导致 softmax 输入过大，梯度消失。除以 $\sqrt{d_k}$ 将方差归一化为 1。

**Q2: MQA 和 GQA 的区别及各自适用场景？**
A: MQA 所有 head 共享一套 K/V——KV Cache 最小但质量损失最大；GQA 分组共享——在 MHA 和 MQA 间折中。MQA 适合低延迟流式场景（小模型），GQA 适合大模型部署（LLaMA-2/3）。

**Q3: KV Cache 的内存增长速度？**
A: $2 \times b \times L \times n \times d \times \text{bytes\_per\_elem}$（bf16: 2 bytes）。随序列长度 $n$ 线性增长，对长上下文是最大的内存瓶颈。

**Q4: RoPE 相比其他位置编码（Sinusoidal / ALiBi / Learned）的优势？**
A: (1) 天然支持相对位置建模；(2) 可通过 NTK-aware 等方法外推训练时未见过的长度；(3) 无额外可学参数；(4) 与 Attention 计算融合，效率高。

**Q5: 一句话解释 Scaling Laws 的实践意义？**
A: 给定计算预算，用 Chinchilla 定律 $D \approx 20N$ 分配参数和数据，而非盲目增大模型。

**Q6: SwiGLU vs GELU——为什么现代 LLM 选用 SwiGLU？**
A: SwiGLU 的 gating 机制让 FFN 能够选择性通过信息，实验证明在相同计算量下质量更好。

**Q7: Transformer 推理时，Prefill 阶段和 Decode 阶段的计算特点有何不同？**
A: Prefill 阶段一次性处理整个 prompt，Attention 计算密度高（$O(n^2)$），Compute-bound；Decode 阶段每次只生成一个 token，KV Cache 已存储历史，每次只算当前 token 的 attention，Memory-bound（受限于带宽而非算力）。

## 10. 本讲总结

Transformer 和 LLM 的核心在于 **Attention 机制 + 规模化架构设计**。本讲从 Self-Attention 基础出发，深入 MHA → MQA → GQA 的演化路径——这是一个"用轻微质量换大幅效率"的递进。RoPE 作为位置编码的事实标准，因其相对位置建模和长度外推能力取代了此前各种方案。

**计算瓶颈全景**：
- Attention 复杂度 $O(n^2 d)$：长序列时由 $n^2$ 主导
- FFN 复杂度 $O(n d^2)$：大模型时 $d^2$ 主导
- KV Cache 大小随序列长度线性增长，是推理内存瓶颈

**Scaling Laws** 给出了最优资源配置的数学指导——在固定算力下，Chinchilla 定律告诉我们应该按 $D \approx 20N$ 配比训练数据，而非无限堆参数。

下一讲将进入 LLM 推理优化的实战——量化、稀疏化、PagedAttention 和 FlashAttention——这些是让大模型真正"跑起来"的关键技术。
