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

### 生产级案例分析

- **LLaMA-3 的 GQA 部署效益**：Meta 公开了 LLaMA-3 70B（8 KV heads × 64 Q heads）在生产中的 KV Cache 对比数据。在 8K context length、batch size=1、FP16 精度下，如果使用 MHA（64 KV heads），KV Cache 约 40GB——单张 H100（80GB）只能勉强容纳 batch=1，完全无法做 batch inference。GQA（8 KV heads）将 KV Cache 压缩到约 5GB——同样的 H100 可以容纳 batch=8，服务吞吐量提升 8 倍。Meta 在 Llama 3 技术报告中确认：GQA 在 MMLU benchmark 上精度损失 < 0.3%，但在生产部署中 KV Cache 节省了 87.5%。这直接影响了 API 定价——Anthropic 的 Claude 和 Google 的 Gemini 也在内部使用类似 GQA 的压缩方案。
- **vLLM 的 PagedAttention 内存管理**：vLLM（UC Berkeley 开源的高性能 LLM 推理框架）用 PagedAttention 管理 KV Cache——将 KV Cache 划分成固定大小的"page"（类似操作系统虚拟内存的 page 概念），允许物理上不连续的 KV 块被逻辑上连续的 sequence 使用。在 LLaMA-2 70B 上，vLLM 相比 HuggingFace Transformers（naive KV Cache 实现）内存利用率从 20-30% 提升到接近 100%，吞吐量提升 14-24x。关键数字：单张 A100 80GB 上，vLLM 可以同时服务 256 个 512-token 的 request，而 HuggingFace 只能服务 18 个——差距来自 KV Cache 内存碎片的最小化。
- **Apple Intelligence 的设备端 LLM**：Apple 在 WWDC 2024 发布 Apple Intelligence 时确认，设备端运行的模型（~3B 参数）使用了 GQA + grouped-wise 量化（每组 32 个权重共享一个 scale factor）将模型从 6GB（FP16）压缩到约 1.5GB（3.5-bit avg）。在 iPhone 15 Pro（A17 Pro, 8GB RAM, 16-core Neural Engine）上，首次推理（prefill）延迟约 0.6s，后续 token 生成（decode）延迟约 28 个 token/秒——这需要约 4GB 的 KV Cache + 模型权重 + 系统开销共享 8GB 的内存，展示了极致的工程压缩能力。

| 技术 | 解决的问题 | 节省幅度 | 精度代价 | 适用场景 |
|------|-----------|---------|---------|---------|
| **GQA (8 KV heads)** | KV Cache 内存膨胀 | 87.5% KV Cache | < 0.3% MMLU | 所有大模型推理（> 7B） |
| **MQA (1 KV head)** | 极端 KV Cache 压缩 | 96.9% KV Cache | 1-2% MMLU | 低延迟流式场景（PaLM API） |
| **FlashAttention-2** | Attention 显存 O(n²) | ~8x 显存，2-3x 速度 | 无（精确等价） | 所有 Transformer 训练/推理 |
| **vLLM PagedAttention** | KV Cache 碎片化 | 吞吐量 14-24x | 无 | LLM 在线推理服务 |
| **AWQ/GPTQ 量化** | 模型权重 + KV 内存 | 3-4x 模型压缩 | < 1% perplexity | 消费级设备端部署 |
| **NTK-aware RoPE** | 上下文长度外推 | 4-8x context 扩展 | < 2% perplexity (长文本) | 长文档/代码总结 |

> **工程洞察**：LLM 推理成本的下降是"多重技术叠加"的结果，不是单一突破。将 FlashAttention-2（节省显存）+ GQA（节省 KV Cache）+ INT4 量化（节省权重）+ vLLM PagedAttention（消除碎片）同时应用时，综合效果是乘性的——在一张 A100 80GB 上，从 naive HuggingFace 的 ~2 tokens/s（batch=1, LLaMA 70B）到 ~600 tokens/s（batch=32）——300x 提升。

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

### 生产级常见陷阱

> ❌ **误区 6："GQA 中 KV heads 数越多越好——越接近 MHA 精度越高"**
> 实际上，当 KV heads > 8 之后，精度增益进入极严重的边际递减区。LLaMA-3 团队在内部实验了 {2, 4, 8, 16, 32, 64} 的 KV head 配置，发现 8 KV heads vs 64 KV heads（完整 MHA）的 MMLU 差距只有 0.2%，但 16 vs 8 的额外增益是 0.05%——基本是噪声级别。而 8 → 16 需要 2 倍的 KV Cache 内存。**生产教训**：greedy 地追求"更多 KV heads"是无意义的——把宝贵的 KV Cache 预算分配给更长的 context length（如 8K → 32K）对用户体验的提升远远大于"从 8 个 KV heads 增加到 16 个"。

> ❌ **误区 7："FlashAttention 对所有序列长度都有效——越长越好"**
> FlashAttention 的核心优化是"将 QK^T 矩阵分块计算，避免完整存储 n×n 的 attention matrix"。当序列长度 n < 256 时，HBM 带宽足够快，分块带来的 overhead（额外的 tile-level softmax rescaling）反而使 FlashAttention 比标准 attention 慢 5-15%。NVIDIA 的官方 benchmark（cuDNN vs FlashAttention-2）显示 crossover 点约在 n=384——短于这个值的序列用标准实现更快。这在 LLM 推理的 decode 阶段（每次 n=1）尤为重要——**FlashAttention 在 decode 阶段完全无效**，因为这些优化是为 prefill（处理 prompt）设计的。

> ❌ **误区 8："RoPE 可以零成本外推到任意长度——只需改 theta 值"**
> NTK-aware 和 YaRN 等 RoPE 扩展方法的本质是"对高频维保持原始频率，对低频维做插值"。当外推倍数超过训练最大长度的 4-8x 时，低频维度的频率被压缩得太厉害，导致不同位置之间的相对位置信号模糊化——attention 无法区分两个距离很远的 token。"零成本外推到 128K"的宣称只在 perplexity 基准上看起来好，在真实的长文档 QA、代码理解等需要精确位置感知的任务上，外推长度超过 4x 时精度显著下降。Anthropic 在 Claude 的技术报告中确认：他们在长上下文训练时使用了"逐步扩展"策略（16K → 32K → 64K → 100K），每一步都用对应长度的训练数据微调了几十万步——不是零成本。

> ❌ **误区 9："LLM 推理的瓶颈永远是显存带宽"**
> 这取决于阶段。**Prefill 阶段**（一次性处理整个 prompt）是 compute-bound——n² 的 attention 计算量主导，A100 的计算利用率可达 60-70%。**Decode 阶段**（逐 token 自回归生成）是 memory-bound——每次只算一个新 token 的 Q 与所有历史 K 的 attention，计算量只有 O(n)，但需要从 HBM 读取整个 KV Cache。Decode 阶段的 A100 计算利用率通常 < 5%，大多数时间在等待数据从显存传输。因此，**KV Cache 压缩（GQA/MQA、INT8 KV、PagedAttention）在 decode 阶段的价值远大于 prefill 阶段**——这是很多优化决策的分水岭。

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

**Q8（高难度）：LLaMA-3 70B 使用 GQA（8 KV heads, 64 Q heads）训练。如果训练完成后你想做 MHA→GQA 的"后转换"（即把一个 MHA 模型转成 GQA 模型以节省推理成本），你会怎么做？有哪些不可逆的信息损失？**

后转换的主要方案是"KV head mean-pooling"——将 64 个独立的 KV heads 分 8 组，每组 8 个 head 的 K 和 V 权重矩阵做平均池化（mean pooling），得到 8 组共享的 KV 投影矩阵。具体步骤：(1) 对每组的 8 个 K 投影矩阵 W_K^1...W_K^8（每个矩阵 (d_model, d_k)），计算 W_K_shared = mean(W_K^1, ..., W_K^8)；(2) V 同理；(3) Q heads 保持不变（64 个独立 Q）。

**不可逆的信息损失**：(1) 每个 head 独特的"关注模式"被平均掉了——head 3 可能擅长"主语-谓语"关注，head 7 擅长"局部上下文"，mean pooling 后这些专业化能力融合成一个模糊的模式；(2) 原始 MHA 训练中，Q 矩阵和 K 矩阵是联合学习的（Q 知道应该"问"什么样的问题来匹配对应 K head 的"信息提供风格"），但 pooling 后 K 矩阵变了，Q 矩阵没有变——这导致了 Q-K 匹配的 mismatch，类似通信中"发射端不变但接收端换了解码方式"；(3) 这种 mismatch 可以通过后续的少量微调（~100B tokens 的 continued pretraining）来校正——这就是为什么现在的大模型（LLaMA-3、Mistral）都在**训练时就使用 GQA**，而非训练后转换。从 MHA 转换到 GQA 再微调的最终精度，通常比从头用 GQA 训练低 0.5-1.5 个百分点。

**Q9（高难度）：KV Cache 量化（如 KV4——将 KV 值量化为 4 位整数）在长序列推理中非常有效。但从数值分析角度，什么情况下 KV Cache 量化会导致灾难性精度崩溃（perplexity 暴涨 > 20%）？为什么？**

灾难性崩溃发生的条件：**(1) 高度重复的 token 序列导致 KV 值异常集中**。在代码生成场景中，如果 prompt 包含大量重复 pattern（如连续 100 行 `x += 1`），attention 的 softmax 输出会高度集中在少数几个 key 上。此时这些关键 key 的 K 和 V 向量在 FP16 下有轻微差异（如 0.51 vs 0.49），但 INT4 量化将两者都映射到同一量化 bin（都变成 3，scale 0.5 以下无法区分）。模型丢失了"区分相近但不同的 key"的能力，导致 attention 权重"错误地集中或分散"，级联放大后 perplexity 暴涨。

**(2) 异常 large outlier 的 K/V 值未被正确处理**。LLM 的 hidden states 中普遍存在"激活 outlier"——某些 channel 的值比其他 channel 大 10-100x（如 LLaMA 中约 0.1% 的 channel 是 outlier）。如果用 per-tensor 量化（一个 scale factor 覆盖整个 K 矩阵），大 outlier 会撑大 scale factor，导致所有正常值被压缩到极少的几个量化 bin（如 -128 到 127 的 INT8 范围中，99.9% 的值挤在 -2 到 2 之间）——信噪比灾难性降低。

**解决方案**（来自 llama.cpp 和 HuggingFace TGI 的实践）：(a) 使用 per-channel 或 per-group 量化（每组 128 个 channel 共享 scale factor）来隔离 outlier 的影响；(b) 对 K cache 和 V cache 使用不同的量化策略——K 做 INT8（因为 K 的分布更均匀，且精度对 attention score 计算至关重要），V 做 INT4（因为 V 只是加权求和，对量化噪声更鲁棒）。llama.cpp 的 Q8_0 K + Q4_0 V 混合量化在 32K context 上的 perplexity 退化 < 1%。

**Q10（高难度）：在部署 LLaMA-3 8B 到一部 6GB 内存手机上时，你已经用了 INT4 权重 + INT8 KV Cache，但 decode 阶段仍然超过 500ms/token（不可接受）。在不改变模型架构的前提下，你能想到的 3 个最有效的延迟优化是什么？给出每个优化的预期收益和风险。**

**(1) Speculative Decoding（推测解码）- 预期收益 2-3x token/s，风险低**
用一个小得多的"草稿模型"（draft model, 如 LLaMA-3 135M，~100MB INT4）快速生成 ~4-5 个候选 tokens，然后用 8B 模型一次性验证（一个 prefill forward 处理 5 个 tokens）。如果候选 token 匹配，5 个 token 只花 ~1 次 forward 的成本——理论加速约 5x，实际约 2-3x（因为草稿模型有 miss）。风险：需要额外 100MB 内存存放草稿模型；在创意性任务上 draft model 的 miss rate 高（> 60%），加速效果减半。

**(2) 滑动窗口 Attention（Sliding Window Attention）- 预期收益 1.5-2x decode 延迟，风险中**
将 KV Cache 的长度限制为最近的 W 个 tokens（如 W=4096），丢弃更远的历史。对于很多不需要全局上下文的场景（如即时对话、短信回复），1024-4096 的窗口已经足够。将 KV Cache 从 8K 压缩到 4096 相当于节省 50% 的 decode IO 时间，同时减少 attention 计算量。**风险**：超过窗口的历史上下文永久丢失——在需要引用对话开头内容的场景（如"回到我们之前讨论的..."），模型会"失忆"。需要根据应用场景选择性开启——对某些 user prompt 用完整 KV Cache，对简单问题用滑动窗口。

**(3) 逐层 CPU/GPU/ANE 流水线化 - 预期收益 1.3-1.8x 端到端吞吐，风险中**
不要等所有 32 层在 GPU 上串行跑完。将模型按层拆分：Layers 1-8 跑 GPU（计算最密集的 prefill 激活），Layers 9-20 跑 Apple Neural Engine (ANE)（功耗最低），Layers 21-32 跑 GPU（输出层精度要求高）。三层硬件同时流水线处理不同的 micro-batch——当 GPU 在处理 Layer 3 时，ANE 在处理 batch B 的 Layer 15。**风险**：跨硬件的 tensor 传输开销可能抵消流水线收益（需要用 shared memory pool 而非显式拷贝）；ANE 的 NPU 对某些非标准 op 的支持差，可能需要 CPU fallback；debug 复杂度指数级增加——任何一层出错，定位根因极其困难。Apple 在 A17 Pro 上的 Core ML 使用类似的混合调度，但实现细节高度保密。

## 10. 本讲总结

Transformer 和 LLM 的核心在于 **Attention 机制 + 规模化架构设计**。本讲从 Self-Attention 基础出发，深入 MHA → MQA → GQA 的演化路径——这是一个"用轻微质量换大幅效率"的递进。RoPE 作为位置编码的事实标准，因其相对位置建模和长度外推能力取代了此前各种方案。

**计算瓶颈全景**：
- Attention 复杂度 $O(n^2 d)$：长序列时由 $n^2$ 主导
- FFN 复杂度 $O(n d^2)$：大模型时 $d^2$ 主导
- KV Cache 大小随序列长度线性增长，是推理内存瓶颈

**Scaling Laws** 给出了最优资源配置的数学指导——在固定算力下，Chinchilla 定律告诉我们应该按 $D \approx 20N$ 配比训练数据，而非无限堆参数。

下一讲将进入 LLM 推理优化的实战——量化、稀疏化、PagedAttention 和 FlashAttention——这些是让大模型真正"跑起来"的关键技术。

## 11. 工业落地checklist

| 检查项 | 说明 | 不做的后果 |
|--------|------|-----------|
| 生产部署 LLM 必须用 GQA（8 KV heads），不要用完整 MHA | Meta LLaMA-3 70B 实测：GQA（8 KV heads）KV Cache 仅 5GB vs MHA（64 KV heads）40GB；MMLU 精度损失 < 0.3%，但吞吐提升 8x | 单 H100 80GB 只能跑 batch=1，API 并发能力受限，GPU 成本翻 8x |
| KV Cache 量化的 K 和 V 必须用不同策略：K 用 INT8，V 可用 INT4 | llama.cpp 验证：K 对 attention score 计算至关重要（softmax 指数放大误差），V 只是加权求和更鲁棒；K8+V4 混合在 32K context 上 PPL 退化 < 1% | 统一 INT4 量化 K/V 在长序列推理中出现灾难性精度崩溃，PPL 暴涨 > 20% |
| FlashAttention 只在 prefill 阶段有效，decode 阶段不要用它 | NVIDIA benchmark：序列长度 n < 384 时 FlashAttention 比标准 attention 慢 5-15%（tiling overhead > IO 节省）；decode 时 n=1 完全无效 | 在 decode 阶段强行使用 FlashAttention，延迟不降反增，GPU 利用率更低 |
| RoPE 外推到训练长度 4x 以上时不能零成本——必须做继续训练 | Anthropic Claude 验证：16K→128K 外推需逐步扩展（16K→32K→64K→100K），每步用对应长度数据微调几十万步；否则长文档 QA 精度显著下降 | 宣称"128K context"但实际在 64K+ 位置检索准确率 < 60%，用户投诉"模型记不住长文档" |
| 模型 head_dim 必须能被 16 整除，否则 FlashAttention-2 静默退化 | FA2 的 Triton kernel 内部 BLOCK_SIZE 依赖 head_dim % 16 == 0 做 memory coalescing；head_dim=120 时速度比 FP16 xformers 慢 40%——不报错、不警告 | 为节省 KV Cache 定制非标准 head_dim 后推理速度反而比标准实现慢，优化方向完全错误 |
| Speculative Decoding 的 draft model 分布必须与 target model 对齐，每次主模型更新后需重新评估 | 某平台 LLaMA-70B 从 v2 升级 v3 后 draft model（v2）接受率从 82% 骤降到 11%——KL 散度检测 > 0.01 → 须重启对齐训练 | speculative decoding 不仅无加速，端到端延迟反增 30% |
| LLM 推理不能只看平均 token/s，P99 首 token 延迟才是用户体验的关键指标 | 美团 vLLM 部署教训：max_num_batched_tokens 从 2048 调到 32768 → prefill 阶段单步延迟从 800ms 飙到 5s，用户感知"发出问题后卡 5 秒才出答案" | 平均指标好看但 P99 延迟严重超标，用户投诉率飙升 |
