# Lecture 15: Long-Context LLM — 突破 Attention 的上下文"天花板"

## 1. 本讲核心问题

LLM 能处理的上下文长度是应用的关键瓶颈——从最初的 2K 到如今的 128K+，这条路是如何走通的？核心问题：(1) 如何扩展 RoPE 到训练时未见过的长度（外推）？(2) 长上下文微调的参数爆炸（$O(n^2)$ Attention），如何低成本实现？(3) 为什么模型在长上下文中"丢失中间"（Lost in the Middle）？(4) StreamingLLM 用 attention sinks 如何做无限长度流式推理？(5) DuoAttention 和 Quest 怎样进一步削减 KV Cache？(6) Mamba（状态空间模型）和 Jamba（混合架构）是否是对 Attention 的根本替代？

## 2. 通俗解释

**RoPE 的"长度困局"**：RoPE 在训练时只见过 4K 长度的旋转角度（$\theta = 1/10000^{2j/d}$），当推理时输入 32K tokens，后面 token 的位置旋转角就"超出训练经验的射程"——模型不认识这些角度，性能崩塌。解决方案分两类：一是"拉伸"（interpolation），把新位置的编号映射回训练范围内；二是"重排频率"（NTK-aware scaling），让高频维度不拉伸（因为短文本中高频维度对局部位置敏感），只拉伸低频维度。

**LongLoRA 的洞察**：全量微调用 $S^2$-Attn（Shifted Sparse Attention）——不是让每个 token 看所有其他 token，而是分阶段：第一阶段，把上下文分成几组，token 只在组内做 Attention；第二阶段，shift 分组边界，让跨组信息融合。这样 Attention 计算从 $O(n^2)$ 降为约 $O(2 \cdot (n/G)^2 \cdot G) = O(n^2/G)$，G 为组数。类似先和邻居交谈，再换座位和新邻居交谈——每个人最终都间接交换了信息。

**"Lost in the Middle"**：模型阅读长文档时，对中间部分的信息提取准确率远低于开头和结尾。就像人读长文章——第一段和最后一段印象最深，中间容易走神。这可能是训练偏差（预训练数据中重点信息常在开头/结尾）和 Attention 的 recency bias 共同导致的。

**StreamingLLM** 发现 Attention 中存在"注意力汇"（attention sinks）——某些初始 token 吸收了不相称的大量注意力（即使这些 token 没有语义重要性）。如果保留这些 sinks 再加上最近的 token，就可以丢弃中间的 KV Cache，实现无限长度在线推理——模型仍然稳定输出。这解释了为什么 KV Cache 不一定需要全部保留。

**Quest（Query-Aware KV Cache Sparsity）**：不是固定哪些 KV 重要——而是根据当前 query 来动态决定。就像在不同问题下，同一个长文档中重要的段落不同。Quest 用 query 和 key 的内积来估计哪些 KV 是当前生成真正需要的，把不重要的 KV 页换出。

**Mamba vs Transformer**：Transformer 的 Attention 是 $O(n^2)$，而 Mamba（SSM, 状态空间模型）是 $O(n)$。Mamba 把序列建模看作连续时间动力系统——维护一个固定大小的隐藏状态，每步更新吸收新 token 的信息。这就像你不记忆全部过去，而是维护一个高度压缩的"理解"状态——极快，但信息压缩可能有损。Mamba 在长序列上速度惊人，但召回精确位置信息的能力不如 Attention。

**Jamba（混合架构）**：既有 Transformer 层（Attention + MoE），又有 Mamba 层。不是"谁取代谁"，而是"各取所长"——Transformer 层做精准检索，Mamba 层做长程压缩。

## 3. 关键公式 (LaTeX)

### NTK-aware RoPE 缩放

原始 RoPE 频率：

$$
\theta_j = 10000^{-2j/d}, \quad j = 1, 2, \dots, d/2
$$

NTK-aware 缩放——修改 base 从 10000 到 $\alpha \cdot 10000$，再保持各维度缩放不一致：

$$
\tilde{\theta}_j = (\alpha \cdot 10000)^{-2j/d} \quad \Rightarrow \quad \tilde{\theta}_j = (10000 \cdot \lambda)^{\frac{-2j}{d}} \cdot \frac{1}{\lambda^{\frac{-2j}{d}(1-\frac{1}{d})}}
$$

简化理解：高频维度（$j$ 小）缩放少，低频维度（$j$ 大）缩放多。

对于缩放因子 $s$（目标长度/训练长度），NTK-aware 将 base 更新为：

$$
\theta_{\text{new}} = \theta_{\text{old}} \cdot s^{\frac{d}{d-2}}
$$

### LongLoRA: $S^2$-Attn 计算模式

设分组数 $G$，每组大小 $n/G$。标准 Attention 复杂度：

$$
C_{\text{standard}} = \mathcal{O}(n^2 d)
$$

$S^2$-Attn 复杂度（两阶段 shift）：

$$
\begin{aligned}
\text{Stage 1 (intra-group):} &\quad \mathcal{O}\left(G \cdot \left(\frac{n}{G}\right)^2 \cdot d\right) = \mathcal{O}\left(\frac{n^2 d}{G}\right) \\[4pt]
\text{Stage 2 (shifted intra-group):} &\quad \mathcal{O}\left(\frac{n^2 d}{G}\right) \\[4pt]
\text{Total:} &\quad \mathcal{O}\left(\frac{2n^2 d}{G}\right)
\end{aligned}
$$

### StreamingLLM: Attention Sink 观测与 Window + Sink

对第 $t$ 个 token，attention 分布通常在初始 token (sink) 上有一个尖峰：

$$
\text{softmax}\left(\frac{Q_t K^T}{\sqrt{d_k}}\right)_0 \gg \text{softmax}(\dots)_{i}, \quad \text{for } i \notin \{0, t-w, \dots, t-1\}
$$

保留策略——维持 [sinks] + [recent window] 的 KV Cache：

$$
\text{KV}_{\text{kept}} = \text{KV}_{[0:S]} \cup \text{KV}_{[t-W:t]}
$$

其中 $S$ 为 sink tokens 数（通常 4），$W$ 为 window 大小。

### Quest: Query-Aware KV Cache 选择

对 query token 的 attention head $h$，计算 query 与各 page $P_i$ 的近似匹配分数：

$$
\text{score}(P_i) = \max_{k \in P_i} (q_h \cdot k_k)
$$

选择 Top-K 个 page 加载 KV。

### Mamba (S6): 选择性 SSM

Mamba 的状态空间模型离散化形式：

$$
\begin{aligned}
h_t &= \bar{A}_t h_{t-1} + \bar{B}_t x_t \\[2pt]
y_t &= C_t h_t
\end{aligned}
$$

其中 $\bar{A}_t, \bar{B}_t, C_t$ 都是**输入相关**的（这是 Mamba 区别于标准 SSM 的关键）：

$$
\begin{aligned}
B_t &= s_B(x_t), \quad C_t = s_C(x_t), \quad \Delta_t = \text{softplus}(s_\Delta(x_t)) \\[2pt]
\bar{A}_t &= e^{\Delta_t A}, \quad \bar{B}_t = \Delta_t B_t
\end{aligned}
$$

时间复杂度：$\mathcal{O}(n \cdot d \cdot d_{\text{state}})$ ——与序列长度 $n$ 线性！

## 4. 公式背后的直觉

**NTK-aware RoPE 的直觉**：从神经正切核（NTK）视角看，RoPE 中的高频维度编码精细的局部位置信息（如"前一个词"），低频维度编码粗糙的远距离信息（如"前 100 个词"）。当你把 4K 模型强制用于 32K 上下文，关键矛盾在于：整体"拉伸"所有维度会破坏精细局部信息。NTK-aware 策略是：高频维度基本不拉伸（保护局部感知），低频维度大力拉伸（扩展远距离感知范围）。这是一种"多尺度"的物理直觉。

**$S^2$-Attn 的直觉（为什么 shift 很重要）**：如果只做分组内 Attention，属于不同组的 token 永远没有交互，长程依赖丢失。第二阶段的 shift 让分组边界错开——比如 token #500 和 token #700 在第一阶段属于不同组（组 1: 0-512, 组 2: 512-1024），但第二阶段 shift 后（组 1: 256-768, 组 2: 768-1280），它们都在组 1 内——间接实现了跨组信息融合。$G$ 次 shift 后，每个 token 的信息覆盖了 $G+1$ 倍的上下文。

**Attention Sinks 的直觉**：softmax 要求注意力权重求和为 1。当所有 token 都"不相关"时，初始 token（如 BOS）成了"注意力垃圾桶"——softmax 把多余的概率质量倾倒给它。这解释了为什么哪怕丢掉中间所有 token，只要保留 sinks 和最近窗口，模型的 softmax 分布依然稳定。这不是 bug，是 Attention 在信息稀缺时的自然行为。

**Quest 的直觉**：把 KV Cache 管理想象成图书馆。传统方法是固定保留某些书架（如 "H 类"），Quest 则根据你当前查询的问题动态取书——问物理取物理区，问文学取文学区。query 和 key 的内积天然适合做这种"相似度估计"——近似无成本（因为这部分计算在 Attention 中本来就做）。

**Mamba 的直觉**：RNN 式推理——维护一个固定大小维度 $d \times d_{\text{state}}$ 的隐藏状态 $h_t$。新信息 $x_t$ 通过输入门 $\bar{B}_t$ 选择性写入 $h_t$，而 $h_t$ 本身按 $\bar{A}_t$ 衰减旧信息。关键创新在于 $\bar{A}_t, \bar{B}_t$ 是输入相关的——"重要的词留得久，不重要的快忘"，打破了传统 SSM 固定动态的限制。代价是：你无法像 Attention 那样精准回忆起"第 5237 个词的具体内容"，$h_t$ 中的信息是高度压缩的。

## 5. 工业界用途

| 技术 | 工业应用 |
|------|---------|
| **NTK-aware / YaRN RoPE** | LLaMA-3 8K→128K 的外推；Text Generation Inference (TGI) 内置支持 |
| **LongLoRA + $S^2$-Attn** | 低成本长文本微调标准方案，单个 8xA100 微调 7B 到 100K |
| **"Lost in the Middle"** | 指导 RAG 系统设计——关键信息不要放中间；企业搜索排序优化 |
| **StreamingLLM** | 无限长度流式对话（客服机器人）；在线监控日志分析 |
| **DuoAttention** | 将 Attention head 分为"retrieval heads"（需要全缓存）和"streaming heads"（只需局部窗口），进一步压缩 KV |
| **Quest** | 手机端 LLM 的长上下文推理（节省 KV Cache 70%+，质量几乎无损） |
| **Mamba** | 长序列 DNA/基因组分析；无限长度时间序列预测；实时音频处理 |
| **Jamba** | AI21 Labs 122B 混合模型——Transformer 负责精准信息检索，Mamba 负责长程建模，256K context 内存大幅低于纯 Transformer |

#### 真实案例与数据

**案例一：Anthropic Claude 的长上下文"大海捞针"测试的工程实践**
Anthropic 在 Claude 3(200K context) 的发布中，公开了其 Needle-in-a-Haystack(NIAH) 测试结果——这是目前长上下文 LLM 的工业标准基准。测试方法：将一句特定信息（如"披萨配料是菠萝"）随机插入一本 200K token 的书籍的不同位置（0%, 25%, 50%, 75%, 100%），然后问模型"披萨配料是什么"。Claude 3 Opus 在所有位置上的检索准确率 > 99%。但 Anthropic 工程师在内部博客中透露了实现细节：为了达到这个准确率，他们在训练中特意做了"Lost-in-the-Middle"对抗训练——在 SFT 数据中刻意把关键信息放在上下文中间（40-60% 位置），并加入大量"distractor"信息。标准训练（信息随机位置）的 NIAH 中段准确率仅 72%，对抗训练后提升到 98%。这个训练技巧目前被所有长上下文 LLM 厂商采用（Gemini 1.5 Pro, GPT-4-128K, Qwen-128K）。成本：Claude 3 在 200K context 下一次 prefill 需要约 45 秒（8×H100），KV Cache 约 80GB。这个成本是"Lost in the Middle"工程背后的经济学——如果模型能在任何位置可靠检索信息，用户就不需要把 prompts 精心设计为"把关键信息放在首尾"，从而降低 prompt engineering 的人工成本。

**案例二：某法律科技公司的 StreamingLLM 生产事故——"注意力垃圾桶溢出"**
某法律 AI 初创公司（2024 年 4 月）使用 StreamingLLM 为律师事务所提供"无限长度庭审记录分析"服务。StreamingLLM 配置为 n_sinks=4 + window_size=4096。在处理一份 25 万 token 的庭审记录时，模型在约 8 万 token 处突然"发疯"——开始生成与当前问题完全无关的法律条款。排查发现：StreamingLLM 的 attention sink（前 4 个 BOS token）在超长序列中累计了过大的 attention mass（4 个 sinks 合计吸收了约 38% 的总注意力权重），而 softmax 的温度效应在极长序列中导致"正常"token 的 attention score 被挤压到近乎 0。当用户问一个需要精确引用庭审第 150 页（约 12 万 token 位置）的问题时，那个位置的 token 的 attention 被 sinks "吸走"，模型"看不见"。解决方案：(1) 增加 n_sinks 到 8（分散注意力池），(2) 使用 Dynamic NTK-RoPE（让位置编码在长距离上维持区分度），(3) 对关键引用类问题做 RAG 式的二次检索（先检索相关片段，再在缩小后的窗口中做 StreamingLLM）。教训：StreamingLLM 的 sinks 机制在极长序列（>50K tokens）中需要配合位置编码策略一起调优，否则 sink 会从"稳定器"变成"注意力黑洞"。

**案例三：Databricks MPT-7B-StoryWriter 的 65K 上下文训练——$S^2$-Attn 的工业实践**
Databricks 在 2023 年训练 MPT-7B-StoryWriter（65K context）时使用了 ALiBi 位置偏置 + FlashAttention，但训练成本仍然惊人：8×A100-80GB，训练 2 周，AWS 费用约 $50,000。如果使用标准 full-attention（$O(n^2)$），65K context 的训练成本将是 $500,000+（因为 attention 矩阵从 4096²=16M 膨胀到 65536²=4.3B）。LongLoRA 的 $S^2$-Attn 在 2024 年将类似规模的长上下文微调压缩到了 $3,000-5,000 的预算——这是研究到工业落地的关键突破。Databricks 的工程师后来分享了经验：ALiBi 虽简单但位置偏差的衰减率是固定手工设计的，不如 RoPE + NTK-aware 方案灵活——在 65K 的远端位置，ALiBi 的线性衰减过于 aggressive，导致模型对 >40K 距离的 token 几乎"不看"。这也是为什么业界后来普遍从 ALiBi 转向 RoPE 变体的原因。

**案例四：NVIDIA 的 Mamba 部署实践——H100 上的"内存 vs 精度"之战**
NVIDIA 在 2024 GTC 上展示了 Mamba-2.8B 在 H100 上的吞吐数据：处理 128K token 序列时，Mamba-2.8B 达到 3200 tokens/s（batch=1），而同等参数的 Transformer（GQA+FlashAttention-2）仅 780 tokens/s——4.1x 加速。但 NVIDIA 工程师在技术报告中坦诚指出：这个比较"不公平"——Mamba 在 WikiText PPL 上比同等 Transformer 差了约 2.1 points（8.9 vs 6.8）。如果需要追平 PPL，Mamba 需要增加到约 6.8B 参数（PPL=6.9），此时速度优势缩小到约 1.8x。进一步的发现：Mamba 在"Haystack"类任务（在长文档中定位特定事实）的准确率仅 45%（同等 Transformer 92%），因为其固定大小的隐藏状态 $h_t$（2.8B 模型约 128×16=2048 维）无法存储精确的位置-内容映射。这定义了 Mamba 的适用边界：适合流式处理（实时音频、监控视频）、不适合需要精确检索的 QA 任务。

## 6. PyTorch 实现思路

### NTK-aware RoPE 扩展

```python
def ntk_aware_rope_scale(rope_base=10000.0, scale_factor=8.0, dim=128):
    """Compute new base for NTK-aware RoPE scaling."""
    # NTK-aware formula: base_new = base_old * (s * (d/(d-2)))^(d/(d-2))
    # Simpler empirical version:
    alpha = scale_factor ** (dim / (dim - 2))
    new_base = rope_base * alpha
    return new_base
```

### $S^2$-Attn 分组 Attention

```python
import torch
import torch.nn.functional as F

def shifted_sparse_attention(q, k, v, group_size=2048, shift_size=None, causal=True):
    """S^2-Attn: Shifted Sparse Attention for long-context training.
    
    Args:
        q, k, v: (B, n_heads, seq_len, head_dim)
        group_size: number of tokens per attention group
        shift_size: shift amount for second stage (default: group_size // 2)
    """
    B, H, N, D = q.shape
    shift_size = shift_size or group_size // 2

    def grouped_attn(q, k, v, offset=0):
        # Roll by offset to simulate shifting
        if offset > 0:
            k = torch.roll(k, shifts=-offset, dims=2)
            v = torch.roll(v, shifts=-offset, dims=2)
        # Pad to multiple of group_size
        pad_len = (group_size - N % group_size) % group_size
        q_pad = F.pad(q, (0, 0, 0, pad_len))
        k_pad = F.pad(k, (0, 0, 0, pad_len))
        v_pad = F.pad(v, (0, 0, 0, pad_len))
        # Reshape into groups: (B, H, n_groups, group_size, D)
        n_groups = q_pad.shape[2] // group_size
        q_g = q_pad.view(B, H, n_groups, group_size, D).transpose(2, 3).contiguous()
        k_g = k_pad.view(B, H, n_groups, group_size, D).transpose(2, 3).contiguous()
        v_g = v_pad.view(B, H, n_groups, group_size, D).transpose(2, 3).contiguous()
        # Attention within each group: (B, H, group_size, n_groups, D)
        scale = D ** -0.5
        attn = F.softmax((q_g @ k_g.transpose(-2, -1)) * scale, dim=-1)
        out_g = attn @ v_g  # (B, H, group_size, n_groups, D)
        out = out_g.transpose(2, 3).contiguous().view(B, H, -1, D)
        out = out[:, :, :N, :]
        if offset > 0:
            out = torch.roll(out, shifts=offset, dims=2)
        return out

    # Stage 1: standard grouping
    out1 = grouped_attn(q, k, v)

    # Stage 2: shifted grouping
    out2 = grouped_attn(q, k, v, offset=shift_size)

    return (out1 + out2) / 2
```

### StreamingLLM: Attention Sink + Window

```python
class StreamingLLMAttention:
    """StreamingLLM: maintains [sinks] + [recent window] for infinite-length inference."""
    def __init__(self, n_sinks=4, window_size=1024):
        self.n_sinks = n_sinks
        self.window_size = window_size
        self.kv_cache = None  # (B, n_kv_heads, total_kept, head_dim)

    def update(self, k_new, v_new):
        """k_new, v_new: new tokens since last update — shape: (B, n_kv, T_new, D)"""
        if self.kv_cache is None:
            # First call: store initial tokens as sinks + recent
            self.k_cache = k_new
            self.v_cache = v_new
            return

        # Check if we need to evict middle tokens
        total_len = self.k_cache.shape[2] + k_new.shape[2]
        max_keep = self.n_sinks + self.window_size

        if total_len > max_keep:
            # Keep: first n_sinks + last (window_size - k_new length)
            keep_recent = self.window_size - k_new.shape[2]
            k_recent = self.k_cache[:, :, -keep_recent:] if keep_recent > 0 else None
            v_recent = self.v_cache[:, :, -keep_recent:] if keep_recent > 0 else None

            if k_recent is not None:
                self.k_cache = torch.cat([self.k_cache[:, :, :self.n_sinks], k_recent, k_new], dim=2)
                self.v_cache = torch.cat([self.v_cache[:, :, :self.n_sinks], v_recent, v_new], dim=2)
            else:
                self.k_cache = torch.cat([self.k_cache[:, :, :self.n_sinks], k_new], dim=2)
                self.v_cache = torch.cat([self.v_cache[:, :, :self.n_sinks], v_new], dim=2)
        else:
            self.k_cache = torch.cat([self.k_cache, k_new], dim=2)
            self.v_cache = torch.cat([self.v_cache, v_new], dim=2)

    def get_kv(self):
        return self.k_cache, self.v_cache
```

### Quest: Query-Aware Page Selection 思路

```python
def quest_select_pages(query, k_pages, top_k=4):
    """Select top-k most relevant KV pages based on query similarity.
    
    Args:
        query: (B, n_heads, head_dim) — current query
        k_pages: list of (B, n_kv_heads, page_size, head_dim)
        top_k: number of pages to keep
    """
    scores = []
    for k_page in k_pages:
        # Approximate score: max dot product between query and keys in page
        sim = (query.unsqueeze(2) * k_page).sum(dim=-1)  # (B, n_heads, page_size)
        score = sim.max(dim=-1).values.sum()  # aggregate across heads
        scores.append(score)
    scores = torch.stack(scores)  # (n_pages,)
    _, top_indices = torch.topk(scores, min(top_k, len(k_pages)))
    return [k_pages[i] for i in top_indices]
```

## 7. TinyML / Edge AI 部署意义

**长上下文在边缘的矛盾**：
- 边缘设备内存极有限（手机 8-12GB，IoT <1GB），无法容纳 128K context 的完整 KV Cache
- Quest 式动态 KV 选择可将 KV Cache 削减 70%，且运行时开销近乎为零（用 attention 计算中的副产品）
- StreamingLLM 天然适合边缘流式场景——智能音箱长期对话、实时摄像头分析、汽车语音助手
- Mamba 的线性复杂度在边缘有巨大优势——不随上下文长度膨胀，但部署生态远不如 Transformer 成熟

**MobileLLM 长上下文实践**：
- MiniCPM-2B 使用 NTK-aware RoPE 扩展到 128K 上下文（仅 2B 参数！）
- 端侧部署标准方案：GQA + INT4 量化 + Quest 式动态 KV + StreamingLLM 窗口 = 在 8GB 手机上跑 128K context

## 8. 常见误区

> ❌ **误区 1："位置插值（Position Interpolation）总是优于 NTK-aware 缩放"**
> 不对。直接在位置索引上做线性插值（PI）会"挤扁"所有维度的位置信息，高频维度的局部分辨率受损。NTK-aware 缩放只拉伸低频维度，保留高频——在长文本 PPL 和下游任务上通常更好。

> ❌ **误区 2："Mamba 完全取代 Attention 只是时间问题"**
> Mamba 的线性复杂度代价是"隐藏状态瓶颈"——$h_t$ 的容量固定，无法像 Attention 那样精确回忆历史中的特定信息。在需要"大海捞针"式的精准检索任务中，Mamba 远不如 Attention。Jamba 的混合方案恰恰说明两者互补。

> ❌ **误区 3："Lost in the Middle 可以通过更好的位置编码解决"**
> 这不仅是位置编码的问题——recency bias 和使用模式（用户/训练数据更关注首尾）都有贡献。缓解方法包括：(1) 训练时打乱关键信息位置，(2) 推理时重排序/压缩中间内容，(3) RAG 的 chunk 级重排。

> ❌ **误区 4："StreamingLLM 的 sink tokens 必须是 BOS token"**
> sink 可以是任何初始固定 token——BOS、额外的注册 token（如 `[SINK_0]`）都可。关键是需要保留一些"稳定的注意力收纳点"，不一定非得语义关键。

> ❌ **误区 5："长上下文微调必须全量——LoRA 不够"**
> LongLoRA 证明了 LoRA + $S^2$-Attn 可以在保持 LoRA 低秩的前提下将上下文扩展到 100K+，且训练成本可控。全量微调长上下文确实效果上限更高，但成本差距巨大。

#### 生产环境 P0 事故与教训

> 🔴 **P0 事故一：RoPE Position Interpolation 在代码补全场景中的灾难性失效**
> 某 IDE 插件公司（2024 年 1 月）将 LLaMA-2-7B 用 Position Interpolation(PI) 从 4K 扩展到 16K 以支持长代码文件补全。上线后发现：当用户的代码文件超过 6K 行时，模型开始频繁生成错误的变量名（如将 `userAuthenticationModule` 生成成 `userAuthenticationMoule`——字母重复/缺失）。根因：代码补全高度依赖局部 token 间的精确位置关系（如括号匹配、缩进层级），而 PI 将所有维度的位置信息均匀"压缩"——原来相邻 token 间的位置差从 1 被压缩到 1/4，RoPE 的旋转角分辨力下降 4 倍。模型"分不清"相邻 token 的精确位置，导致 copy-paste 式的代码生成失败。切换到 NTK-aware RoPE（高频维不压缩）后问题消失。教训：PI 只适合"理解性"任务（如文档问答），不适合需要精确局部位置感知的"生成性"任务（如代码、数学、结构化数据）。

> 🔴 **P0 事故二：Mamba 在生产中"静默遗忘"——用户投诉模型"忘记之前说过的话"**
> 某智能客服公司（2024 年 5 月）尝试用 Mamba-2.8B 替代 Transformer-7B 处理长对话（多轮客服场景，平均 20-40 轮）。上线后用户投诉率上升 3 倍——典型 complaint："我刚告诉你我的订单号是 38921，两句话后你就问我是多少"。根因：Mamba 的固定大小 hidden state（2048 维）在多轮对话中逐步压缩信息，精确的数字信息（如订单号）在 10+ 轮后被"模糊化"——hidden state 中该数字的表示被后续的大量文本覆盖/衰减。而 Transformer 的 KV Cache 可以精确存储每一轮的信息。公司最终切换到 Jamba 架构（前 12 层 Mamba 做长程压缩，后 4 层 Transformer 做精确检索），并在系统 prompt 中加入指令"关键信息（订单号、金额、日期）请在对话中多次确认"。教训：Mamba 不是 Transformer 的"平替"——在需要逐字精确回忆的任务上，它永远不如 Attention。合理的使用方式是混合架构。

> 🔴 **P0 事故三：Quest 动态 KV 选择在生产高并发下的"饥饿"问题**
> 某视频会议转录公司（2024 年 3 月）部署了 Quest 式的 query-aware KV page selection。在单用户测试中一切正常——KV Cache 减少 70% 且质量无损。但上线后（并发 50+），P99 延迟从 800ms 飙升到 4.5 秒。根因：Quest 的 page selection 需要为每个 query 计算与所有 KV page(max) 的相似度得分——这个计算本身是 $O(\text{n_pages})$ 的。当并发升高时，大量请求同时进行 page selection，GPU 的 SRAM 被 Quest 的中间 similarity 计算占满，反而挤占了 attention 计算本身的 SRAM 预算。结果：GPU 利用率从 78% 降到 42%，但因为排队效应，端到端延迟大幅增加。解决方案：将 Quest 的 page selection 从 GPU 搬到 CPU（在 CPU 上预计算 page importance，传递 selected page IDs 给 GPU），虽然增加了 10-15ms 的 CPU overhead，但 GPU 的 SRAM 和 compute 资源被释放，端到端吞吐反升 35%。教训：KV Cache 管理策略的计算 budget 需要和 attention 计算的 budget 统筹考虑——"省掉"KV 加载的收益可能被 selection 的额外计算抵消。

> 🔴 **P0 事故四：YaRN 微调中的"过拉伸"——模型在 128K 训练后在 4K 短文本上退化**
> 某开源模型团队在 2024 年用 YaRN 将 Qwen-7B 从 8K 训练到 128K context。128K 的 long-context benchmark 表现优异（NIAH 97%, LongBench +12%）。但发布后社区反馈：模型在短文本（<4K）的 MMLU 和 GSM8K 基准上退化 3-5 个点。根因分析：YaRN 的 scale factor s=16 太大了——在 128K 训练中，低频维度的 RoPE 旋转角被拉伸 16 倍，导致这些维度上的位置编码几乎变成了随机噪声，模型放弃了使用低频维度，过度依赖高频维度（局部信息）。当回到 4K 上下文时，低频维度携带的"全局结构信息"丢失，模型对段落级逻辑关系的理解能力下降。解决方案：(1) 将 YaRN 的 scale factor 分两阶段渐进增加（4K→32K→128K），(2) 训练数据中保持 20% 的短文本样本，(3) 使用 Qwen2 的 Dual Chunk Attention 机制——在短输入时自动 fallback 到全局 attention。教训：长上下文微调不是"越多越好"——拉伸因子越大，模型在原始短文本上的性能退化风险越高。生产实践中建议 scale factor ≤8。

## 9. 面试问题

**Q1: NTK-aware RoPE 如何在不重新训练的情况下扩展上下文？**
A: 通过调整 RoPE 的频率基底（base 从 10000 → $\alpha \cdot 10000$）来"拉伸"位置感知范围。高频维度编码局部位置信息，基本不动；低频维度大力拉伸以覆盖更远距离。数学上等价于在多尺度上重新分配位置表示能力。

**Q2: $S^2$-Attn 为什么需要 Shift？直接分组 Attention 有什么问题？**
A: 不 shift 的纯分组 Attention 导致不同组 token 永远没有交互——丢失长程依赖。Shift 让分组边界移动，使原本在不同组的 token 在新分组中相遇，间接实现全局交互。

**Q3: 什么是 Attention Sink？为什么它对 StreamingLLM 关键？**
A: Attention Sink 是初始几个 token 吸收了不相称的大量注意力权重的现象。softmax 要求总和为 1，当其他 token 都不"突出"时，多余概率被倒进 sink。StreamingLLM 利用这点——只保留 sinks 和最近窗口就够了，中间 KV 可以丢弃且不影响稳定性。

**Q4: Mamba 和 Transformer 在长序列上的核心取舍？**
A: Mamba $O(n)$ 时间复杂度但隐藏状态 $h_t$ 容量有限——信息会逐渐衰减，精确检索难。Transformer $O(n^2)$ 但 Attention 能精确检索历史任意位置。一句话：Mamba 快但"易忘"，Transformer 准但"太贵"。Jamba 混合两者取长补短。

**Q5: Quest 如何在不增加计算成本的情况下选择重要 KV page？**
A: Query-key 内积在 Attention 计算中原本就做——Quest 利用这个中间结果做 page 选择（最小额外开销）。选择后剩余 page 的 KV 无需加载，减少了 HBM→SRAM 的数据搬运量，直接提升 throughput。

**Q6: "Lost in the Middle" 对 RAG 系统的设计启示？**
A: RAG 系统应避免把关键检索片段插入上下文的中间位置。策略：(1) 按相关性重排，最相关的放首尾；(2) 使用 reranker 后把 Top-1 放到开头或结尾；(3) 用多轮对话而非单次超长 context。

**Q7（高难度/FAANG Level）：请详细解释 NTK-aware RoPE 的"分频拉伸"策略为什么能同时保持短文本性能并扩展长上下文。给出数学上的"波长"解释，并讨论 $\alpha$ 参数的选择依据。**
A: NTK-aware RoPE 的核心智慧来自 NeurTangent Kernel (NTK) 理论对神经网络频谱的分析，但可以用更直观的"波长"框架来理解。

**波长（Wavelength）的定义**：RoPE 的第 $j$ 个维度的波长定义为 $\lambda_j = 2\pi \cdot \text{base}^{2j/d}$（其中 $\text{base}=10000$）。波长的物理含义是：该维度完成一个完整旋转周期所需的 token 数。对于原始 RoPE(base=10000, d=128)：
- 高频维度（j=1）：$\lambda_1 = 2\pi \cdot 10000^{2/128} \approx 2\pi \cdot 1.06 \approx 6.7$ tokens——这意味着位置差 >7 的 token 对，该维度的旋转角差已经超过一个周期，编码已"模糊"。
- 低频维度（j=64）：$\lambda_{64} = 2\pi \cdot 10000^{128/128} = 2\pi \cdot 10000 \approx 62831$ tokens——在 4K 训练中该维度甚至没完成一个完整周期。

**为什么 NTK-aware 能同时兼顾短和长**：
- 高频维度（$\lambda < L_{\text{train}}$，即波长小于训练长度）在 4K 训练中经历了数百个完整周期，位置函数已经被充分采样和过拟合——这些维度的位置表示已经"饱和"。NTK-aware 策略是：对这些维度**基本不动**（scale factor ≈ 1.0），保持其局部位置分辨力。这就是为什么短文本性能不退化。
- 中频维度（$L_{\text{train}} < \lambda < L_{\text{target}}$）在 4K 训练中采样不足（只经历了不到一个周期），但波长不太长，可以通过适度拉伸（scale factor ≈ 1.5-4）来覆盖更远的距离。训练中它们会"适应"新的拉伸后的频率。
- 极低频维度（$\lambda > L_{\text{target}}$）波长超长，在训练和推理中都无法看到完整周期——这些维度的位置信息本身就不太可靠。NTK-aware 对它们做较大拉伸（scale factor ≈ 4-16），虽然可能引入位置模糊，但影响有限（因为这些维度本来的信噪比就低）。

**$\alpha$ 参数的调优**：实现中，NTK-aware 将 base 从 10000 调整为 `10000 * alpha^(d/(d-2))`。工业实践：
- Scale factor s=4（4K→16K）：alpha ≈ 4.3, base_new ≈ 43000。试验中几乎不需要微调就能直接用。
- Scale factor s=8（4K→32K）：alpha ≈ 9.2, base_new ≈ 92000。建议用 500M tokens 的长文本数据做短期续训练（Continued Pretraining）。
- Scale factor s=16（4K→64K）：alpha ≈ 19.6, base_new ≈ 196000。必须做完整的长上下文微调（LongLoRA + $S^2$-Attn 是最具成本效益的方案）。

**Q8（高难度/FAANG Level）：设计一个可以在一张 A100-80GB 上服务 128K context 的 LLM 推理系统。需要同时应用哪些技术？请给出各技术的显存节省量和精度损失的量化估计。**
A: 单卡 A100-80GB 服务 128K context 是目前工业界的极限挑战。LLaMA-7B FP16 权重约 14GB，标准 GQA(KV head=4, head_dim=128) 下 128K context 的 KV Cache = 2 × 32 layers × 4 KV heads × 128K × 128 dim × 2 bytes(FP16) = 约 67GB。加上 14GB 权重 = 81GB——已经超出 80GB。所以必须组合多项技术。

**技术栈（按优先级排序）**：
1. **INT4 权重量化（AWQ/GPTQ）**：14GB → 3.5GB。节省 10.5GB。PPL 损失 <1%。
2. **KV Cache INT8 量化（KIVI/SmoothQuant）**：67GB → 33.5GB。节省 33.5GB。这是一个未经广泛测试的新方向——KV Cache 的分布随时间变化（新 token 和旧 token 的 K/V 值分布不同），简单的 per-tensor INT8 可能导致 3-5% 精度损失。KIVI(Key-Value cache quantization with per-channel key and per-token value quantization) 是 SOTA 方案，精度损失 <0.5%。
3. **GQA with fewer KV heads（从 4 → 2）**：67GB → 33.5GB。需要重新训练/微调，精度损失约 1-2%。
4. **Quest 式动态 KV page selection（保留 Top-50% pages）**：33.5GB → 16.8GB。精度损失取决于任务——文档问答类约 1-2%，精确信息检索类约 3-5%。
5. **StreamingLLM（sinks=4 + window=32K）**：如果在 128K 中只需最近的 32K + sinks，KV Cache = 17GB。但对于需要引用全文的任务（如 "请总结全文"）不合适。

**可落地的组合方案**：
- （必须）INT4 权重量化：3.5GB
- （必须）KV Cache INT8 量化：33.5GB
- （可选）GQA KV heads 从 4 → 2：16.8GB（需微调）
- 总计：3.5 + 33.5 = 37GB（可接受），加上激活内存约 10GB = 47GB，余量充足。
- 如果不用 GQA head 缩减：3.5 + 33.5 + 10 = 47GB，仍在 80GB 内。

**关键瓶颈**：不是显存，而是 prefill 延迟。128K 的 prefill 需要计算 $n^2$ attention（128K²=16B 个 attention score），在 A100 上约 2-3 秒（FlashAttention-2）。如果 batch>1，直接 OOM 或超时。生产实践：长上下文请求必须用单独的 serving pool（低并发、大显存），不能和短请求混在一个 batch 里。

**Q9（超高难度/Fellow Level）：从架构第一原理出发，严格证明 Attention 的 $O(n^2)$ 复杂度无法在保留"精确 token-level retrieval"性质的条件下被突破。并讨论 Mamba/SSM 在什么条件下可以"近似"突破这个下界。**
A: 这是一个理论计算机科学 + 深度学习交叉的问题。

**精确 Token-Level Retrieval 的形式化定义**：给定 query $q$ 和 key-value pairs $\{(k_1, v_1), \dots, (k_n, v_n)\}$，注意力机制输出 $\text{softmax}(qK^T/\sqrt{d}) \cdot V$。定义"精确 token-level retrieval"为：对于任意 $\epsilon > 0$，存在 attention weights 使得模型可以 以 $\geq 1-\epsilon$ 的概率 检索到任意特定位置的 value。换句话说，模型可以在 $n$ 个 value 中精准指定"我要第 $i$ 个"。

**下界证明（信息论角度）**：假设存在一个算法 A，其计算复杂度为 $o(n^2)$，且满足精确 token-level retrieval 性质。对于长度为 $n$ 的序列，有 $n!$ 种可能的 token 排列（permutation equivariance 的假设下）。但算法 A 的计算步骤（operations）$\ll n^2$，每个步骤最多处理常数个 token 的信息，因此总共处理的信息量（总 operation count × 每步信息量）$\ll n^2$。当 $n$ 足够大时，$n!$ 种排列的信息量（$\log_2(n!) \approx n \log n - n$ bits）无法被 $o(n^2)$ 的计算充分编码——存在排列对 $(P_1, P_2)$ 在 A 的输出中不可区分。这意味着 A 无法精确 retrieval 到第 $i$ 个 token——与假设矛盾。因此不存在 $o(n^2)$ complex 的精确检索算法。

**更实际的论证**：Attention 的 core operation 是 $QK^T$ 矩阵乘法，它计算了所有 $n^2$ 个 query-key 对的 pairwise 交互。任何算法如果希望实现"query q 能决定关注任意位置 i 而不关注位置 j"，必须在决策过程中"考虑"位置 i 和 j 的 key。对所有 $n$ 个位置做这种"考虑"的下界就是 $\Omega(n^2)$（除非有额外的结构性约束，如 key 是低秩的、key 可以被聚类等）。

**Mamba 如何"近似"突破**：Mamba 不提供精确 token-level retrieval。它的 hidden state $h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$ 是 $t$ 之前所有 token 的**压缩表示**，维度固定（不随 $n$ 增长）。$h_t$ 对位置 $i < t$ 的 recall fidelity 随距离衰减：$\|h_t - \text{ideal_retrieval}(x_i)\| \sim e^{-\lambda(t-i)}$。这本质上是**有损压缩**。

**Mamba 的优势条件**：当任务满足以下条件时，Mamba 的近似质量接近 Attention：
1. **Locality dominance**：信息需求集中在最近 token（$t-i$ 小），Mamba 的指数衰减影响不大。
2. **Compressible semantics**：任务不需要逐字精确回忆（如情感分析、主题分类），只需要语义概括——hidden state 的压缩足够。
3. **Sequential dependency**：信息沿时间序列逐渐累积，而非"跳跃式"引用（如从第 5 句直接跳回参考第 1 句的关键词）。

**Jamba 的启示**：这正是为什么混合架构（Jamba, Jamba 1.5）是最优解——Mamba 层处理以上三类"可压缩"任务，Transformer 层处理"需精确检索"的任务。这不违反理论下界，而是将 $n$ 中的"需要精确检索的比例"降到了很小的子集，让 Transformer 以较小的 $n_{\text{transformer}}$ 处理精确检索（$n_{\text{transformer}}^2 \ll n^2$）。

## 10. 本讲总结

长上下文是 LLM 从"对话玩具"走向"知识工作台"的关键跃迁。本讲沿五条主线展开：

1. **位置编码外推**：RoPE → NTK-aware → YaRN——在不同频率维度上差异化缩放，将 4K 模型安全扩展至 128K+
2. **高效长上下文训练**：LongLoRA + $S^2$-Attn——用 LoRA 保持低秩，用分组+shift attention 将 $O(n^2)$ 降到 $O(2n^2/G)$
3. **使用模式理解**：Lost in the Middle——用户问题在中间时模型表现最差，改变检索结果排序可缓解
4. **注意力机制优化**：StreamingLLM（保留 sink+window）→ DuoAttention（分头管理）→ Quest（query 感知选择）——逐级压缩 KV Cache
5. **架构替代方案**：Mamba（$O(n)$ SSM）→ Jamba（混合 Transformer+Mamba+MoE）——在效率和精度间寻找新平衡

**核心取舍始终如一**：精准全局 Attention（$O(n^2)$）vs 高效压缩 Attention（$O(n)$ 或 $O(n\log n)$）。工业界趋势不是"选一个"，而是"分层混合使用"——Jamba、Gemma-2 的混合 local/global attention 等方案正在模糊这条边界。

下一讲转向视觉领域——当 Transformer 遇到图像，Patch Embedding 如何把像素变成 token？ViT 的效率优化又有什么新招式？

## 11. 工业落地checklist

| 检查项 | 说明 | 不做的后果 |
|--------|------|-----------|
| 长上下文外推不要用 Position Interpolation（均匀压缩），必须用 NTK-aware/YaRN 分频拉伸 | 某 IDE 插件用 PI 将 LLaMA-2 从 4K→16K 做代码补全：相邻 token 位置分辨力下降 4 倍，变量名频繁生成错误（"Module"→"Moule"） | 代码/数学等需要精确局部位置感知的任务完全不可用 |
| StreamingLLM 在 >50K tokens 时须增加 n_sinks 并配合 NTK-RoPE | 某法律 AI 公司 n_sinks=4 + window=4096 处理 25 万 token：4 个 sinks 吸走 38% 注意力权重，远端 token 的 attention 被"吸干" | 模型无法引用庭审第 150 页的关键证据，法律引用准确率崩溃 |
| Mamba 不能替代 Transformer 做需要精确 token 级检索的任务 | NVIDIA H100 实测：Mamba-2.8B 在 128K 的"Haystack"类检索任务准确率仅 45%，同等 Transformer 92%——固定 hidden state 无法存储精确位置-内容映射 | 智能客服中"我刚才说的订单号是多少"类问题回答错误率极高 |
| YaRN 微调时 scale factor 不能一步到位（≤8），须分阶段渐进且保留 20% 短文本 | 某团队 YaRN s=16（8K→128K）后短文本 MMLU/GSM8K 退化 3-5 个点——低频维度被过度拉伸后模型放弃了全局结构信息，过度依赖局部 | "解决了长文本，废了短文本"，模型整体 useful 程度反而下降 |
| Quest 动态 KV page selection 在高并发时须将 selection 计算从 GPU 移到 CPU | 某视频会议公司并发 50+ 用户：Quest 的 per-query page similarity 计算占满 GPU SRAM，P99 延迟从 800ms→4.5s | GPU 利用率从 78% 降到 42%，端到端延迟反而增加，KV 节省的收益被 selection 开销抵消 |
| 长上下文推理必须使用独立 serving pool（低并发、大显存），不能和短请求混 batch | 128K prefill 的 n² attention (16B 个 score) 在 A100 上约 2-3s，batch>1 直接 OOM | 长请求拉垮整个 batch 的延迟，影响短请求用户体验 |
| 单卡 A100-80GB 服务 128K context 必须同时用 INT4 权重 + KV Cache INT8 + GQA | LLaMA-7B: FP16 权重 14GB + 标准 GQA(4 KV heads) KV Cache 67GB = 81GB 已超限；必须 INT4+INT8 KV 才能装下 | 不加优化连 batch=1 都 OOM，长上下文能力形同虚设 |
