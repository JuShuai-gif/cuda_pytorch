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

## 10. 本讲总结

长上下文是 LLM 从"对话玩具"走向"知识工作台"的关键跃迁。本讲沿五条主线展开：

1. **位置编码外推**：RoPE → NTK-aware → YaRN——在不同频率维度上差异化缩放，将 4K 模型安全扩展至 128K+
2. **高效长上下文训练**：LongLoRA + $S^2$-Attn——用 LoRA 保持低秩，用分组+shift attention 将 $O(n^2)$ 降到 $O(2n^2/G)$
3. **使用模式理解**：Lost in the Middle——用户问题在中间时模型表现最差，改变检索结果排序可缓解
4. **注意力机制优化**：StreamingLLM（保留 sink+window）→ DuoAttention（分头管理）→ Quest（query 感知选择）——逐级压缩 KV Cache
5. **架构替代方案**：Mamba（$O(n)$ SSM）→ Jamba（混合 Transformer+Mamba+MoE）——在效率和精度间寻找新平衡

**核心取舍始终如一**：精准全局 Attention（$O(n^2)$）vs 高效压缩 Attention（$O(n)$ 或 $O(n\log n)$）。工业界趋势不是"选一个"，而是"分层混合使用"——Jamba、Gemma-2 的混合 local/global attention 等方案正在模糊这条边界。

下一讲转向视觉领域——当 Transformer 遇到图像，Patch Embedding 如何把像素变成 token？ViT 的效率优化又有什么新招式？
