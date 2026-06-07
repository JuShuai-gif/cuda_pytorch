# Lecture 13: LLM Deployment — 推理优化使大模型"飞"起来

## 1. 本讲核心问题

LLM 训练完成后，如何让它真正"跑起来"？本讲聚焦 LLM 推理部署的"三座大山"：**显存墙（memory wall）、吞吐量瓶颈、延迟要求**。具体回答：为什么要量化？Weight-Activation 量化和 Weight-Only 量化有何不同？SmoothQuant 如何解决激活值异常值（outlier）问题？AWQ 和 GPTQ 分别如何实现高效的仅权重量化？稀疏化（Wanda, DejaVu）如何进一步压缩模型？vLLM 的 PagedAttention 如何将 KV Cache 利用率从 ~30% 提升到接近 100%？FlashAttention 到底"Flash"在哪里？Speculative Decoding 为何能让生成速度翻倍？

## 2. 通俗解释

**Memory Wall（显存墙）问题**：想象一个图书馆，你要基于 5000 本书写一篇论文。每次写新段落，你都得翻阅之前的笔记。这些笔记就是 KV Cache。问题是——你的书桌（GPU HBM）只有 80GB 空间，而笔记+书本（模型权重）已经占了 75GB，剩下的空间只够放几张草稿纸。这就是 LLM 推理的现状：你没被算力卡住，而是被"放不下"卡住了。

**量化**就是把 16 位精度的数据压缩到 8 位甚至 4 位。就像一本精装书（FP16）变成袖珍版（INT4）——内容基本保留，但占用空间减少 4 倍。代价是偶尔出现"印刷模糊"（精度损失），需要聪明的方法来弥补。

**SmoothQuant** 解决一个棘手问题：LLM 中某些激活值特别巨大（outliers，可达正常值的 100 倍）。直接量化这些 outliers 会"炸掉"整个精度。SmoothQuant 的想法绝妙——通过数学变换把"锅"从激活推到权重的另一边，两边各承担一部分，都不至于崩。

**AWQ** 的洞察更巧妙：不是所有权重同等重要。某些权重通道对结果影响巨大（约 1% 的通道贡献了 90% 的重要性）。AWQ 只保护这些"精英通道"不量化，其余大胆压缩。

**PagedAttention** 中的"Page"借用了操作系统的虚拟内存（paging）概念。传统 KV Cache 像一整块连续内存——为最大长度预分配，浪费严重。PagedAttention 把 KV Cache 切成小块（block），按需分配和回收，就像 OS 管理内存页。这直接解决了 KV Cache 碎片化和利用率低的问题。

**FlashAttention** 的关键创新是避免把整个 Attention 矩阵写到 HBM（显存）再读回来。传统 Attention 分三步：`S = QK^T`（写入 HBM）→ `P = softmax(S)`（写入 HBM）→ `O = PV`（写入 HBM）。FlashAttention 把它们融合（fuse）在一起，只用 SRAM/register 做中间计算。IO 减少使得速度提升 2-4 倍。

**Speculative Decoding** 用一个"草稿模型"（draft model，小且快）快速生成候选 token，再由大模型（target model）一次性验证。这利用了 LLM 推理的一个特点：大模型一次验证多个 token 的成本和验证一个 token 差不多（因为计算量大头在加载权重而非实际计算），所以如果能"猜中"多个，就省了很多步。

## 3. 关键公式 (LaTeX)

### 均匀量化（Uniform Quantization）

将 FP 值 $r$ 映射为 $b$-bit 整数 $q$：

$$
r = S \cdot (q - Z), \quad S = \frac{r_{\max} - r_{\min}}{2^b - 1}
$$

其中 $S$ 为 scale（步长），$Z$ 为 zero-point（零点）。

### SmoothQuant: 平滑激活异常值

核心变换——将激活的量化难度"转移"给权重：

$$
Y = X \cdot W = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W) = \hat{X} \cdot \hat{W}
$$

平滑因子：

$$
s_j = \max(|X_j|)^{\alpha} / \max(|W_j|)^{1-\alpha}
$$

其中 $\alpha \in [0, 1]$ 控制转移程度。$\alpha=0.5$（对半转移）通常是好的默认值。

### AWQ: Activation-Aware Weight Quantization

基于激活幅值搜索最优 per-channel scale：

$$
s^* = \arg\min_s \mathcal{L}(s), \quad \mathcal{L}(s) = \|\text{quant}(W \cdot \text{diag}(s)) \cdot \text{diag}(s)^{-1} \cdot X - WX\|^2
$$

其中 quant 为量化函数，优化目标是使量化后的输出尽可能接近原始 FP 输出。

### GPTQ: 基于 OBS (Optimal Brain Surgeon) 的逐列量化

核心——每量化一列权重后，通过 Hessian 信息补偿剩余列：

$$
\delta_F = -\frac{w_q - \text{quant}(w_q)}{[H^{-1}]_{qq}} \cdot H^{-1}_{:, q}
$$

其中 $H = 2XX^T$ 为 Hessian（损失函数二阶信息），$w_q$ 是第 $q$ 列的权重值。

### FlashAttention: Tiling + Recomputation

不计算完整 $S = QK^T$——分块（tiling）计算并在线更新 softmax：

对于块 $i$，维护 running max $m_i$ 和 running sum $\ell_i$：

$$
\begin{aligned}
m_i^{\text{new}} &= \max(m_i^{\text{old}}, \tilde{m}_i) \\
\ell_i^{\text{new}} &= e^{m_i^{\text{old}} - m_i^{\text{new}}} \ell_i^{\text{old}} + e^{\tilde{m}_i - m_i^{\text{new}}} \tilde{\ell}_i \\
O_i^{\text{new}} &= \text{diag}(e^{m_i^{\text{old}} - m_i^{\text{new}}}) O_i^{\text{old}} + e^{\tilde{m}_i - m_i^{\text{new}}} \tilde{P}V
\end{aligned}
$$

### Wanda: 权重+激活联合判定的剪枝准则

对权重 $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$，重要性分数：

$$
S_{ij} = |W_{ij}| \cdot \|X_j\|_2
$$

即权重绝对值乘以对应输入通道的激活范数。

### Speculative Decoding 加速比

设草稿模型每次生成 $\gamma$ 个 token，接受率 $\alpha$：

$$
\text{Speedup} = \frac{1 - \alpha^{\gamma+1}}{(1 - \alpha)(1 + \gamma \cdot c_d / c_t)}
$$

其中 $c_d, c_t$ 为草稿/目标模型单 token 推理成本。当 $\alpha \to 1$ 且 $c_d \ll c_t$ 时，speedup $\to \gamma+1$。

## 4. 公式背后的直觉

**SmoothQuant 的直觉**：发现 LLM 激活中的 outliers 高度集中在特定的几个 channel 维度（这些 channel 始终产出巨大值）。如果在 Activation 侧量化（将 scale 除以极大值），outlier channel 被压碎；如果在 Weight 侧把所有责任推给权重，权重精度被连累。SmoothQuant 做的是"两边摊"——通过数学恒等变换（乘一个对角阵再除回来），把 outliers 的部分量级平滑迁移到权重，双方都不极端。

**AWQ 的直觉**：不是所有通道等价。约 1% 的显著通道承载了大部分信息。与其对所有通道统一量化后 finetune，不如识别出这些关键通道并保护它们（保持高精度或给更大 scale）。类比：压缩照片时，人脸区域需要高分辨率，背景可以模糊。

**GPTQ 的直觉**：贪心逐列量化时会损失信息，但"后量化"的列可以借此前的误差信息做补偿——这就是 Optimal Brain Surgeon（OBS）思想。量化 $w_q$ 后对所有未量化权重做一步补偿更新，使整体输出误差最小。

**FlashAttention 的直觉**：GPU 的 HBM（80GB, ~2TB/s）和 SRAM（~20MB, ~20TB/s）之间带宽差 10x。传统做法把 $n \times n$ 的注意力矩阵（$O(n^2)$）写到 HBM 再读回——当 $n=32K$ 时矩阵本身就有 1GB（FP32）。FlashAttention 的核心创新是把计算切成 block，在 SRAM 内完成——每个 block 只从 HBM 读一次输入、写一次输出，中间全在高速 SRAM 进行。

**Speculative Decoding 的直觉**：LLM 推理的 decode 阶段是 memory-bound——GPU 大部分时间在等数据从 HBM 搬运到 SM，计算单元大量空闲。让一个大模型同时验证 5 个 token（计算量约等于验证 1 个），如果能接受其中 4 个（80% 接受率），相当于用 1 次大模型推理完成了 5 次 decode 的工作。

## 5. 工业界用途

| 技术 | 工业应用 |
|------|---------|
| **SmoothQuant** | NVIDIA TensorRT-LLM 内置支持；8-bit 推理接近 FP16 质量 |
| **AWQ** | vLLM 原生支持；HuggingFace TGI 集成；@4-bit 下精度几乎无损 |
| **GPTQ** | 静态 INT4 量化标准之一；AutoGPTQ 库；llama.cpp GGUF 格式 |
| **TinyChat** | 端侧 INT4 LLM 高效推理引擎（AWQ 团队开发） |
| **Wanda / SparseGPT** | 训练后一次性剪枝 50% 权重不掉点；配合稀疏硬件加速 |
| **DejaVu** | Contextual sparsity——推理时根据输入动态选择激活的神经元 |
| **vLLM PagedAttention** | 开源社区最广泛使用的 LLM 推理框架，吞吐量提升 14-24x |
| **FlashAttention** | PyTorch 2.0+ `scaled_dot_product_attention` 默认后端；已成为训练/推理标配 |
| **Speculative Decoding** | 生产部署（Together AI, Groq, Anyscale）的延迟优化利器 |

#### 真实案例与数据

**案例一：美团 vLLM PagedAttention 部署 70B 模型**
美团外卖智能客服在 2024 年将 LLaMA-2-70B 推理服务从 FasterTransformer 迁移到 vLLM。核心收益来自 PagedAttention 的 KV Cache 管理：传统方案为每个请求预分配 4096 token 长度的 KV Cache（70B 模型 8K context 下单请求约 4.5GB KV Cache），但实际用户对话平均仅 600 tokens——利用率不到 15%。切到 vLLM 后，block size=16 的 PagedAttention 实现了动态分配，KV Cache 利用率从 ~30% 跃升到 ~95%。在 4×A100-80GB 的节点上，并发吞吐从 **5 req/s → 18 req/s**，P99 首 token 延迟从 3.2s 降到 1.1s。更关键的是，PagedAttention 支持同一 session 内多轮对话共享 prefix KV——预填一次 system prompt 的 KV Cache，后续所有轮次复用，prefill 成本降低 70%+。

**案例二：vLLM Prefix Caching 在生产中的坑**
某电商平台用 vLLM 部署客服系统时开启了 `enable_prefix_caching` 功能。上线第三天出现 P0 事故——大量请求超时。根因是：prefix cache 哈希表的哈希碰撞（default hash 基于 block tokens 的前 16 bytes）在相似的 system prompt 之间频繁误判 cache hit，导致错误的 KV block 被复用，产生乱码输出。vLLM 社区后来加入了更严格的 hash 校验（full block hash on GPU）。教训："prefix caching 在生产中不是免费午餐——需要监控 cache hit rate、hash collision rate，并对监控到的 collision 做 fallback re-computation。"

**案例三：字节跳动 QLoRA 微调 LLaMA-7B**
字节 AI Lab 在 2024 年初公开了其基于 QLoRA 的 LLaMA-7B 微调方案。传统全量微调 LLaMA-7B 需要 8×A100-80GB，总成本约 $2000（按 10 小时计）。QLoRA(NF4, r=64, alpha=16) 将需求降到 **单张 A100-40GB（约 $200）**，且训练时间仅增加 20%（因为计算时需要反量化到 BF16 做前向/反向）。关键经验：(1) NF4 的 double quantization 将量化元数据从 0.5GB 压缩到 0.13GB；(2) LoRA 的 target_modules 从仅 Q/V 扩展到所有 attention + FFN 的 linear 层（gate_proj, up_proj, down_proj），在数学推理任务上提升了 3.2 个点；(3) 使用 paged AdamW 8-bit optimizer 进一步节省 ~40% 优化器内存。最终方案实现了 65B 模型在单张 RTX 3090 24GB 上的微调——此前这个配置只能推理，不能训练。

**案例四：FlashAttention 在 NVIDIA TensorRT-LLM 中的实际加速**
NVIDIA 在 TensorRT-LLM(0.8.0) 中集成 FlashAttention-2 后，官方公布：在 H100 上运行 LLaMA-70B decode（batch=64，seq_len=2048），端到端延迟从 38ms/token 降到 22ms/token（1.7x）。但这个数据有陷阱——实际生产环境中，随着 batch size 增大（>128）和序列变长（>8K），FlashAttention 的优势缩小到约 1.2-1.3x，因为此时瓶颈从 attention 计算转移到 MLP 和 all-reduce（TP 通信）。在批量小、序列短（<2K）的交互式场景中，FlashAttention 的加速比最大——因为此时 attention 是 compute-bound 而非 memory-bound。团队内部的经验法则是：`batch × seq_len > 64K` 时，瓶颈开始从 attention 转移，需要同时优化 MLP 的 kernel fusion。

## 6. PyTorch 实现思路

### SmoothQuant 核心实现

```python
import torch

def smooth_quant_linear(linear, x, alpha=0.5):
    """Apply SmoothQuant to a Linear layer.
    Transforms: Y = XW -> Y = (X diag(s)^-1) (diag(s) W)
    """
    W = linear.weight  # (out_feat, in_feat)
    x_max = x.abs().max(dim=0).values  # per-channel max activation
    w_max = W.abs().max(dim=0).values  # per-channel max weight

    # Smoothing factor: split the quantization difficulty
    s = (x_max.float() ** alpha) / (w_max.float() ** (1 - alpha) + 1e-8)

    # Apply smoothing
    linear.weight.data = W * s.unsqueeze(0)  # scale up weights
    return x / s.unsqueeze(0)  # scale down activations (caller handles this)


def quantize_weight_rtn(w, bits=4):
    """Round-to-Nearest weight quantization."""
    w_flat = w.float()
    w_max = w_flat.abs().max()
    w_min = -w_max  # symmetric quantization

    scale = (w_max - w_min) / (2**bits - 1)
    # Scale should be per-channel in practice
    w_q = torch.clamp(torch.round(w_flat / scale), -(2**(bits-1)), 2**(bits-1) - 1)
    w_q_int = w_q.to(torch.int8)
    return w_q_int, scale
```

### AWQ 核心实现思路

```python
def awq_search_scale(weight, x_calib, n_grid=20, alpha=0.5):
    """Search per-channel optimal scales for AWQ.
    Protect salient channels by adjusting their quantization scale.
    """
    out_feat, in_feat = weight.shape
    x_max = x_calib.abs().max(dim=0).values  # (in_feat,)

    best_scales = torch.ones(in_feat)
    best_loss = float('inf')

    for ratio in torch.linspace(0, 1, n_grid):
        # Scale up salient channels
        scales = x_max.pow(ratio).clamp(min=1e-4)  # (in_feat,)
        w_scaled = weight * scales.unsqueeze(0)  # (out_feat, in_feat)
        w_q = quantize_weight_rtn(w_scaled, bits=4)[0].float()
        w_deq = w_q / scales.unsqueeze(0)  # dequantize back

        # Measure output error on calibration data
        out_orig = x_calib @ weight.t()
        out_quant = x_calib @ w_deq.t()
        loss = ((out_orig - out_quant) ** 2).mean()

        if loss < best_loss:
            best_loss = loss
            best_scales = scales

    return best_scales
```

### PagedAttention 核心数据结构

```python
from dataclasses import dataclass
from typing import List, Optional
import torch

@dataclass
class KVCacheBlock:
    """A single block (page) of KV cache."""
    block_id: int
    k: torch.Tensor   # (n_kv_heads, block_size, head_dim)
    v: torch.Tensor   # (n_kv_heads, block_size, head_dim)
    ref_count: int = 0  # reference counting for sharing


class PagedKVCache:
    """Paged KV Cache manager — inspired by vLLM."""
    def __init__(self, n_blocks, block_size, n_kv_heads, head_dim, device='cuda'):
        self.block_size = block_size
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        # Pre-allocate all blocks as a pool
        self.blocks: List[KVCacheBlock] = [
            KVCacheBlock(
                block_id=i,
                k=torch.zeros(n_kv_heads, block_size, head_dim, device=device),
                v=torch.zeros(n_kv_heads, block_size, head_dim, device=device),
            )
            for i in range(n_blocks)
        ]
        self.free_blocks = list(range(n_blocks))

    def allocate(self) -> Optional[int]:
        """Allocate a free block. Returns block_id or None."""
        if not self.free_blocks:
            return None
        bid = self.free_blocks.pop(0)
        self.blocks[bid].ref_count += 1
        return bid

    def free(self, block_id: int):
        """Release a block back to the free pool."""
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks.append(block_id)
```

### Speculative Decoding 框架

```python
def speculative_decode(target_model, draft_model, prefix, gamma=5, max_new=100):
    """Speculative decoding with a draft model."""
    generated = list(prefix)

    # Step 1: Run draft model to propose gamma tokens
    with torch.no_grad():
        draft_tokens = draft_model.generate(generated, max_new=gamma)
        candidates = draft_tokens[-gamma:]

    # Step 2: Target model verifies all candidates in parallel
    with torch.no_grad():
        logits = target_model(generated + candidates)
        target_probs = torch.softmax(logits[-gamma-1:-1], dim=-1)

    # Step 3: Accept/reject with rejection sampling
    accepted = []
    draft_probs = draft_model.get_probs(generated + candidates)[-gamma-1:-1]

    for i in range(gamma):
        p_target = target_probs[i, candidates[i]]
        p_draft = draft_probs[i, candidates[i]]
        if torch.rand(1) < (p_target / p_draft).clamp(max=1.0):
            accepted.append(candidates[i])
        else:
            # Reject: sample from residual distribution
            residual = (target_probs[i] - draft_probs[i]).clamp(min=0)
            residual = residual / residual.sum()
            accepted.append(torch.multinomial(residual, 1).item())
            break

    return accepted
```

## 7. TinyML / Edge AI 部署意义

**量化对边缘部署的直接影响**：
- 7B 模型 FP16：~14GB → INT4：~4GB → 可部署到旗舰手机（8-12GB RAM）
- TinyChat 在 NVIDIA Jetson Orin 上能以 INT4 精度实时运行 LLaMA-7B
- 端侧场景量化质量极为敏感——因无法回退到"云"做纠错，AWQ 的保护关键通道策略在低比特时尤其重要

**稀疏化的现实**：
- 虽然 NVIDIA Ampere+ 支持 2:4 structured sparsity，但不规则稀疏（Wanda）的硬件加速仍在发展中
- 边缘设备（如 Qualcomm Hexagon, Apple ANE）的稀疏支持更有限——结构化稀疏比非结构化更有实际价值

**Speculative Decoding 在边缘**：边缘设备用极小的草稿模型（如 50M 参数 LSTM）配合 2-3B 目标模型，可在手机端实现 2x 生成加速——这对实时语音助手等场景意义重大。

## 8. 常见误区

> ❌ **误区 1："4-bit 量化必然导致严重精度损失"**
> 不对。AWQ 在 LLaMA-7B 上 INT4 仅损失 <0.5% perplexity；GPTQ 在精心校准下也接近无损。关键在于量化方法（per-channel vs per-tensor）和是否保护 salient channels。

> ❌ **误区 2："FlashAttention 是 Attention 算法的改进"**
> FlashAttention 不改变 Attention 的数学公式，它改变的是计算顺序和 IO 模式（tiling + recomputation）。结果是**数学等价**的，但速度更快、内存更少（$O(n)$ 而非 $O(n^2)$ 的内存）。

> ❌ **误区 3："量化和稀疏化可以无限制叠加"**
> 叠加量化+稀疏化后精度退化是非线性的。4-bit + 50% 稀疏通常不等效于 8x 压缩的量化效果，而可能比单个技术差得多。

> ❌ **误区 4："PagedAttention 就是为了省内存"**
> PagedAttention 的优势不仅在于省内存——它还能实现跨请求的 KV Cache 共享（如 beam search 中共享 prefix），以及减少碎片化带来的系统性吞吐下降。

> ❌ **误区 5："Speculative Decoding 需要专门训练"**
> 虽然可以对草稿模型做针对性训练（如 Medusa heads），基本的 speculative decoding 可以直接使用同系列的较小模型（如 LLaMA-68M 作为 LLaMA-7B 的 draft）。

#### 生产环境 P0 事故与教训

> 🔴 **P0 事故一：vLLM prefill batch size 配置过大导致首 token 延迟 5 秒+**
> 某出行平台（2024 年 6 月）升级 vLLM 版本后，将 `max_num_batched_tokens` 从默认的 2048 调到 32768（意图提高吞吐）。结果预填（prefill）阶段 batch size 暴涨，单次 prefill 的 attention 矩阵从 2048²=4M 膨胀到 32768²≈1B 个元素——单个 prefill step 的延迟飙到 5-8 秒。用户看到的是"发出问题后页面卡住 5 秒，然后瞬间出答案"。根因：vLLM 的 continuous batching 在 prefill 阶段会尽量填满 `max_num_batched_tokens`，而你设置 32768 意味着它会把积压请求的 prefill 全摞在一起处理直到达到上限。教训：prefill 和 decode 对 batch size 的敏感度完全不同——prefill 是 compute-bound，过大 batch 直接 OOM 或超时；decode 是 memory-bound，batch 影响小。生产环境中 `max_num_batched_tokens` 需要配合 `max_num_seqs` 和 GPU 型号（A100 40GB 推荐 ≤8192，H100 80GB 推荐 ≤16384）联合调优，而不是盲目加大。

> 🔴 **P0 事故二：FlashAttention-2 在非 16 整除 head_dim 上静默退化**
> 某 LLM 推理框架团队在 2024 年为节省 KV Cache 将 head_dim 从 128 改为 120（非标准值），FlashAttention-2 的 Triton kernel 内部 `BLOCK_SIZE` 依赖 head_dim 能被 16 整除来做 memory coalescing。head_dim=120 时，FA2 的 backward kernel 退化为标准 non-fused attention——不报错、不警告，但速度比 FP16 xformers 还慢 40%。团队花了 5 天定位到这个"静默性能 bug"。教训：FA2 对 head_dim 的约束不是"crash or work"，而是"work correctly but slowly"——需要显式检查 `head_dim % 16 == 0` 并在不满足时 fallback 到 xformers 或 FA1。NVIDIA 在 FA2 2.5.0+ 中加入了此检查，但老版本没有。

> 🔴 **P0 事故三：AWQ 量化在 MoE 模型上的精度崩塌**
> 某金融公司尝试将 Mixtral 8×7B 用 AWQ INT4 量化后部署。量化完成后，base model 的 perplexity 仅从 6.2 → 6.4（看似正常），但在金融合规场景（精确的数字提取和法规引用）中，准确率从 91% 暴跌到 64%。根因：Mixtral 的 router 网络（gate network）在 AWQ 的 per-channel scale 搜索中被错误地分配了过小的 scale——因为 calibration 数据中不同 expert 被激活的频率差异极大，导致 gate 的激活分布严重偏移。AWQ 默认的 128 条 calibration 样本不足以覆盖所有 8 个 expert 的激活模式。教训：MoE 模型的量化需要为 gate module 单独设计校准策略——要么单独跑 500+ calibration samples 覆盖所有 expert，要么对 gate 保持 FP16（仅 1.3M 参数，开销可忽略）。

> 🔴 **P0 事故四：Speculative Decoding 的 draft model 离线更新未同步**
> 某对话平台使用 LLaMA-68M 作为 LLaMA-70B 的 draft model 做 speculative decoding。模型团队对 LLaMA-70B 主模型做了 SFT+DPO 更新（v2→v3），但 draft model 保持 v2 版本。结果上线后接受率从 82% 骤降到 11%——draft 猜测的内容几乎全被拒绝，speculative decoding 不仅没有加速，反而因为 draft model 的额外推理+全拒绝后的重采样，端到端延迟反增 30%。教训：草稿模型的分布必须和目标模型保持对齐——每个主模型版本升级都需要重新评估 draft model 的接受率，或至少用 KL 散度检测分布偏移是否在可接受范围内（经验阈值 KL<0.01）。

## 9. 面试问题

**Q1: 解释 SmoothQuant 如何解决激活值 outliers 的量化问题？**
A: 通过数学恒等变换将激活中的大值部分地"转移"给权重。具体是计算 per-channel 平滑因子 $s_j = \max(|X_j|)^\alpha / \max(|W_j|)^{1-\alpha}$，在 activation 和 weight 之间重新分配量化误差。

**Q2: AWQ 和 GPTQ 的核心区别？**
A: (1) **方法**：AWQ 通过搜索最优 per-channel scale 保护 salient channels；GPTQ 基于 OBS 按列贪心量化并做误差补偿。(2) **依赖**：AWQ 只需少量校准数据搜索 scale；GPTQ 需要 Hessian 矩阵信息。(3) **速度**：AWQ 极快（几分钟）；GPTQ 较慢但理论上更精确。(4) **适用**：AWQ 对激活敏感，理论上更鲁棒；GPTQ 静态量化，部署时不需要激活信息。

**Q3: PagedAttention 如何解决传统 KV Cache 管理的问题？**
A: 传统 KV Cache 为 batch 内每个序列预分配最大长度的连续内存——造成 (1) 内存碎片化，(2) 大量预分配浪费（大部分序列未达最大长度），(3) 无法共享。PagedAttention 把 KV Cache 拆成固定大小的 block，按需动态分配，利用率从 ~30% 提升到 ~96%，且支持 prefix 共享。

**Q4: FlashAttention 为什么不在数学上改变 Attention，却更快？**
A: 通过 tiling 和 online softmax 技巧将计算全部保留在高速 SRAM 中，避免 $O(n^2)$ 的中间矩阵写入/读取 HBM。IO 复杂度从 $O(n^2)$ 降至 $O(n^2 d^2 / M)$（$M$ 为 SRAM 大小），实际加速 2-4x。

**Q5: Speculative Decoding 何时可能比标准 decoding 更慢？**
A: 当接受率 $\alpha$ 很低时（draft 和目标模型分布差异大），draft 模型的计算成为纯 overhead。极端情况下（$\alpha \to 0$），每次只接受 0-1 个 token 却多跑了 draft 模型。

**Q6: 什么是 DejaVu 的 contextual sparsity？**
A: 不同于静态稀疏（固定剪掉相同神经元），DejaVu 在推理时根据当前输入动态预测哪些注意力头和 FFN 神经元是激活的——大约 80% 可被稀疏化，且吞吐量提升显著。

**Q7（高难度/FAANG Level）：PagedAttention 的 block size 如何影响 memory fragmentation 和 scheduling overhead？请从 vLLM 源码层面分析 trade-off。**
A: Block size 是 PagedAttention 最关键的调优参数之一（vLLM 默认 16，SGLang 默认 256），其 trade-off 需要从三个层面理解：

**(1) 内部碎片（Internal Fragmentation）**：最后一个 block 通常装不满。平均浪费 = (block_size - 1) / 2 个 token 空间。Block size=16 时平均浪费 7.5 tokens（约占 16 的 47%），block size=256 时平均浪费 127.5 tokens（约占 256 的 50%）。对平均长度 200 tokens 的用户请求，用 block_size=256 意味着首 block 只装 200 tokens（浪费 56 个位置）——浪费率高达 28%。这直接转化为 GPU 显存浪费。Block size 越小，内部碎片越少。

**(2) 外部碎片与调度开销**：Block 越小，总 block 数越多（4096 context / 16 = 256 blocks per request）。vLLM 的 scheduler 需要为每个 request 维护 block table（类似 OS 的页表），block 数量直接决定了：(a) kernel launch 时传给 GPU 的 block_table 参数大小（kv_cache 的 `reshape_and_cache` 内核需要遍历所有 block），(b) GPU 上 attention kernel 内的 block 查找开销——需要将 block_id 转换为物理 KV Cache 地址。当 block_size=8 时，batch=128 的请求总共可能有 128×512=65536 个 blocks，block table 本身的 HBM→SRAM 加载就成为了新瓶颈。SGLang 的 radix attention 通过树结构管理 block 引用解决了一部分问题。

**(3) Prefix Cache 共享效率**：vLLM 的 Automatic Prefix Caching（APC）以 block 为单位做缓存。Block size 越小，共享粒度越细，cache hit 机会越多——但 hash 计算和比较的开销也越大。比如 system prompt "你是一个有帮助的AI助手"（约 12 tokens），用 block_size=16 时整个 prompt 在一个 block 内，一个请求用完即可被下一个请求完全复用。用 block_size=128 时，一个 block 可能混入了不同请求的独有 token（如 "你是一个有帮助的AI助手，请帮我写一封邮件给张三..."——"张三" 在同一个 block 内），导致 block hash 变化、cache miss。

**生产环境经验法则**：对于交互式对话场景（平均 prompt ~500 tokens），block_size=16 是最佳平衡（vLLM 默认值）；对于文档处理/批量推理场景（prompt >4K tokens），block_size=64-128 可降低 block table 开销。NVIDIA 的 TensorRT-LLM 默认 block_size=64。注意：修改 block_size 需要重新编译 vLLM 的 CUDA kernel——这不是运行时可调的。

**Q8（高难度/FAANG Level）：一个训练好的 LLaMA-13B，如何在不重新训练的情况下将其 MAX context length 从 4K 扩展到 32K？请比较 Position Interpolation、NTK-aware Scaling 和 YaRN，说明各自的数学原理和实际效果差异。**

A: 这是一个经典的"外推困境"——模型在训练时只见过 [0, 4095] 的位置编码，推理时遇到位置 4096-32767 的 token 时，Rotary Position Embedding (RoPE) 的旋转角度超出了训练分布，导致 attention score 计算混乱。

**(1) Position Interpolation (PI)**：直接将新位置线性映射回训练范围：`pos_new = pos_original × (L_train / L_target)`。例如位置 32000 → 32000 × 4096/32768 = 4000（缩回训练范围内）。对所有维度一视同仁地"挤扁"。问题：高频维度（负责局部位置关系，如相邻词）也被压缩——原本位置差 1 的两个 token（如位置 100 和 101），在 PI 后位置差变成 0.125（100×0.125 和 101×0.125），旋转角度差缩小到原来的 1/8。模型对相邻词的位置区分能力退化，在需要精确局部理解的任务（如代码补全、数学推理）上表现崩塌。实验数据：LLaMA-7B PI 扩展到 32K 后，passkey retrieval 的准确率从 4K 的 100% 降到 32K 的 68%——远端的"针"几乎被完全忽略。

**(2) NTK-aware Scaling (Neural Tangent Kernel)**：灵感来自 NTK 理论——神经网络的不同"频率"维度对应不同的学习动态。RoPE 的维度 $j$ 对应的频率 $\theta_j = 10000^{-2j/d}$，j 小=高频（编码局部位置），j 大=低频（编码远距离位置）。NTK-aware 的思路是**不均匀缩放**：高频维度基本不动（因为局部位置关系已经在训练中充分学习，不需要扩展），低频维度大力拉伸（因为远距离感知范围需要扩大）。实际操作是将 RoPE base 从 10000 修改为 `10000 × scale^(d/(d-2))`。这在"大海捞针"（Needle-in-a-Haystack）测试上的表现远远优于 PI：32K 位置上的检索准确率通常 >95%（vs PI 的 <70%）。更重要的是，NTK-aware 不需要重新训练——直接改 base 就能用。

**(3) YaRN (Yet another RoPE extensioN)**：在 NTK-aware 的基础上加了两招：(a) 按波长（wavelength）分桶——将 RoPE 维度按 $\lambda_j = 2\pi / \theta_j$ 分组，波长小于 L_train 的维度（高频）保持原样，波长在 L_train 和 L_target 之间的做 NTK 缩放，波长大于 L_target 的（极低频）做 PI 式挤压（因为极低频维度在原始 4K 训练中采样不足，本身就没学好的东西再拉伸也没用）；(b) 加入"温度"参数 $t$ 来调节 softmax attention 的熵——长上下文下 attention 分布变得更 uniform（因为可选择的 token 变多了），引入温度补偿让高 attention score 的 token 更突出。YaRN 是目前长上下文外推的事实标准——LLaMA-3 的 8K→128K 扩展、Mistral 的 32K、Qwen 的 128K 均基于 YaRN 或其变体。

**实际部署效果对比（LLaMA-2-7B, 4K → 32K, 不训练）**：
- PI: PPL@32K = 8.9, Passkey retrieval@32K = 68%, 代码补全 pass@1 = 31%
- NTK-aware: PPL@32K = 5.6, Passkey retrieval@32K = 96%, 代码补全 pass@1 = 58%
- YaRN (s=8): PPL@32K = 5.1, Passkey retrieval@32K = 99%, 代码补全 pass@1 = 62%

YaRN 之所以最优，本质是因为它承认了一个事实：RoPE 的不同维度在训练中接收到的有效训练信号量不同（低频维度因为波长长，在 4K 训练样本中可能只经历 1-2 个完整周期，学习不充分），因此不应该均匀扩展。这是一个从"物理信号处理"视角来思考位置编码的精彩洞见。

**Q9（超高难度/Fellow Level）：你负责设计一个 LLM 推理服务系统，需要在 8×A100-80GB 上部署 4 个不同的 LLaMA-70B 微调变体（finance, medical, coding, general），共享相同的 base model chunk。请求是混合的且 QPS 约 200。请设计整体架构并论证每个子系统的选择。**

A: 这是一个典型的"多 LoRA adapter serverless"场景。核心矛盾：4 个完整的 70B 模型没法装进 8 张卡（每个 FP16 约 140GB，4 个共需 560GB，卡总容量仅 640GB），但 4 个 LoRA adapter（每个约 50MB）完全可以。

**架构设计：**

**第一层：Router / Dispatcher**
根据请求的 `model_type` 字段（由上游意图识别模块提供），将请求路由到对应的 adapter。这层最轻量——在 API gateway 层实现，不需要 GPU。

**第二层：Base Model Serving（核心挑战）**
使用 vLLM 作为 base engine，加载一个 FP16 LLaMA-70B（TP=4，占用 4 张 A100）。关键配置：
- `enable_lora=True` —— 启用 vLLM 的 LoRA adapter 支持
- `max_lora_rank=64` —— 预分配 LoRA 的 workspace memory
- `max_loras=4` —— 限制同时加载的 adapter 数量
- `max_cpu_loras=100` —— 热备 adapter 在 CPU RAM 中，可毫秒级切换到 GPU
- `fully_sharded_loras=True` —— LoRA 权重也跨 TP workers 分片，避免单个 GPU 成为 bottleneck

vLLM 的 LoRA 实现（v0.4.0+）会将 adapter 权重预先存储在 GPU 的 `punica` workspace 中。每个 decode step 时，根据当前请求的 `lora_request_id` 动态选择对应的 LoRA 矩阵（`lora_A`, `lora_B`），通过 `bgmv` kernel（batched GEMV）高效计算 adapter 贡献。关键 trick：vLLM 的 continuous batching 允许同一 batch 内混合不同 adapter 的请求——这是通过将 batch 按 adapter ID 分组，每组跑不同的 `bgmv` 参数实现的。

**第三层：KV Cache 管理**
这是多 adapter 共享 base model 的隐藏难点。Base model 的 self-attention 层参数是所有 adapter 共享的（KV projection 权重相同），因此 KV Cache 的 key/value 可以直接共享——不同 adapter 的请求可以在同一个 batch 内利用相同的 KV Cache。但有一个陷阱：如果不同 adapter 的 LoRA 应用在 `q_proj` 上，那 query 不同，导致 attention score 不同。解决方案：(1) LoRA 只应用在 `v_proj` 和 `o_proj`（保持 K 的一致性），(2) 或者接受 KV Cache 共享的微小精度损失（实测 <0.3% PPL）。

**第四层：GPU 资源分配**
8 卡分配方案：
- GPU 0-3：TP=4 的 LLaMA-70B base model + 4 Adapters（约 20GB adapter workspace）
- GPU 4-7：第二个独立的 TP=4 group，做 reduncancy + load balancing

如果 QPS=200（每个请求平均 500 tokens 输出 → 约 100K tokens/s），单个 TP=4 group 的吞吐上限约 50-80 tokens/s（取决于 input length），所以需要 2 个独立 serving group。也可以考虑 TP=8 单 group（更大的 batch size → 更高的 GPU 利用率），但 TP=8 的通信开销（all-reduce 跨 8 卡）约为 TP=4 的 1.8 倍——在 decode 阶段（memory-bound）这个开销尤其显著。经验上，TP=4 是 70B 模型的最佳 trade-off。

**第五层：Adapter 热更新与回滚**
生产环境中 adapter 需要频繁更新（比如金融模型每天吸收新的财报数据后重新微调）。vLLM 支持在线 `add_lora` / `remove_lora` API——新 adapter 加载到 CPU RAM 后 2-3 秒即可在 GPU 上可用（punica workspace 的热替换），期间不中断已运行的请求。结合 `max_loras=4` 的 GPU budget，可以做到"新 adapter 先在 CPU 上加载，通过 A/B testing 验证效果，确认无误后再将 GPU workspace 中的一个 slot 分配给新 adapter"——实现无缝切换。

**性能数据（参考）**：单 TP=4 A100-80GB 上，70B base model + 4 LoRA adapter，batch=64 的 MMLU benchmark（混合请求）：单请求 P50 延迟 320ms，P99 延迟 980ms，GPU 利用率 78%。不加 adapter 的纯 base model 在相同并发下 GPU 利用率 82%——LoRA overhead <5%。这是目前 Anchor、Anthropic、Google 内部多租户 LLM 服务的标准架构（Anthropic 的 Claude API 使用类似方案来同时服务不同技能的 model variant）。

## 10. 本讲总结

LLM 推理部署的核心矛盾是 **"模型太大，硬件太小"**。本讲系统梳理了五大优化方向：

1. **量化 (Quantization)**：SmoothQuant（激活量化，8-bit）→ AWQ/GPTQ（权重量化，4-bit）→ 两阶段将模型压缩 2-4x
2. **稀疏化 (Sparsity)**：Wanda（静态，weight+activation 联合判据）→ DejaVu（动态，contextual）→ 进一步压缩 1.5-2x
3. **KV Cache 管理**：PagedAttention 从操作系统借来分页思想，利用率 30% → 96%
4. **Attention 算子优化**：FlashAttention（IO-aware，2-4x 加速）→ FlashAttention-2/3（进一步优化 thread block 调度）
5. **解码策略**：Speculative Decoding（draft-verify 范式，2-3x 加速）

这五项技术在实际部署中通常组合使用——例如 vLLM + AWQ + FlashAttention 是当前最流行的开源 LLM 推理栈。理解了 memory wall 的本质，就会理解为什么这些技术不是"锦上添花"而是"雪中送炭"。

## 11. 工业落地checklist

| 检查项 | 说明 | 不做的后果 |
|--------|------|-----------|
| AWQ 量化 MoE 模型时必须为 gate network 单独设计校准策略 | Mixtral 8×7B 量化教训：gate 的 128 条 default calibration 样本不足以覆盖所有 8 个 expert 的激活模式——gate 被分配过小 scale，金融合规任务准确率从 91% 暴跌到 64% | 模型在特定 domain 完全不可用，量化投入的近一周工作白费 |
| vLLM prefill 的 max_num_batched_tokens 必须配合 max_num_seqs 和 GPU 型号联合调优 | A100 40GB 推荐 ≤ 8192，H100 80GB 推荐 ≤ 16384；某出行平台调到 32768 → prefill 单步延迟 5-8s → P0 事故 | 用户首 token 延迟 SLA 严重超标，需紧急回滚配置并重新压测 |
| vLLM prefix caching 上线前必须监控 hash collision rate 并做 GPU full block hash 校验 | 某电商平台开启 prefix_caching 后，相似 system prompt 间的 hash 碰撞导致错误 KV block 被复用，产生乱码输出——P0 事故 | 客服系统产生错误回答，用户投诉信任危机，需紧急下线 |
| 量化+稀疏化叠加时精度退化是非线性的，必须逐组合做消融测试 | 4-bit + 50% 稀疏通常不等效于 8x 压缩，叠加后精度塌陷远超单技术损失之和 | 多技术组合后模型 accuracy 雪崩到不可用，需数周排查交互效应 |
| Speculative Decoding 主模型更新时必须同步更新 draft model | 某平台 LLaMA-70B v2→v3 后 draft model 保持 v2，接受率从 82%→11%，端到端延迟反增 30% | speculative decoding 从加速变成拖慢，且问题隐蔽（loss 正常下降但端到端慢） |
| SmoothQuant 的 α 参数（激活到权重的转移比例）不能默认 0.5，需根据每层 outlier 严重程度分别调 | 不同层的激活 outlier 幅度差异大（embedding 层 vs deep layer），统一 α=0.5 使某些层量化误差远超其他层 | 深层精度损失累积导致最终输出偏差无法接受，量化方案需返工 |
| FlashAttention-2 部署前必须显式检查 head_dim % 16 == 0 | FA2 Triton kernel 对非 16 整除 head_dim 静默回退到 non-fused attention——不报错不警告，但比 xformers 慢 40% | 推理速度比预期慢 40%，排查 5 天才定位到是 head_dim 配置问题 |
