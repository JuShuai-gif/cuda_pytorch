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

## 10. 本讲总结

LLM 推理部署的核心矛盾是 **"模型太大，硬件太小"**。本讲系统梳理了五大优化方向：

1. **量化 (Quantization)**：SmoothQuant（激活量化，8-bit）→ AWQ/GPTQ（权重量化，4-bit）→ 两阶段将模型压缩 2-4x
2. **稀疏化 (Sparsity)**：Wanda（静态，weight+activation 联合判据）→ DejaVu（动态，contextual）→ 进一步压缩 1.5-2x
3. **KV Cache 管理**：PagedAttention 从操作系统借来分页思想，利用率 30% → 96%
4. **Attention 算子优化**：FlashAttention（IO-aware，2-4x 加速）→ FlashAttention-2/3（进一步优化 thread block 调度）
5. **解码策略**：Speculative Decoding（draft-verify 范式，2-3x 加速）

这五项技术在实际部署中通常组合使用——例如 vLLM + AWQ + FlashAttention 是当前最流行的开源 LLM 推理栈。理解了 memory wall 的本质，就会理解为什么这些技术不是"锦上添花"而是"雪中送炭"。
