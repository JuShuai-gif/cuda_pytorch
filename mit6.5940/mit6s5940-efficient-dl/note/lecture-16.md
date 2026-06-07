# Lecture 16: Vision Transformer (ViT) — 当 Transformer 遇见图像

## 1. 本讲核心问题

Transformer 在 NLP 成功后，如何迁移到视觉领域？核心挑战——图像是像素的二维网格（显性结构），而 NLP 处理的是 token 序列（线性结构），两者有根本不同。ViT（Vision Transformer）如何将图像切分成 patches 作为"视觉词"？ViT 的变体如何解决其两大核心弱点——(1) 需要海量训练数据（因为没有卷积的归纳偏置），(2) 自注意力对高分辨率图像的计算爆炸（$O(n^2)$ 在 224×224 图像上就是 $n=196$，高分辨率时急剧膨胀）？高效的 ViT 设计（Window Attention、Linear Attention、Sparse Attention）如何控制计算？自监督学习（对比学习，Masked Image Modeling）如何补足 ViT 的数据需求？HART 如何实现自回归图像生成？ViT 部署的独特挑战有哪些？

## 2. 通俗解释

**什么是 Patch Embedding**：NLP 的"词"是显性的——"我爱编程"天然就是三个词。图像没有"词"——它是一个连续的二维像素矩阵。ViT 的做法是拿一把"菜刀"把图像切成很多方形小块（如 16×16 的 patch），每个 patch 像一个"词"。然后把每个 patch 里的像素拍扁成一个长向量，再通过一个线性投影（就是乘一个矩阵）得到 patch embedding。最后在最前面加一个特殊的 `[CLS]` token（类比 BERT），所有信息汇集到它那里做分类。

**为什么 ViT 比 CNN 更需要数据**：CNN 有"内置"的视觉偏置——它天然知道近邻像素关系重要（通过局部卷积核），也具有平移等变性（不管猫在图的左边还是右边，卷积核的处理方式一样）。这都是有益的归纳偏置（inductive bias）。ViT 的 Self-Attention 一开始完全"不看关系"——每个 patch 和所有其他 patch 的关系都得自己从头学。所以没有几百万甚至几亿张图片，ViT 学不会这些视觉基础规律。但一旦数据充足，ViT 比 CNN 更强——因为它不受卷积的局部性和平移不变性限制，能学到更灵活的视觉关系。

**Window Attention**：全图 Attention 太贵（$O(N^2)$），Window Attention 把图像分成不重叠的窗口（如 7×7），只在窗口内做 Attention。问题是——窗口之间没了交互。Swin Transformer 的解决方案是用"shifted windows"——两层之间错开窗口边界，让信息在窗口间流动（和 LongLoRA 的 $S^2$-Attn 同理）。

**Linear Attention**：Self-Attention 的 $O(n^2)$ 来源于 softmax——必须计算所有 $n^2$ 个 pairwise 得分。但如果把 softmax 换成可分解的 kernel 函数（如 $\phi(Q)\phi(K)^T$），就能先算 $\phi(K)^T V \in \mathbb{R}^{d \times d}$（与 $n$ 无关！），再算 $Q$ 乘它，复杂度降至 $O(n d^2)$。当 $n \gg d$ 时这是巨大优势——但在 ViT 中通常 $n$ 不大（196 for 224×224），优势有限。

**自监督学习（SSL）**：如果标注数据不够，就让模型在大量无标注图像上"自学"。两大流派：(1) **对比学习**（MoCo, SimCLR）——给模型两张图，让它判断是不是同一张图的不同视角/增强；学会的表征自然捕捉了视觉语义。(2) **掩码图像建模（MIM）**——像 BERT 一样，随机遮住 60% 的 patches，让模型猜被遮的部分是什么。MAE（Masked Autoencoder）就是这个思路的 ViT 版本。

**HART 自回归图像生成**：把图像也当成一种"语言"——将连续的像素值离散化为有限个 token（VQ-VAE 或类似 tokenizer），然后用 Transformer 以自回归的方式从左到右、从上到下逐个生成 image token。这等于用语言模型的方法来"写"图像。

## 3. 关键公式 (LaTeX)

### ViT: Patch Embedding

输入图像 $I \in \mathbb{R}^{H \times W \times C}$，切成 $N$ 个 $P \times P$ patch：

$$
N = \frac{HW}{P^2}
$$

每个 patch 压平为向量 $\mathbf{x}_p^i \in \mathbb{R}^{P^2 \cdot C}$，线性投影到 $D$ 维：

$$
\mathbf{z}_0 = [\mathbf{x}_{\text{class}}; \mathbf{x}_p^1 \mathbf{E}; \mathbf{x}_p^2 \mathbf{E}; \dots; \mathbf{x}_p^N \mathbf{E}] + \mathbf{E}_{\text{pos}}
$$

其中 $\mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}$ 为 patch embedding 投影矩阵，$\mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$ 为位置编码（可学习）。

### Window-based Self-Attention (Swin Transformer)

将 $N$ 个 patches 划分为 $M \times M$ 的不重叠窗口，每窗口有 $\frac{N}{M^2}$ 个 patches：

$$
\Omega_{\text{W-MSA}} = 4HWC^2 + 2M^2 HWC
$$


而标准 MSA：
$$
\Omega_{\text{MSA}} = 4HWC^2 + 2(HW)^2 C
$$

关键：MSA 的 $2(HW)^2 C$ 替换为 $2M^2 HWC$——复杂度从 $\mathcal{O}(N^2)$ 降为 $\mathcal{O}(N M^2)$。

Shifted Window：相邻层间窗口偏移 $(\lfloor M/2 \rfloor, \lfloor M/2 \rfloor)$，实现跨窗口信息交互。

### Swin Transformer Block（连续两层）

$$
\begin{aligned}
\hat{\mathbf{z}}^l &= \text{W-MSA}(\text{LN}(\mathbf{z}^{l-1})) + \mathbf{z}^{l-1} \\[2pt]
\mathbf{z}^l &= \text{MLP}(\text{LN}(\hat{\mathbf{z}}^l)) + \hat{\mathbf{z}}^l \\[2pt]
\hat{\mathbf{z}}^{l+1} &= \text{SW-MSA}(\text{LN}(\mathbf{z}^l)) + \mathbf{z}^l \\[2pt]
\mathbf{z}^{l+1} &= \text{MLP}(\text{LN}(\hat{\mathbf{z}}^{l+1})) + \hat{\mathbf{z}}^{l+1}
\end{aligned}
$$

### Linear Attention（以 Performer 为例）

用可分解 kernel 近似 softmax attention：

$$
\text{Attention}(Q, K, V) \approx \frac{\phi(Q)(\phi(K)^T V)}{\phi(Q)(\phi(K)^T \mathbf{1})}
$$

计算顺序：先算 $\phi(K)^T V \in \mathbb{R}^{d \times d}$，再左乘 $\phi(Q)$——复杂度 $O(N d^2)$。

常用 $\phi$ 函数：

$$
\phi(\mathbf{x}) = \frac{h(\mathbf{x})}{\sqrt{m}} \left[ f_1(\omega_1^T \mathbf{x}), \dots, f_m(\omega_m^T \mathbf{x}) \right]
$$

其中 $\omega_i \sim \mathcal{N}(0, I_d)$，$f_j$ 常用 $\exp$ 或 ReLU。

### MAE (Masked Autoencoder) 损失

给定被遮住 75% patches 的图像，重构被遮 patches 的像素值：

$$
\mathcal{L}_{\text{MAE}} = \frac{1}{|M|} \sum_{i \in M} \| \hat{\mathbf{x}}_i - \mathbf{x}_i \|^2_2
$$

其中 $M$ 为被遮 patches 的索引集。

### HART: 自回归图像生成

用 VQ tokenizer 将图像编码为离散 token 序列 $[t_1, t_2, \dots, t_S]$：

$$
p(t_1, \dots, t_S) = \prod_{s=1}^{S} p(t_s | t_1, \dots, t_{s-1})
$$

与 GPT 的 next-token prediction 完全一致。

## 4. 公式背后的直觉

**为什么 ViT 需要位置编码（CNN 不需要）**：CNN 的卷积操作天然具有位置信息——在 $(1,1)$ 处卷积和 $(10,10)$ 处的输出值不同，因为对应不同感受野。Self-Attention 是"permutation equivariant"——如果你把所有 patches 随机打乱，attention 输出也跟着打乱，不会改变。没有位置编码，ViT 无法区分"左上角的猫脸"和"右下角的猫脸"——所以位置编码是必需的。

**Window Attention 的互补直觉**：标准 MSA 的复杂度中 $(HW)^2$ 部分来自 QK^T 的注意力矩阵。限制在 $M \times M$ 窗口内（如 $M=7$, $HW=56^2=3136$），MA 矩阵从 $3136^2$ 降到每个窗口的 $7^2=49$ 内有 $49^2$ 个——对于 $56/7=8$ 个窗口，总复杂度 $= 8 \times 49^2 \approx 19,208$，而全图 MSA 是 $3136^2 \approx 9,834,496$——下降了 512x！代价是需要在不同层间 shift 窗口来保证跨窗口信息流。

**MAE 高遮罩率（75%→>85%）的直觉**：图像信息有大量冗余——遮住一大块草地，人仍知道那应该是草地。遮少了反而让模型"偷懒"（从邻居 patch 直接插值就能猜出）。高遮罩率迫使模型学习全局语义理解——必须分析"这只狗在草地上跑"的整体场景才能填回草地的像素细节。

**Linear Attention 在 ViT 中的尴尬**：$n=196$（224×224）时 $O(n^2)$ 其实不大（196²=38K），而 $O(n d^2)$ 中 $d^2$ 往往远大于 $n$（如 $d=768$, $d^2=590K$）。所以 Linear Attention 反而更慢！只有当分辨率极高（1024×1024 → $n=4096$, $n^2=16M$）时，Linear Attention 的优势才显现。

## 5. 工业界用途

| 技术 | 工业应用 |
|------|---------|
| **ViT (基础)** | 图像分类（ImageNet SOTA）、Google 多模态模型基础 |
| **Swin Transformer** | 目标检测（COCO）、语义分割（ADE20K）的主流 backbone；视频理解的 Video Swin |
| **DINO / DINOv2** | 自监督视觉特征提取（Meta）；图像检索、语义匹配 |
| **MAE** | 医学图像分析（标注数据稀缺）；遥感图像预训练 |
| **HART / 自回归生成** | 图像生成（Parti, Muse）；可控图像编辑 |
| **Contrastive SSL (CLIP)** | 多模态（OpenAI CLIP）；零样本分类；DALL-E 的文本-图像对齐基础 |
| **Window Attention** | 几乎所有高分辨率 ViT 的标准组件（Swin, MaxViT, EfficientViT） |
| **高效 ViT** | 手机端视觉（MobileViT, EfficientFormer）；实时视频分析 |

## 6. PyTorch 实现思路

### ViT Patch Embedding + 基本 Block

```python
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """Split image into patches and embed them."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):  # (B, 3, H, W)
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class ViTBlock(nn.Module):
    """Standard ViT block: MHA + MLP with pre-LN."""
    def __init__(self, dim, n_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x
```

### Window Attention (Swin 风格)

```python
def window_partition(x, window_size):
    """Partition input into non-overlapping windows.
    x: (B, H, W, C) → (B * n_windows, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """Reverse window partition."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias."""
    def __init__(self, dim, window_size, n_heads, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), n_heads)
        )

    def forward(self, x, mask=None):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, n_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # Add relative position bias
        attn = attn + self._get_relative_position_bias(N)
        if mask is not None:
            attn = attn + mask
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)
```

### MAE: Masked Autoencoder 核心

```python
def random_masking(x, mask_ratio=0.75):
    """Randomly mask patches. Keep only visible patches for encoder."""
    B, N, D = x.shape
    len_keep = int(N * (1 - mask_ratio))
    noise = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    ids_mask = ids_shuffle[:, len_keep:]
    # Keep only visible patches
    x_visible = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
    return x_visible, ids_mask, ids_keep


class MaskedAutoencoder(nn.Module):
    def __init__(self, encoder, decoder, mask_ratio=0.75):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = mask_ratio

    def forward(self, imgs):
        # Encode only visible patches
        x = self.patch_embed(imgs)
        x_visible, ids_mask, ids_keep = random_masking(x, self.mask_ratio)
        latent = self.encoder(x_visible)

        # Decode: concat visible latents + mask tokens
        mask_tokens = self.mask_token.expand(imgs.shape[0], ids_mask.shape[1], -1)
        # Unshuffle to original order
        all_tokens = torch.cat([latent, mask_tokens], dim=1)
        # ... restore original order using ids_keep and ids_mask ...
        pred = self.decoder(all_tokens)
        # Compute loss only on masked patches
        loss = ((pred - target) ** 2).mean()
        return loss, pred
```

### EfficientViT (Mobile ViT)

```python
class EfficientViTBlock(nn.Module):
    """Efficient ViT block combining local conv and lightweight attention."""
    def __init__(self, dim, n_heads, window_size=7, expand_ratio=4):
        super().__init__()
        # Lightweight multi-scale attention with small window
        self.attn = WindowAttention(dim, window_size, n_heads)
        # Depth-wise conv for local feature extraction
        self.local_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        # FFN
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * expand_ratio, 1),
            nn.GELU(),
            nn.Conv2d(dim * expand_ratio, dim, 1),
        )

    def forward(self, x):
        # x: (B, H, W, C)
        x = x + self.local_conv(x.permute(0,3,1,2)).permute(0,2,3,1)
        x = x + self.attn(x)
        x = x + self.ffn(x.permute(0,3,1,2)).permute(0,2,3,1)
        return x
```

## 7. TinyML / Edge AI 部署意义

**ViT 部署的特殊挑战**：
- ViT 的 Self-Attention 需要频繁的矩阵转置和 reshape（`view`/`permute`），对 NPU/TPU 不友好
- 位置编码使动态分辨率处理困难——固定 224² 训练的 ViT 无法直接处理 640² 输入而不改变精度
- CNN 的 `im2col`+GEMM 范式在硬件上高度优化，而 Attention 的 softmax+matmul 在边缘 NPU 上支持不完善

**Mobile ViT 的实际方案**：
- MobileViT / EfficientFormer：CNN 负责大部分层（卷积在边缘硬件上极快），ViT 只在最后几层做全局推理
- EdgeNeXt：将 Depthwise Conv 与 split attention 结合，最小化 attention 使用
- 量化：ViT 中 LayerNorm + softmax 在高比特时数值稳定，低比特（INT8/INT4）需要特殊处理（如 INT8 GELU via LUT）

**部署框架选择**：
- Apple Core ML：对 ViT 的 attention 操作有原生优化（ANE）
- Qualcomm SNPE：MobileViT 系列支持最好
- ONNX + TensorRT：Swin Transformer 的 window attention 需要自定义插件

## 8. 常见误区

> ❌ **误区 1："ViT 直接代替 CNN 做所有视觉任务"**
> ViT 在数据不足时远不如 CNN（缺少归纳偏置），在高分辨率任务上计算量爆炸。实践中：数据少用 CNN/ConvNeXt，数据多用 ViT/Swin；实时任务中 CNN 仍占主导。

> ❌ **误区 2："Window Attention 和 Swin 的 Shifted Window 等同于全图 Attention 效果"**
> 多次 shift window 扩大了感受野，但仍不是真正的全局 Attention——token 信息需要通过多层的逐窗口间接传播，类似 CNN 的感受野逐层扩大。对于需要"跨图关系"的任务（如目标追踪），纯 window attention 可能不够。

> ❌ **误区 3："MAE 的预训练任务只是图像重建"**
> 关键是 MAE 只重建被遮住的 patches（而非全部），且遮罩率高（75%+）。这让模型被迫学语义理解而非像素插值。MAE 本质是"让模型在极度信息缺失下做推理"。

> ❌ **误区 4："Linear Attention 在所有 ViT 场景下都比标准 Attention 快"**
> 在标准 224²（n=196）下，Linear Attention 计算量中的 $d^2$ 常数因子大于 $n^2$，反而更慢。只有在极高分辨率（≥1024²）时才显出优势。

> ❌ **误区 5："ViT 不需要 CNN 的 inductive bias，因为它更通用"**
> 更通用意味着更需要数据来"教会"这些偏置。ImageNet-1K（120 万张）对 ViT 不够，需要 ImageNet-21K（1400 万）或 JFT-300M（3 亿）。对大多数非互联网巨头来说，这是无法获取的数据量。

## 9. 面试问题

**Q1: ViT 如何将 224×224×3 的图像转换为 token 序列？输出序列长度是多少？**
A: 用 16×16 的卷积核（stride=16）同时做分块和投影，得到 14×14=196 个 token。加上 `[CLS]` token 后，序列长度为 197。每个 token 维度为 embed_dim（如 768）。

**Q2: Swin Transformer 如何在不做全图 Attention 的情况下扩大感受野？**
A: (1) 层级结构——通过 patch merging 逐步降低分辨率（类似 CNN pooling），同时扩大每个 token 对应的原图面积；(2) Shifted Window——相邻两层的窗口位置错开，token 通过窗口内 attention 与不同邻居交互，逐层扩大有效感受野。

**Q3: MAE 高遮罩率（>75%）为什么比低遮罩率训练效果更好？**
A: 低遮罩率下模型可以从相邻可见 patch 直接插值推测被遮内容，不需要语义理解；高遮罩率迫使模型利用全局布局和物体类别知识来重建，学到的表征更有语义性、迁移能力更强。

**Q4: DINO 和 MAE 作为自监督方法的核心区别？**
A: (1) **DINO（对比学习）**：通过 teacher-student 框架学习全局表征——同一张图的两个随机增强应产生相似表征。学到的特征天然适合语义分割和检索。(2) **MAE（MIM）**：通过重建被遮 patches 学习"看图填空"。学到的特征更适合需要局部细节和像素级理解的下游任务。

**Q5: HART 自回归图像生成和扩散模型（如 Stable Diffusion）的核心区别？**
A: HART 把图像当作离散 token 序列——用 Transformer 预测"下一个 token"来逐步生成图像，类似 GPT 写文本。扩散模型把生成看作逐步去噪——先加噪到纯噪声再学习逆向去噪。自回归生成的优势是架构统一（同一 Transformer 既处理文本又处理图像），劣势是逐 token 生成本质上比扩散的并行采样慢。

**Q6: 为什么在边缘设备上 EfficientFormer 比 Swin-T 更快？**
A: EfficientFormer 用 MetaBlock 设计——早期层用卷积（边缘硬件优化好），后期才引入 attention（全局推理但层数少）。卷积的计算模式（滑窗）对 NPU 的并行计算和内存访问模式天然友好，而 attention 需要大量 reshape/transpose 和 softmax，增加了硬件利用难度。

## 10. 本讲总结

ViT 把图像做成了语言模型的食物——Patch Embedding 代替 tokenization，Self-Attention 代替 convolution。但这顿"饭"的代价是巨大的：**数据饥渴**（没有 CNN 的归纳偏置需要更多数据）、**计算饥渴**（$O(n^2)$ attention 让高分辨率图像的算力需求暴涨）、**硬件不友好**（reshape/transpose 在边缘设备上笨拙）。

本讲的解决方案围绕三条线索：
1. **效率 ViT**：Window Attention（计算定位到局部）→ Swin's Shifted Window（信息流动跨窗口）→ EfficientFormer（CNN+ViT 混合，边缘友好）
2. **数据不足**：对比学习（DINO, CLIP）→ MIM（MAE）→ 自监督预训练 + 下游微调
3. **注意力线性化**：Linear Attention → Sparse Attention → 在高分辨率场景下选择性应用

关键认知：**CNN 和 ViT 不是敌人，而是工具箱中的互补工具**。CNN 在低数据、高实时性场景仍是王者；ViT 在大数据、高灵活性的前沿任务上超越 CNN。现代最佳实践（ConvNeXt, EfficientFormer, MaxViT）往往是两者的杂交体。

下一讲探讨 Transformer 在更特殊的模态——GAN、视频和点云上的效率优化挑战。
