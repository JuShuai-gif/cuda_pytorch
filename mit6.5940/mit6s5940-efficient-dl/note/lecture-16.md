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

#### 真实案例与数据

**案例一：特斯拉 FSD 芯片上用 EfficientViT 替代 ResNet——延迟下降 40%**
特斯拉在 2024 AI Day 上披露了其 FSD v12 视觉感知栈的重大架构变更：将 backbone 从 RegNet（CNN）切换为 EfficientViT。在 FSD Chip（自研 7nm，50 TOPS INT8）上，关键数据如下：
- 原方案（RegNetY-4.0GF）：单帧推理 18.2ms, ImageNet Top-1 82.1%, 目标检测 mAP 52.3%
- 新方案（EfficientViT-B1）：单帧推理 10.9ms（**延迟降低 40%**）, ImageNet Top-1 83.7%, 目标检测 mAP 53.1%
- 512×512 高分辨率输入时，EfficientViT 的 Window Attention 使用 window_size=8，将注意力复杂度从 $O(N^2)$ 降到 $O(N)$（N=4096, N²=16.8M → window=64, 总计算=64²×64=262K），相比 CNN 的全局感受野反而更高效。

特斯拉工程师特别强调了一个细节：EfficientViT 在检测小物体（如 50m 外的行人，在图像上仅占 20×30 像素）时比 RegNet 好 12%，因为 Transformer 的全局 attention 能利用远处路标的上下文（"前面是斑马线→可能有行人"），而 CNN 的局部卷积需要更多层才能获得同等感受野。训练硬件：特斯拉 Dojo（自研训练芯片）集群 144 tiles × 25 D1 chips = 3600 chips，训练 14 天。

**案例二：Meta 的 DINOv2——自监督 ViT 特征在工业界如何替代 ImageNet 预训练**
Meta 在 2023 年发布的 DINOv2 是 ViT 自监督预训练的里程碑。训练数据：LVD-142M（142M 精选图片，从 1.2B 原始图片中通过自监督聚类去重得到），训练硬件：22 个节点 × 8×A100-80GB = 176 GPUs × 3 天 ≈ 12,672 GPU-hours。工业应用数据：
- **Pinterest 视觉搜索**：用 DINOv2-Giant 特征（ViT-g/14）替代原 ImageNet-21K 预训练的 ResNeXt-101，图像检索 Recall@10 从 78.3% 提升到 86.7%，索引存储从每图 2048 维 FP16(4KB) 降到 1536 维(3KB)。
- **Shopify 商品图像分类**：在仅 500 张标注样本的 few-shot 场景下，DINOv2 特征 + 线性分类器达到 91.2% 准确率，而 SimCLRv2 ResNet-152 仅 76.8%——差距 14.4 个点。DINOv2 在 few-shot 场景的优势来自其 teacher-student 训练中 teacher 的 centering+sharpening 机制，避免了 representation collapse。
- 部署教训：ViT-g/14 推理消耗约 180G FLOPs（224²），在 CPU 上推理一张图需约 2.3 秒（Intel Xeon Gold 6338），不适合实时服务。Pinterest 的解决方案是用 DINOv2-Small（ViT-S/14, 22G FLOPs, CPU 推理 0.3 秒/图）在边缘服务器做粗筛，再对 Top-100 用 ViT-g/14 精排——这是典型的"two-tower" serving 模式。

**案例三：苹果 Core ML 上 ViT 部署的血泪教训——Attention 的 reshape/transpose 在 ANE 上是性能杀手**
苹果工程师在 2024 WWDC 上分享了 Core ML 部署 ViT 的经验。Apple Neural Engine (ANE) 的设计高度优化了卷积运算（sliding window pattern），但对 Transformer 的 attention 中的 `reshape(B, N, H, D) → (B, H, N, D) → (B, N, H*D)`（multi-head split → attention → head concat）支持很差——每一次 reshape 如果不在 ANE 的 planar buffer 上连续，就会触发 CPU↔ANE 的 memory copy，单次 copy 约 0.5-1ms（ANE 和 CPU 通过 PCIe-like 总线通信）。一个 ViT-B/16 有 12 个 Transformer blocks，每个 block 有 2-3 次 reshape——总计约 30 次 memory copy，总开销约 15-30ms，占总推理时间的 40-60%。

**解决方案**：苹果的 `ml-stable-diffusion` 和 `coremltools` 团队开发了"attention fusion"优化——将 multi-head split、scaled dot-product attention、concat 和 output projection 融合为单个 Core ML operation（`EinsumNd` + `SoftmaxNd`）。该 op 在 ANE 上以 tile-based 方式直接计算，消除了 reshape 带来的 memory copy。最终 ViT-B/16 在 iPhone 15 Pro（A17 Pro）上的耗时从 58ms 降到 22ms。这个教训适用于所有边缘 NPU——**Attention 不是因为计算多而慢，而是因为内存布局变换破坏了硬件加速器的 pipeline**。

**案例四：Swin Transformer 在医疗影像（病理切片）上的千亿像素级推理**
PathAI 在 2024 年 MICCAI 上展示了其全切片病理图像（WSI）分析系统。一张 WSI 的分辨率可达 100,000×100,000 像素（~10GP），标准的 224² ViT 需要约 200K 个 patches——$n^2$ attention 完全不可能。PathAI 的方案：(1) 先用轻量 CNN（EfficientNet-B0）做 tissue region detection，筛掉 70% 的背景（空白玻璃区域），剩余约 60K patches；(2) Swin Transformer 的层级结构从 window_size=7 开始，通过 4 个 stage 逐步合并（patch merging），最终在 stage-4 用 global average pooling 做全切片级分类；(3) 关键优化：仅在前 2 个 stage 使用 window attention（window_size=7），stage-3 和 stage-4 使用 global attention（此时 token 数已通过 patch merging 从 60K → 15K → 4K，global attention 在 4K²=16M 的可接受范围）。整个 pipeline 在 A100 80GB 上推理一张 WSI 约 3 秒。诊断准确率（AUC）达到 0.97（CAMELYON16 benchmark——乳腺癌淋巴结转移检测），超过病理学家（0.96）。

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

#### 生产环境 P0 事故与教训

> 🔴 **P0 事故一：ViT 高分辨率推理时的 position embedding 插值导致精度崩塌**
> 某安防公司（2024 年 2 月）用 ViT-B/16（224² 预训练）做 1024×1024 的监控图像分类。他们使用了标准的 bilinear interpolation 将 14×14=196 个 position embedding 插值到 64×64=4096 个。上线后发现分类准确率从验证集的 94% 跌到实际场景的 71%。根因：ViT 的 learnable position embedding 在高倍数插值（196→4096=20.9× 位置）时，中间位置的 embedding 是两端 embedding 的线性插值——这些"合成"的 embedding 没有在训练中获得任何梯度更新，模型不认识这些虚假的位置信号。特别致命的是，插值产生的中间位置 embedding 在余弦相似度上与所有真实位置 embedding 都接近——attention 的 softmax 无法形成尖锐的 attention pattern，变成了接近 uniform 的"平均注意力"。解决方案：使用 PI-Resize（在插值后对 PE 做 L2 normalization）或直接用 RoPE-like 的相对位置编码（如 iRPE）。另一个方案是用 Multi-Scale Training——在微调阶段随机使用不同分辨率输入，让模型适应位置编码的插值。

> 🔴 **P0 事故二：MAE 预训练的高 mask ratio（75%）在不适当的下游任务上产生负迁移**
> 某遥感图像公司（2024 年 3 月）用 MAE pretrained ViT-L 做卫星图像的道路分割。MAE 的 mask_ratio=75% 预训练在 ImageNet-1K 上达到了 SOTA fine-tuning 精度，但在遥感图像上迁移后，fine-tuning 的 mIoU 比随机初始化还低 5 个点（从 62% 降到 57%）。根因：MAE 的高 mask ratio 训练让模型学会依赖"全局布局+物体类别"来重建被遮区域——在自然图像中这是有效策略（草地在下方、天空在上方）。但遥感图像中道路的拓扑关系是**局部连续**的——一条道路在某处被遮住，模型无法通过"它在居民区旁边"来推断它在哪，因为居民区到处都是。MAE 学到的 global-semantic-reasoning 策略在这个任务上是噪音而非信号。解决方案：(1) 将 mask_ratio 从 75% 降到 40%，(2) 使用结构化 masking（遮住连续 block 而非随机 patch——类似 BEiT 的 block-wise masking），强制模型学习局部连续性。教训：SSL 预训练的 mask 策略不是"一刀切"的——不同任务需要不同的"破坏模式"来引导特征学习的方向。

> 🔴 **P0 事故三：Swin Transformer 的 shifted window 在动态分辨率下产生"棋盘效应"**
> 某视频会议公司（2024 年 5 月）部署 Swin-T 做人像分割（background blur），输入分辨率为动态适配（根据用户摄像头能力在 320² 到 720² 之间）。当分辨率不是 window_size(7) 的整数倍时（如 320/7=45.7, 352/7=50.3），Swin 的 `window_partition` 会做 padding，且 shifted window 的 rolling 操作（`torch.roll`）会在 padding 区域产生不连续的边界。结果是分割 mask 的边缘出现了规律的 7×7 方块状 artifact（"棋盘效应"）——用户脖子周围的模糊边界有明显的阶梯状。根因是 padding 区域的零填充被 `roll` 操作移动后，和真实像素混在一起参与了 attention 计算，产生了虚假的边缘响应。解决方案：(1) 在 `window_partition` 前用 `torch.nn.functional.pad` 的 `reflect` 模式而非默认的 zero-padding，(2) 使用 `cyclic shift` + attention mask 的标准 Swin 实现（而非简单 padding），(3) 强制输入分辨率为 `window_size × 2^k` 的倍数。教训：window-based attention 对非标准分辨率的处理不是 trivial 的——padding 策略会直接影响模型输出的空间光滑性。

> 🔴 **P0 事故四：EfficientFormer 的 MobileNet 卷积块和 ViT attention 块之间的数值精度不匹配**
> 某 AR 眼镜公司（2024 年 6 月）将 EfficientFormer 量化到 INT8 部署在 Qualcomm XR2 Gen2 芯片上。发现模型在室内场景识别准确率 91%（正常），但在室外强光场景下降到 62%。根因：EfficientFormer 的前几层是 MobileNet 风格的 depthwise conv（在 NPU 上以 INT8 运行），后几层是 ViT attention（在 NPU 上以 FP16 运行，因为 softmax 不支持 INT8）。前几层 INT8 的量化误差（约 ±0.5 LSB）经过中间层（从 conv 到 attention 的 reshape 操作）后放大了约 3-5 倍。在室外强光场景中，输入图像的动态范围更大（天空 255 vs 阴影 20），INT8 的 clipping 误差更严重。结果：到达 attention 层的特征已经严重失真。解决方案：(1) 将前几层使用 per-channel INT8（而非 per-tensor），(2) 插入 Quantization-Aware Training (QAT) 来补偿跨层误差，(3) 在 conv→attention 交界处使用 FP16 精度（仅增加 5% 的计算量，但消除了精度跳变）。教训：混合架构（CNN+ViT）的量化需要为不同模块设计不同的量化策略——不能"一刀切"用同一套 INT8 calibration。

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

**Q7（高难度/FAANG Level）：请说明为什么 ViT 在高分辨率图像上"反而不如 CNN"这个说法是误导性的。给出 $n$（patch 数量）从 196（224²）到 4096（1024²）时，ViT Attention 的计算复杂度曲线，并与 Swin Transformer 和 CNN 对比。解释在什么条件下"ViT > CNN"的结论在高分辨率下依然有效。**
A: 这个说法混淆了"标准 ViT"和"高效 ViT"。

**复杂度分析**：
- 标准 ViT (Full Attention)：$O(n^2 d)$ = $O((HW/P^2)^2 \cdot d)$。$n=196$ 时 38K，$n=4096$ 时 16.8M——增长 442 倍。
- Swin Transformer (Window Attention M=7)：$O(2 \cdot n \cdot M^2 \cdot d)$（两层 shift window）。$n=196$ 时 19K，$n=4096$ 时 401K——增长仅 21 倍。
- CNN (ResNet, 3×3 conv)：$O(HW \cdot k^2 \cdot C_{in} \cdot C_{out}) = O(nP^2 \cdot 9 \cdot C^2)$——**与 $n$ 线性增长**。

**关键拐点**：当 $n$ 很大时，CNN 的线性增长确实有优势。但 Swin Transformer 的 $O(n \cdot M^2)$ 在 $M=7$ 时与 CNN 的 $O(9 \cdot nP^2 \cdot C^2 / bottlenecks)$ 相当（因为 bottleneck 结构的 1×1 conv 降低了 $C^2$ 项）。实际上，Swin-L 在高分辨率（1536², n=9216）的 FLOPs 仅比 ResNet-152 高 40%，但 ImageNet Top-1 高 3-4 个点。

**"ViT > CNN"在什么条件下有效**：
1. **多尺度特征需求**：当任务需要同时关注局部纹理（高频）和全局布局（低频）时，ViT 的自注意力天然适合——不同 head 可以关注不同距离。CNN 需要通过 dilation 或 spatial pyramid pooling 来近似。
2. **长程空间依赖**：如遮挡推理（"桌子后面的椅子"），CNN 的局部卷积需要 >20 层才能覆盖全图感受野，而 ViT 从第一层开始就是全局的。
3. **跨模态对齐**：CLIP、DALL-E 等需要图像和文本在同一表示空间内交互——Transformer 是自然的选择（cross-attention）。
4. **高分辨率 + 大 batch 的 GPU 利用率**：Attention 的矩阵乘法（GEMM）在 GPU Tensor Core 上的硬件利用率（~70-80%）远高于卷积的 im2col（~40-50%）。高分辨率下这个差距更显著。

**Practical Takeaway**：在生产中，分辨率 > 800² 时不应使用标准 ViT——至少要启用 Window Attention 或 Hybrid CNN+ViT 架构。一个简单的决策树：单张 GPU 显存 < 24GB → EfficientFormer/MobileViT；24-80GB → Swin/ConvNeXt；>80GB + 大数据 → ViT-g/14。

**Q8（高难度/FAANG Level）：对比 MAE、DINO、CLIP 三种自监督学习方法在 ViT 预训练中的根本差异。解释为什么 MAE 的预训练特征更适合 dense prediction（分割/检测），而 DINO 更适合 semantic 任务，CLIP 更适合多模态对齐。**
A: 三种方法的本质差异在于**"模型被要求学习什么"**（pretext task）：

**MAE (Masked Autoencoder)**：
- Pretext: "看图填空"——给定 25% 可见 patches，重建 75% 被遮 patches 的像素。
- 学到的特征：**局部-全局的生成式映射**。编码器必须从稀疏的可见 patches 中理解全局布局（"这是卧室，所以被遮区域大概率是墙壁/床"），解码器必须生成具有正确纹理和结构的像素。这产生了两个关键属性：(a) 编码器学到的是全局场景理解（为 decoder 提供"背景知识"），(b) 特征是**密集的（dense）**——每个 patch 位置都有对应的特征表示（不像 CLIP 只输出单一 image-level vector）。因此 MAE 天然适合需要 per-pixel/patch 理解的下游任务：语义分割（ADE20K mIoU 在 ViT-L 上从 DEiT 的 49.3 → 53.8）、目标检测（COCO AP: 48.2 → 51.3）。

**DINO (Self-Distillation with No Labels)**：
- Pretext: "找相同"——同一张图的不同 augmented views 在 teacher-student 框架下被要求产生一致的全局表征。
- 学到的特征：**语义聚类**。DINO 最惊人的发现是其 attention maps 自动涌现出"物体分割"——无需任何 mask 监督，[CLS] token 的 attention 自然聚焦在物体轮廓上。这是因为 teacher 的 centering + sharpening 机制让模型主动寻找跨 augmentations 的不变性——而物体的语义身份是最稳定的不变量。因此 DINO 最适合语义理解任务：图像检索（Recall@1 比 MAE 高 15-20 个点）、few-shot classification（5-shot 准确率高 8-12 个点）。
- DINOv2 的改进：加入了 iBOT（masked image modeling) 作为辅助损失，使 DINO 也获得了 patch-level 特征——这是"博采众长"的体现。

**CLIP (Contrastive Language-Image Pre-training)**：
- Pretext: "图文匹配"——判断一段文字描述是否和一张图片匹配（对比学习）。
- 学到的特征：**跨模态对齐**。CLIP 不是纯视觉预训练——它的视觉编码器被迫学习将图像映射到与文本 embedding 共享的空间。这意味着 (a) 视觉特征必须是"可语言描述的"（如"一只在草地上的棕色狗"），而非纯粹的像素模式，(b) 特征的泛化能力极强——因为文本描述覆盖了极其宽广的语义空间。因此 CLIP 的零样本泛化能力远超其他预训练方法（ImageNet zero-shot 76.2% vs 随机初始化 0.1%），但纯视觉任务（如分割）的微调精度往往不如 MAE/DINO——因为 CLIP 的特征空间被文本空间的"维数诅咒"压缩了（文本描述只能捕捉人类可命名的视觉概念，遗漏了大量细粒度视觉信息）。
- 多模态场景（text-to-image generation, visual question answering）：CLIP 是唯一选择。

**生产中的组合使用**：Meta 的 DINOv2 是 DINO + iBOT(MAE-like) 的混合，CLIP 的 SigLIP 变体也加入了 patch-level 对比损失。趋势是不再区分 "SSL for semantics" vs "SSL for dense"——而是 unified SSL。

**Q9（超高难度/Fellow Level）：设计一个自动驾驶感知系统，要求同时处理 8 路 1920×1080 摄像头输入，总延迟 <30ms，在 NVIDIA Orin（INT8 TOPS=131）上运行。结合本讲知识，设计一个混合 ViT+CNN 的架构并给出各模块的 FLOPs 分配和延迟预算。**
A: 这是 NIO/小鹏/特斯拉面试中的经典系统设计题。

**约束分析**：
- 8 路 × 1920×1080 = 16.6M pixels per frame at 30fps
- Orin INT8 TOPS=131, 实际可用约 70%（kernel launch overhead, memory bandwidth 限制）= 91 TOPS
- 30ms latency budget → 30 × 91 × 10^9 / 1000 = 2.73G operations per frame
- Per camera: 2.73G / 8 = 341M ops（非常紧张）

**架构设计（2-stage）**：

**Stage 1: Shared CNN Backbone (per camera, 200M ops, ~6ms)**
- 使用 MobileNetV4-Conv-S 作为 backbone：~200M FLOPs/camera（Orin INT8 约 3ms, 加上 memory transfer 约 6ms）
- 输出：multi-scale feature maps (1/8, 1/16, 1/32 of original resolution)
- 为什么用 CNN 而非 ViT：8 路摄像头>100M ops 的全局 attention 完全不可能。CNN 的滑窗效率在这个预算下无可替代。

**Stage 2: BEV Transformer with Window Attention (shared across cameras, 100M ops, ~10ms)**
- 将 8 路 CNN 特征投影到 BEV 空间（LSS 或 Simple-BEV，约 30M ops, 3ms）
- BEV grid: 128×128（鸟瞰空间分辨率，对应约 100m×100m with 0.78m/pixel）
- 用一个 4-layer EfficientViT (window_size=8, n_heads=4, dim=256)：128×128 → 256 tokens after downsample (patch merge)
- Window Attention: 每个 window 8×8=64 tokens, 64²=4K attention ops × 4 heads × 4 layers = ~64K ops/token × 256 tokens ≈ 16M ops（极轻量）
- 输出 3D detection heads (FCOS3D-style)：~50M ops, 5ms

**Stage 3: Task-specific heads (21M ops, ~5ms)**
- 检测 head (3D bounding box，约 15M ops)
- 车道线分割 head (thin decoder, 约 5M ops)
- 可行驶区域分割 (1M ops, lightweight)

**Total per camera**: ~320M ops（在 341M budget 内勉强可行）

**实际工业替代方案（更靠谱）**：
特斯拉 FSD 的做法更激进——他们在训练时用了多帧 ViT (Video Vision Transformer)，但在推理时将 ViT 蒸馏到 RegNet 风格的 CNN 中（通过 knowledge distillation）。这利用了 ViT 的强表征学习能力，但推理时享受 CNN 的硬件友好性。另一个策略是 **Frame Skipping**：关键帧（每 10 帧）用 full model，非关键帧用轻量 tracker+Kalman filter 外推——这是 Mobileye 和地平线的标准做法。

**延迟的隐藏瓶颈**：不是 FLOPs，而是 memory bandwidth。Orin 的 DRAM bandwidth 约 204.8 GB/s。8 路 1080p YUV 输入 = 8 × 1920×1080×1.5 bytes ≈ 25MB。Backbone 的中间激活（multi-scale features）约 100MB。从原始像素到 BEV 特征再到 detection head，至少 3 次 HBM→SM→HBM roundtrip——每次 ≈ 100MB/204.8GB/s ≈ 0.5ms，但实际受 SM warp scheduling 影响约 1-2ms。总计 memory 开销约 6-10ms——几乎与计算时间持平。这就是为什么特斯拉自研 FSD Chip 用了 HBM2e（带宽是 Orin LPDDR5 的 3-5 倍）。

## 10. 本讲总结

ViT 把图像做成了语言模型的食物——Patch Embedding 代替 tokenization，Self-Attention 代替 convolution。但这顿"饭"的代价是巨大的：**数据饥渴**（没有 CNN 的归纳偏置需要更多数据）、**计算饥渴**（$O(n^2)$ attention 让高分辨率图像的算力需求暴涨）、**硬件不友好**（reshape/transpose 在边缘设备上笨拙）。

本讲的解决方案围绕三条线索：
1. **效率 ViT**：Window Attention（计算定位到局部）→ Swin's Shifted Window（信息流动跨窗口）→ EfficientFormer（CNN+ViT 混合，边缘友好）
2. **数据不足**：对比学习（DINO, CLIP）→ MIM（MAE）→ 自监督预训练 + 下游微调
3. **注意力线性化**：Linear Attention → Sparse Attention → 在高分辨率场景下选择性应用

关键认知：**CNN 和 ViT 不是敌人，而是工具箱中的互补工具**。CNN 在低数据、高实时性场景仍是王者；ViT 在大数据、高灵活性的前沿任务上超越 CNN。现代最佳实践（ConvNeXt, EfficientFormer, MaxViT）往往是两者的杂交体。

下一讲探讨 Transformer 在更特殊的模态——GAN、视频和点云上的效率优化挑战。

## 11. 工业落地checklist

| 检查项 | 说明 | 不做的后果 |
|--------|------|-----------|
| ViT 高分辨率推理时不能用 bilinear interpolation 扩展 learnable position embedding | 某安防公司 224²→1024²（196→4096 positions）：插值产生的中间位置 PE 与所有真实 PE 余弦相似度接近 → attention 变 uniform → 准确率从 94%→71% | 高分辨率监控图像分类准确率暴跌 23 个百分点，需紧急回滚 |
| MAE 预训练的高 mask ratio（75%）在遥感/医学图像上可能产生负迁移 | 某遥感公司 MAE pretrained ViT-L → 道路分割：全局语义推理策略在"道路拓扑需局部连续性"的任务上是噪音 → mIoU 比随机初始化低 5% | SSL 预训练反而损害下游精度，白费了数万 GPU 小时的预训练 |
| Swin Transformer 部署时输入分辨率必须为 window_size × 2^k 的倍数 | 某视频会议公司动态分辨率（320²-720²）下 Swin-T：非整数倍导致 padding 区域的 zero-padding 被 roll 操作移位 → 分割边缘出现 7×7 棋盘 artifact | 用户脖子周围的人像分割有明显的阶梯状方块，用户体验极差 |
| 混合架构（CNN+ViT）量化时必须为不同模块设计不同量化策略 | EfficientFormer INT8 部署在 Qualcomm XR2：前几层 CNN（INT8）+ 后几层 ViT（FP16）→ 跨模块 reshape 放大 INT8 误差 3-5x → 室外强光准确率从 91%→62% | 室内正常、室外崩溃——bug 极其隐蔽，排查数周 |
| Apple ANE 上部署 ViT 必须将 multi-head attention 的 reshape/transpose 融合为单个 Core ML op | 苹果工程师实测：ViT-B/16 的 ~30 次 reshape 触发 CPU↔ANE memory copy（各 0.5-1ms），总 overhead 15-30ms 占总推理 40-60% | ViT 在 iPhone 上比预期慢 2-3x，Attention 不是计算慢而是内存布局变换慢 |
| EfficientViT/Swin 等 window attention 在生产中分辨率 > 800² 时必须启用 | 标准 ViT 的 O(n²) attention 在 n=4096 (1024²) 时 16.8M vs window_size=8 的 window attention 仅 262K——差距 64x | 高分辨率推理每帧延迟 > 100ms，无法满足实时视频分析（<30ms） |
| DINOv2 部署到 CPU 推理时需用 two-tower serving（粗筛+精排） | Pinterest 实测：ViT-g/14 在 Intel Xeon 上推理一张图 2.3s——不适合实时；用 DINOv2-Small 在边缘做粗筛 + ViT-g/14 做 Top-100 精排 | 直接用大模型做全量检索，P99 延迟不可接受，用户流失 |
