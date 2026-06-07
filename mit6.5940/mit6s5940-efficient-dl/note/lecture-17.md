# Lecture 17: Efficient GAN / Video / Point Cloud — 多模态的高效深度学习

## 1. 本讲核心问题

在 GAN、视频理解和三维点云这三个截然不同的领域，效率优化的"套路"是什么？具体而言：(1) GAN 的推理计算量巨大——如何压缩（GAN Compression）？如何在一个训练好的 GAN 上实现任意分辨率/算力权衡（AnyCost GAN）？不同的数据增强如何稳定 GAN 训练并同时降低计算需求（DiffAugment）？(2) 视频模型要处理的帧数远多于图像——如何利用时空冗余设计高效的时序模块（Temporal Shift Module, TSM）？(3) 点云是稀疏且不规则的——PVCNN/SPVCNN 如何混合体素和点操作来平衡效率与精度？(4) 自动驾驶中的多模态融合——BEVFusion 如何高效地对齐相机和 LiDAR 数据？根本之道：各领域的效率优化都源于**领域特有的冗余模式（domain-specific redundancy）**的发现与利用。

## 2. 通俗解释

**GAN Compression**：把一个"大师级画家"（大型 teacher GAN）的知识迁移到一个"学徒"（小型 student GAN）。但 GAN 的压缩比分类网络更难——因为 GAN 的输出是整张图片，不是单个标签，不能简单地"让输出一致"。解决方法是"中间层对齐"——让学徒不仅学最终画作（pixel loss），还学中间每一层的"创作思路"（intermediate feature matching）。同时用遗传算法自动搜索最优的通道剪枝策略——这些操作都是为了让小 GAN 的画质尽量接近大 GAN。

**AnyCost GAN**：传统思路是为每个目标算力单独训练一个模型（如 100M MACs 版本、50M 版本、20M 版本）。AnyCost GAN 的颠覆性想法——**一个模型，推理时动态调节分辨率**。低延迟时用低分辨率快速生成，高画质时切回高分辨率。这通过在生成器内引入多分辨率的子网络实现，推理时选择一个"子集"运行即可。

**DiffAugment（可微分数据增强）**：GAN 训练面临模式崩塌（mode collapse）——判别器太强或数据太少时，生成器只学会生成少数"安全"样本。数据增强可以缓解，但传统增强用在 GAN 上有副作用（把增强后的假图喂给判别器，判别器学会的是增强带来的痕迹而非真假差异）。DiffAugment 的妙招——**对真图和假图同时做相同增强**，判别器分不清"你变了吗"，只能关注增强前的语义内容。

**TSM（Temporal Shift Module）**：视频是 3D 数据（高×宽×时间），3D 卷积运算量极大。TSM 做了一个极大胆的简化——把"时间维度交互"消减为沿着通道维度"挪一挪"（shift）。比如把第 $t$ 帧的某些通道"借"给第 $t+1$ 帧用，这样每一帧就能看到前后帧的信息。零额外参数，零额外 FLOPs，却让 2D CNN 获得了时序建模能力——这是对"视频相邻帧高度冗余"这一领域特性的巧妙利用。

**PVCNN / SPVCNN**：点云需要同时做两种操作：(1) 局部 3D 卷积（精确但计算量大），(2) 基于体素的稀疏卷积（高效但丢失细节）。PVCNN 将两者融合——高分辨率分支用 MLP（逐点操作，保留细节），低分辨率分支用体素卷积（3D 全局上下文）。SPVCNN 更进一步，在这两个分支之间加入了双向特征交换。

**BEVFusion**：自动驾驶需要融合相机（2D 密集，rich semantics）和 LiDAR（3D 稀疏，accurate geometry）。BEVFusion 的做法是将两种模态都投影到统一的鸟瞰视图（BEV, Bird's Eye View）空间——相机深度估计后投影到 BEV，LiDAR 直接落在 BEV——在 BEV 空间做统一融合。这避免了不同坐标系的痛苦对齐。

## 3. 关键公式 (LaTeX)

### GAN Compression: 联合蒸馏损失

$$
\begin{aligned}
\mathcal{L}_{\text{total}} &= \mathcal{L}_{\text{recon}} + \lambda_{\text{distill}} \cdot \mathcal{L}_{\text{distill}} \\[4pt]
\mathcal{L}_{\text{recon}} &= \mathbb{E}_z\left[ \frac{1}{N} \| G_T(z) - G_S(z) \|_1 \right] \quad \text{(pixel-level)} \\[4pt]
\mathcal{L}_{\text{distill}} &= \sum_{l \in \mathcal{L}_{\text{intermediate}}} \mathbb{E}_z\left[ \| f^l_T(z) - f^l_S(z) \|_2^2 \right] \quad \text{(perceptual)}
\end{aligned}
$$

其中 $G_T$ 为 teacher generator，$G_S$ 为 student generator，$f^l$ 为第 $l$ 层中间特征。

### AnyCost GAN: 多分辨率子网络

AnyCost GAN 的核心——生成器包含 $K$ 个分辨率级别。给定输入 latent $z$，输出由所有分辨率输出加权平均：

$$
G_{\text{anycost}}(z, \alpha) = \sum_{k=1}^{K} \alpha_k \cdot G_k(z), \quad \sum_{k=1}^{K} \alpha_k = 1
$$

其中 $\alpha$ 是推理时可调的分辨率混合权重。当 $\alpha_k = \delta_{k, K}$ 时得到最高分辨率输出，$\alpha_k = \delta_{k, 1}$ 为最低。

### DiffAugment: 可微分增强

对真实图像 $x_{\text{real}}$ 和生成图像 $G(z)$ 施加相同增强 $T$：

$$
\begin{aligned}
\mathcal{L}_D &= \mathbb{E}_{x \sim p_{\text{data}}} [\log D(T(x))] + \mathbb{E}_{z \sim p_z} [\log (1 - D(T(G(z))))] \\[4pt]
\mathcal{L}_G &= \mathbb{E}_{z \sim p_z} [\log D(T(G(z)))]
\end{aligned}
$$

增强 $T$ 组合了：Color（亮度/对比度/饱和度）、Translation（随机平移）、Cutout（随机遮挡）。

### TSM: Temporal Shift

对于输入 $X \in \mathbb{R}^{N \times C \times T \times H \times W}$（$N$=batch, $C$=channels, $T$=frames），shift 操作：

$$
X'_{:, :c/8, t, :, :} = X_{:, :c/8, t-1, :, :} \quad \text{(forward shifted)} \\
X'_{:, 7c/8:, t, :, :} = X_{:, 7c/8:, t+1, :, :} \quad \text{(backward shifted)} \\
X'_{:, c/8:7c/8, t, :, :} = X_{:, c/8:7c/8, t, :, :} \quad \text{(current unchanged)}
$$

仅 1/8 通道做前移，1/8 通道做后移，剩余通道保持原位。

### SPVCNN: Sparse Point-Voxel Convolution

SPVCNN 融合两个分支：

$$
\mathbf{y}_{\text{point}} = \text{MLP}_{\text{point}}(\mathbf{x}_{\text{point}}), \quad \mathbf{y}_{\text{voxel}} = \text{SparseConv3D}(\text{devoxelize}(\mathbf{x}_{\text{voxel}})), \quad \mathbf{y} = \mathbf{y}_{\text{point}} + \mathbf{y}_{\text{voxel}}
$$

其中 voxelize/devoxelize 在点云和体素网格之间双向映射。

### BEVFusion: 多模态 BEV 投影

相机投影（LSS / Lift-Splat-Shoot）：

$$
\mathbf{F}_{\text{camera}}^{\text{BEV}} = \sum_{d} \mathbf{F}_{\text{camera}}(u, v) \odot \mathbf{D}(u, v, d)
$$

其中 $\mathbf{D}$ 是估计的深度分布，沿深度 $d$ 维度求和投影到 BEV。

LiDAR 点直接量化到 BEV 网格：

$$
\mathbf{F}_{\text{lidar}}^{\text{BEV}}(x, y) = \text{Pool}\{ \mathbf{f}_p : (x_p, y_p) \in \text{cell}(x, y) \}
$$

最终 BEV 融合：

$$
\mathbf{F}^{\text{BEV}} = \text{Concat}(\mathbf{F}_{\text{camera}}^{\text{BEV}}, \mathbf{F}_{\text{lidar}}^{\text{BEV}})
$$

## 4. 公式背后的直觉

**GAN Compression 的蒸馏直觉**：分类网络的 teacher-student 只需让 logits 一致（单点知识传递）。但 GAN 需要从 latent code $z$ 生成一整张图——如果只监督最终像素值，student 极容易"糊"（细节丢失）。中间层蒸馏的作用是强制 student 以与 teacher 相似的方式逐步构建图像——从结构（早期层）到纹理（中期层）到细节（后期层）。这保证了压缩后的生成过程"思路正确"。

**AnyCost GAN 的多分辨率直觉**：图像的生成过程本质上是"从粗到细"——早期层画大体形状，后期层加细节。AnyCost GAN 在多级分辨率上都输出完整图像，推理时只需提前退出（early exit）。这和网络的深度-宽度-分辨率复合缩放（EfficientNet）异曲同工——不是"重新设计网络"，而是"暴露网络的内在多尺度能力"。

**DiffAugment 的双向同步增强直觉**：标准数据增强用于 GAN 的问题在于"泄漏"——判别器聪明到会区分"这图片有没有被增强过"而非"这是真图还是假图"。DiffAugment 的做法是——既然两边都可能被泄露，就把两边都做相同增强。因为增强是可微的，梯度能穿过增强操作流向生成器。关键是**同步性**——真图和假图用相同的增强参数（相同的随机种子），确保判别器无法利用增强痕迹作弊。

**TSM 的"零成本"时序直觉**：2D 卷积的参数沿通道维度排列——把某些通道 shift 一帧，只是数据的重新排列，不需要额外参数。关键洞察是**相邻帧高度冗余**——第 $t$ 帧的 7/8 通道和第 $t-1$ 帧的 1/8 通道合并后，信息量几乎一样，但获得了跨时间的上下文。TSM 的成功依赖于视频帧间的高度相关性——对快动作或不相关帧序列，效果会下降。

**SPVCNN 的双分支直觉**：点操作（Point-based）精度高因为每个点都被独立处理，但缺乏空间上下文；体素操作（Voxel-based）有网格结构带来的 3D 上下文，但量化精度损失（多个点可能落入同一体素，单个点无法精确定位）。PVCNN 的分而治之——高分辨率分支用逐点 MLP 保留精确几何，低分辨率分支用体素稀疏卷积获取邻域信息。SPVCNN 加上双向连接，让两个分支的"发现"互相通知。

**BEVFusion 的 BEV 投影直觉**：相机和 LiDAR 数据天然在不同的空间——相机在透视投影的图像平面，LiDAR 在 3D 世界坐标系。BEV（鸟瞰视图）是自动驾驶规划的"母语"——车道、障碍物、轨迹都自然地在 BEV 中表达。BEVFusion 的想法是把两种模态从各自的"方言"翻译成"母语"后再交流，而不是试图直接跨模态配对（如 camera pixel ↔ lidar point 的配对极稀疏且不可靠）。

## 5. 工业界用途

| 技术 | 工业应用 |
|------|---------|
| **GAN Compression** | 手机端实时风格迁移（Prisma）、拍照特效、端侧超分辨率 |
| **AnyCost GAN** | 短视频应用的自适应滤镜（网络好时高清，差时低清保流畅） |
| **DiffAugment** | 小样本 GAN 训练（医学图像生成、稀有场景模拟） |
| **TSM** | 手机端视频动作识别（健身 App 动作计数）；安防监控实时行为分析 |
| **PVCNN / SPVCNN** | 自动驾驶 LiDAR 感知；AR/VR 3D 场景理解；机器人导航 |
| **BEVFusion** | L4 自动驾驶的多传感器融合（Waymo, Cruise, 特斯拉类似的 Occupancy Network） |
| **高效视频架构** | 视频会议的背景替换/美颜（需 <10ms 延迟）；无人机实时避障 |
| **点云稀疏卷积** | 室内 3D 扫描（苹果 LiDAR Scanner）；建筑 BIM 建模 |

## 6. PyTorch 实现思路

### TSM: Temporal Shift Module

```python
import torch
import torch.nn as nn

class TemporalShift(nn.Module):
    """Temporal Shift Module: zero-parameter temporal modeling.
    Shifts part of the channels along the time dimension.
    """
    def __init__(self, n_segment=3, shift_div=8):
        super().__init__()
        self.n_segment = n_segment  # number of frames
        self.shift_div = shift_div  # portion of channels to shift

    def forward(self, x):
        # x: (N * T, C, H, W) — batch & time merged
        N_T, C, H, W = x.shape
        T = self.n_segment
        N = N_T // T
        fold = self.shift_div
        x = x.view(N, T, C, H, W)

        # Channels to shift forward (from t-1)
        out = x.clone()
        out[:, 1:, :C // fold, :, :] = x[:, :-1, :C // fold, :, :]  # shift forward
        # Channels to shift backward (from t+1)
        out[:, :-1, C // fold * (fold - 1):, :, :] = x[:, 1:, C // fold * (fold - 1):, :, :]
        # Middle channels stay unchanged

        return out.view(N_T, C, H, W)


class TSMResBlock(nn.Module):
    """Residual block with temporal shift (TSM-style)."""
    def __init__(self, in_ch, out_ch, n_segment=8):
        super().__init__()
        self.shift = TemporalShift(n_segment=n_segment)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.shift(x)
        out = self.relu(self.conv1(out))
        out = self.conv2(out)
        return out + x if x.shape[1] == out.shape[1] else out
```

### PVCNN: Point-Voxel 混合架构

```python
import torch.nn as nn
import torch.nn.functional as F

def voxelize(points, features, voxel_size):
    """Convert point cloud to voxel grid.
    Returns voxel features and reverse mapping.
    """
    coords = (points / voxel_size).long()
    # Hash coords to voxel index, then scatter mean
    unique_coords, inverse = torch.unique(coords, dim=0, return_inverse=True)
    # (simplified — full implementation needs batch handling)
    return unique_coords, inverse


class PVCNNBlock(nn.Module):
    """PVCNN block: high-res point branch + low-res voxel branch."""
    def __init__(self, in_ch, out_ch, voxel_size=0.05):
        super().__init__()
        # Point branch (MLP, preserves details)
        self.point_mlp = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )
        # Voxel branch (sparse 3D conv, captures context)
        self.voxel_conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, points, features):
        # Point branch
        feat_point = self.point_mlp(features)  # (N, out_ch)

        # Voxel branch: voxelize → 3D conv → devoxelize
        coords, inverse = voxelize(points, features, voxel_size=0.05)
        # scatter features into voxels, do 3D conv, scatter back
        feat_voxel = torch.zeros(len(coords), features.shape[1])
        feat_voxel = feat_voxel.scatter_reduce(0, inverse.unsqueeze(-1).expand(-1, features.shape[1]), features, reduce='mean')
        # ... 3D conv on voxel grid ...
        feat_voxel_back = feat_voxel[inverse]  # scatter back to points

        return feat_point + feat_voxel_back
```

### DiffAugment: 可微分数据增强

```python
import torch
import torch.nn.functional as F

def diff_augment(x, policy='color,translation,cutout'):
    """Differentiable data augmentation for GAN training.
    Apply the SAME augmentation to both real and fake images.
    """
    if 'color' in policy:
        # Random brightness, contrast, saturation
        brightness = torch.rand(x.size(0), 1, 1, 1, device=x.device) * 0.6 + 0.7
        contrast = torch.rand(x.size(0), 1, 1, 1, device=x.device) * 0.6 + 0.7
        saturation = torch.rand(x.size(0), 1, 1, 1, device=x.device) * 0.6 + 0.7
        x = x * brightness
        x_mean = x.mean(dim=[2, 3], keepdim=True)
        x = (x - x_mean) * contrast + x_mean
        # Simplified saturation transform

    if 'translation' in policy:
        # Random translation by up to 1/8 of image size
        H, W = x.shape[2], x.shape[3]
        dx = torch.randint(-H//8, H//8, (1,), device=x.device).item()
        dy = torch.randint(-W//8, W//8, (1,), device=x.device).item()
        x_pad = F.pad(x, (abs(dy), abs(dy), abs(dx), abs(dx)), mode='reflect')
        x = x_pad[:, :, abs(dx)+dx:abs(dx)+dx+H, abs(dy)+dy:abs(dy)+dy+W]

    if 'cutout' in policy:
        # Random occlusion
        mask = torch.ones_like(x)
        for b in range(x.size(0)):
            h = torch.randint(H//4, H//2, (1,)).item()
            w = torch.randint(W//4, W//2, (1,)).item()
            y = torch.randint(0, H-h, (1,)).item()
            x_ = torch.randint(0, W-w, (1,)).item()
            mask[b, :, y:y+h, x_:x_+w] = 0.0
        x = x * mask

    return x
```

### GAN Compression 核心思想

```python
def gan_compression_loss(G_student, G_teacher, z, lambda_distill=5.0):
    """Combined reconstruction + distillation loss for GAN compression."""
    # Generate from same latent code
    with torch.no_grad():
        fake_teacher = G_teacher(z)

    fake_student = G_student(z)

    # Pixel-level reconstruction
    loss_recon = F.l1_loss(fake_student, fake_teacher)

    # Intermediate feature distillation (collect features from both)
    loss_distill = 0.0
    # In practice: register hooks on both G_teacher and G_student
    # to collect intermediate features, then compute L2 distance
    # for layer_t, layer_s in zip(teacher_features, student_features):
    #     loss_distill += F.mse_loss(layer_s, layer_t)

    return loss_recon + lambda_distill * loss_distill
```

## 7. TinyML / Edge AI 部署意义

**视频分析在边缘**：
- TSM 的零参数时序建模意味着 2D 模型体重不加任何额外参数量就能处理视频——极度适合边缘设备
- 手机视频分析（手势识别、健身计数）在 CPU 上即可运行 TSM-ResNet18（<100M FLOPs）
- 部署时只需在线程维度上做 shift 操作，可利用 NEON SIMD 加速

**GAN 的边缘部署**：
- GAN Compression 可将 StyleGAN2 的 FLOPs 降低 5-10x 而几乎无损——使移动端实时照片编辑成为可能
- AnyCost GAN 让同一模型在不同网络条件下自适应画质——短视频 App 的核心需求
- 端侧 GAN 推理的主要瓶颈在生成器的上采样操作（转置卷积 / PixelShuffle）——在高分辨率输出时 dominate

**点云的边缘挑战**：
- 点云处理的稀疏性使批量处理变困难——每个场景点数不同，padding 浪费严重
- Apple LiDAR Scanner + 神经网络的端侧 3D 重建：SPVCNN 的体素分支可用 MPS (Metal Performance Shaders) 加速
- 自动驾驶边缘计算（车端）：BEVFusion 的推理延迟需 <50ms，通常依赖 NVIDIA Orin 等专用芯片

**跨领域的共同智慧**：
- 所有高效设计都根源自对**领域冗余**的深刻理解
- 视频的"帧间冗余"→ TSM
- 图像/视频的"空间分辨率冗余"→ AnyCost GAN 的自适应推理
- 点云的"空间稀疏性"→ SPVCNN 的混合表示
- 多传感器的"表示不匹配"→ BEVFusion 的统一投影

## 8. 常见误区

> ❌ **误区 1："GAN 的蒸馏和分类网络蒸馏是一样的"**
> GAN 不仅需要输出级匹配（pixel loss），还需要中间层特征对齐。因为 GAN 的生成过程是"渐进式"的——早期层画结构，中期层填纹理，后期层补细节——仅监督最终输出会导致 student 在生成路径上走偏。

> ❌ **误区 2："TSM 只是把 channels 顺序 shift 一下，不会有实质效果"**
> TSM 的本质是让 2D 卷积核在时间维度上有了感受野。如果一个 3×3 卷积核在 TSM 后处理图像，它实际上同时看到了 $t-1$、$t$、$t+1$ 三帧的信息——等价于一个 3×3×3 3D 卷积核的部分操作，但零额外参数。

> ❌ **误区 3："点云就是 3D 图像，直接上 3D 卷积就好"**
> 点云极度稀疏（大多数空间是空的），直接 3D 卷积会有大量无用计算（绝大多数体素为空）。稀疏卷积（MinkowskiEngine）虽解决部分问题，但点数百万+体素分辨率高时仍成本高。PVCNN 的混合方案通过将全局上下文放在低分辨率体素来做、将精确几何放在逐点操作来做，利用了信息和成本的"边际递减"规律。

> ❌ **误区 4："BEVFusion 需要精确的深度估计才能将相机图像投影到 BEV"**
> 不需要精确深度。LSS（Lift-Splat-Shoot）方法为每个像素估计一个深度**分布**（而非单值），沿所有可能深度投影后在 BEV 上做 soft accumulation。这相当于用概率分布代替确定值——鲁棒地处理深度不确定性。

> ❌ **误区 5："AnyCost GAN 就是在训练时随机使用不同分辨率"**
> AnyCost GAN 的关键是训练时使用分辨率相关的"子网络"采样策略和渐进式训练。随机采样分辨率可能让训练不稳定——通常采用课程学习（先全分辨率，再逐步引入低分辨率路径）。

## 9. 面试问题

**Q1: GAN Compression 的核心损失函数包含哪些部分，各自解决了什么问题？**
A: (1) **Pixel-level L1 重建损失**：确保 student 的输出和 teacher 在像素上接近；(2) **中间特征蒸馏损失**（L2 匹配 teacher 和 student 的逐层特征）：强制 student 以 teacher 的"生成路径"逐步构建图像；(3) **对抗损失**（标准 GAN loss）：保证压缩后仍能骗过判别器。

**Q2: TSM 如何以零额外参数实现时序建模？其局限性是什么？**
A: TSM 通过在通道维度上做 partial shift——1/8 通道前移一帧、1/8 后移一帧，让 2D 卷积天然获得时序感受野。零参数是因为 shift 只是数据重排。局限性：(1) 仅能建模相邻帧，无法捕获长期时序关系；(2) 对高速运动场景（帧间变化大）效果下降；(3) shift 操作在 ONNX/TensorRT 中不是原生 ops，需要特殊处理。

**Q3: SPVCNN 为什么要同时维护 point-based 和 voxel-based 两个分支？**
A: Point-based 分支精确保留每个点的几何位置（无量化损失），但缺乏邻域上下文；Voxel-based 分支通过规整网格提供了空间邻域信息，但有点→体素的量化误差。两个分支互补——点分支负责精确几何，体素分支负责全局上下文。

**Q4: DiffAugment 的"同步增强"如何防止判别器作弊？**
A: 如果只增强真图（传统做法），判别器可以通过增强痕迹（如 cutout 的黑块边缘、平移后的边界）来区分真假图。同时对真假图做相同增强（相同随机种子），增强痕迹在两边都存在——判别器无法利用它作为判别依据，只能依靠增强前的语义内容。

**Q5: BEVFusion 中相机数据投影到 BEV 的核心步骤？**
A: (1) 为每个像素预测一个深度**分布**（不单值）；(2) 根据深度分布，将像素特征沿相机射线方向"撒"到 3D 空间； (3) 在 BEV 平面上做 voxel pooling/accumulation，得到 BEV 特征图。关键是使用分布而非单点深度，为深度不确定性提供了天然的 soft 处理。

**Q6: AnyCost GAN 如何实现"一个模型，多种推理成本"？**
A: 生成器内嵌多级分辨率的输出头。训练时随机采样分辨率级别（或课程式训练），让所有子网络都能独立生成合理图像。推理时根据延迟预算选择适当的输出头——就像在生成中途提前"退出"。

## 10. 本讲总结

本讲横跨 GAN、视频和点云三个截然不同的领域，回答了"效率优化的统一方法论"：

**核心方法论——领域冗余（Domain-Specific Redundancy）的发现与利用**：

| 领域 | 冗余模式 | 效率优化技术 |
|------|---------|-------------|
| GAN | 生成器层间信息传递高度相关 | 中间层蒸馏 + 通道剪枝（GAN Compression） |
| GAN | 不同场景对分辨率敏感度不同 | 多分辨率子网络提前退出（AnyCost GAN） |
| GAN | 真假图在增强空间有共享特性 | 同步差分增强（DiffAugment） |
| 视频 | 相邻帧内容高度冗余 | 通道 shift 零参数时序建模（TSM） |
| 点云 | 几何精确性和空间上下文信息在不同分辨率上有不同重要性 | 点-体素双分支混合（PVCNN/SPVCNN） |
| 自动驾驶 | 多模态数据存在统一的 BEV 空间 | 坐标系统一投影+融合（BEVFusion） |

**三条跨领域规律**：
1. **近似以换取效率**：不是更精确地建模，而是知道"哪些精度可以牺牲"（TSM 牺牲长期时序精度换取零参数）
2. **分而治之**：把问题拆成"精细部分+粗糙部分"，分别用一个高效分支处理（PVCNN 的点+体素；AnyCost 的多分辨率）
3. **表示统一**：不同模态/不同分辨率的本质冲突可以通过"转换到统一空间"解决（BEVFusion 的 BEV 投影）

本课程从 Lecture 1 的基础概念出发，经过计算图优化、剪枝、量化、神经架构搜索、知识蒸馏，到 Transformer 的效率优化、LLM 部署对齐、长上下文和多模态——最终落脚于这个统一的方法论：**"效率"不是"偷工减料"，而是"把计算花在刀刃上"**。理解并发现每个领域的特有冗余，才是高效深度学习工程师的核心能力。
