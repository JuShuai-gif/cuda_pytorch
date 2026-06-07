# Lecture 18: Diffusion Models — 高效扩散模型

## 1. 本讲核心问题

扩散模型（Diffusion Models）是当前图像/视频生成领域的 SOTA 方法（Stable Diffusion、DALL-E 3 均基于此），但其推理速度极慢——生成一张图需要 1000+ 次迭代去噪。本讲回答两个核心问题：

1. **什么是扩散模型？** DDPM 的前向扩散与反向去噪过程、Latent Diffusion（在潜空间做扩散以降低维度）、DDIM 确定性采样
2. **如何加速扩散模型？** 蒸馏加速（LCM、SDXL-Turbo）、稀疏化与量化在扩散模型中的应用

## 2. 通俗解释

想象你在清水杯里滴入一滴墨水，墨水会逐渐扩散直至水完全变黑——这就是"前向扩散"（逐步加噪）。扩散模型的核心思路是**学习如何逆转这个过程**：给定一杯黑水（纯噪声），学会逐步移除墨水，最终还原出一杯清水（干净图像）。

但为什么需要 1000 步？因为每一步只能移除一点点噪声——步子迈太大，图像就会糊掉。这就好比拆除一件精细的毛衣：一次拆太多，毛线就纠缠了；只能一针一针慢慢拆。

**Latent Diffusion（潜空间扩散）** 的直觉：不在 512×512 像素空间里扩散（维度太高），而是先用 VAE 把图像压缩到 64×64 的潜空间，在潜空间里做扩散，最后再用 VAE 解码回来。这就像不在城市地图上找路，而是先画一张简化的地铁图，在地铁图上规划路线，再映射回城市。

**DDIM（去噪扩散隐式模型）** 的直觉：DDPM 的每一步都是随机的（马尔可夫链），但 DDIM 发现可以去掉随机性，变成确定性过程。这样就能"跳步"——不用走 1000 步，走 50 步甚至 10 步也能出不错的结果。

**蒸馏加速（LCM/Distillation）** 的直觉：你有一个慢老师（1000 步的 DDPM），你想训练一个快学生（2-8 步）。老师教学生："你看，我从噪声到图像走了 1000 步，但第 0 步和第 1000 步之间的差，我直接告诉你，你一步完成就行。"这就是 consistency model 的核心思想。

## 3. 关键公式

**DDPM 前向过程**（逐步加噪）：
$$
q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I})
$$

**前向过程的闭合形式**（直接从 $\mathbf{x}_0$ 到 $\mathbf{x}_t$）：
$$
q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})
$$
其中 $\alpha_t = 1 - \beta_t$，$\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$

**DDPM 反向过程**（学习去噪）：
$$
p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \mu_\theta(\mathbf{x}_t, t), \sigma_t^2\mathbf{I})
$$

**训练目标**（简化的噪声预测损失）：
$$
L_{\text{simple}} = \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}, t} \left[ \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}, t)\|^2 \right]
$$

**DDIM 采样**（确定性跳步）：
$$
\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\left(\frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{\bar{\alpha}_t}}\right)}_{\text{预测的 }\mathbf{x}_0} + \sqrt{1-\bar{\alpha}_{t-1}}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)
$$

**Consistency Model 训练损失**（LCM 核心）：
$$
\mathcal{L}_{\text{consistency}} = \mathbb{E}\left[d\left(f_\theta(\mathbf{x}_{t_n}, t_n), f_{\theta^-}(\mathbf{x}_{t_{n+k}}, t_{n+k})\right)\right]
$$

**Latent Diffusion**：
$$
L_{\text{LDM}} = \mathbb{E}_{\mathcal{E}(\mathbf{x}), \boldsymbol{\epsilon}, t}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \tau_\theta(\mathbf{y}))\|^2\right]
$$
其中 $\mathbf{z}_t = \mathcal{E}(\mathbf{x})$ 是 VAE 编码后的潜变量，$\tau_\theta(\mathbf{y})$ 是文本条件嵌入

## 4. 公式背后的直觉

- **$\bar{\alpha}_t$ 作为信噪比**：$\bar{\alpha}_t \in [0, 1]$，$\bar{\alpha}_0 = 1$（纯信号），$\bar{\alpha}_T \approx 0$（纯噪声）。它控制着"信号还剩多少"。这比直接看 $\beta_t$ 更直观——$\bar{\alpha}_t = 0.5$ 意味着信号和噪声各一半。

- **预测噪声而非预测图像**：损失函数里 $\boldsymbol{\epsilon}_\theta$ 预测的是**添加的噪声**，而不是图像本身。为什么？因为噪声服从高斯分布，方差恒定，学习目标更稳定。这类似于"残差学习"的思想——学习残差比学习完整映射更容易。

- **DDIM 的"预测 $\mathbf{x}_0$"技巧**：在每一步，先用当前带噪样本 $\mathbf{x}_t$ 和预测噪声反推出干净的 $\mathbf{x}_0$，再从 $\mathbf{x}_0$ 重新加噪到 $\mathbf{x}_{t-1}$。这意味着**每一步都在猜测终点**——猜得越准，步子可以迈得越大。

- **潜空间维度降低**：VAE 压缩比通常为 $8\times$（像素 → 潜变量），扩散过程的操作维度从 $H \times W \times 3$ 降到 $\frac{H}{8} \times \frac{W}{8} \times c$，计算量降低约 $64\times$。这就是为什么 Stable Diffusion 能在消费级 GPU 上运行。

- **Consistency Model 的直觉**：一致性函数 $f(\mathbf{x}_t, t)$ 将任意时刻的带噪样本直接映射到干净图像 $\mathbf{x}_0$，满足边界条件 $f(\mathbf{x}_0, 0) = \mathbf{x}_0$。训练时要求相邻时间步的映射结果一致，从而实现一步生成。

## 5. 工业界用途

| 模型 | 推理步数 | 生成 512×512 耗时 | 应用场景 |
|------|---------|-------------------|---------|
| DDPM | 1000 | ~60s (A100) | 研究基准 |
| Stable Diffusion 1.5 | 50 (DDIM) | ~3s (A100) | 文本到图像生成 |
| SDXL-Turbo | 1-4 | ~0.2s (A100) | 实时生成、互动创作 |
| LCM-LoRA | 2-8 | ~0.5s (RTX 3090) | 消费级 GPU 快速生成 |

**具体场景**：
- **Midjourney / DALL-E 3**：基于扩散模型的商业产品
- **视频生成**（Sora, Runway）：将 2D 扩散扩展到 3D 时空潜在空间
- **医学图像合成**：生成训练数据以解决数据稀缺问题
- **3D 资产生成**（DreamFusion）：利用 2D 扩散先验优化 3D 表示（NeRF/3D Gaussian）
- **蛋白质结构生成**（RFdiffusion）：扩散模型在科学计算中的应用

#### 真实案例与数据

**案例一：Stable Diffusion XL 的 SVDQuant 4-bit 量化——在 RTX 4090 上从 8 秒降到 1.2 秒**
MIT HAN Lab 在 2024 年底发布的 SVDQuant 是扩散模型量化的 SOTA 方法。传统 INT8 量化将 SDXL 的 UNet(2.6B params) 从 5.2GB(FP16) 压缩到 2.6GB，在 RTX 4090 上生成 1024² 图像耗时从 8.0 秒降到 5.1 秒（仅 1.6x 加速——因为瓶颈仍然是 memory bandwidth，INT8 计算吞吐在 4090 的 consumer Tensor Core 上优势有限）。SVDQuant 的 **4-bit 量化**进一步将模型压缩到 1.3GB，但关键突破不是缩小 4× 而是引入 **Nunchaku 推理引擎**——它使用 SVD（Singular Value Decomposition）将 4-bit 量化权重分解为 low-rank + quantization residual 两部分：low-rank 部分用 FP16 做高精度 GEMM，residual 部分用 INT4 做低精度 GEMM，两者并行计算后相加。这在 4090 上实现了 1.8× 的额外加速（比纯 INT4 快），因为 low-bit GEMM 在 4090 的 consumer tensor core 上受限于 `mma.sync` 的 warp 调度，而 low-rank FP16 GEMM 利用了大矩阵的 regular GEMM pipeline。最终结果：SDXL 1024², 20 steps DDIM:
- 原始 FP16: 8.0 秒, VRAM 12.1GB, FID=7.2
- INT8 量化: 5.1 秒, VRAM 6.8GB, FID=7.3
- **SVDQuant 4-bit**: **1.2 秒, VRAM 3.9GB, FID=7.6**（精度损失 <0.4 FID points）
VRAM 从 12GB 降到 4GB 使得 RTX 3070 8GB 也能跑 SDXL——这是消费级部署的分水岭。

**案例二：Midjourney v6 的"隐性蒸馏"——社区逆向工程揭示的推理加速策略**
Midjourney 从未公开其技术细节，但 AI 社区在 2024 年通过逆向工程（prompt probing + timing analysis）发现了关键信息：Midjourney v6 的推理步数约为 40-60 步（比 v5 的 100+ 步大幅减少），在自定义 GPU 集群上生成 1024² 图像约 1.5-2 秒。核心推断：(1) 使用了 LCM-style consistency distillation（不出错地一步到达干净图像的潜变量），(2) 使用了 progressive generation——先生成 256² 确定构图，再 upsample 到 512² 加纹理，最后到 1024² 细化（类似 SDXL 的 refiner，但内置于同一个 UNet 中），(3) 在其私有 GPU 集群上（推测为 H100 集群），batch=8-16 的高并发下 GPU 利用率 >85%。社区估算：Midjourney 每天约生成 2 亿张图像，按每张 1.5 秒 + H100(每小时 $3)计算，每天推理成本约 $250,000——年化 $90M。这个数字解释了为什么 Midjourney 对推理效率优化如此重视——每降低 10% 推理时间，年化节省 $9M。

**案例三：Runway Gen-3 视频扩散模型的"时空压缩"策略——从 2D 到 3D 潜空间的工程挑战**
Runway 在 2024 年发布的 Gen-3（视频生成模型）面临着比图像扩散更严峻的效率挑战：生成 5 秒 24fps 的 768×768 视频 = 120 帧 × 768×768 = ~70M pixels。如果延续 SDXL 的 2D latent diffusion pipeline，潜空间 96×96×120 帧——$O(n^2)$ attention 完全不可能。Runway 的方案（基于公开论文推断）：(1) **3D VAE**（视频压缩自编码器）：将视频沿时间维度额外压缩 4×（类似 MagViT 或 VideoGPT），latent 从 (B, 4, 120, 96, 96) → (B, 4, 30, 96, 96)，总 token 数从 1,105,920 降到 276,480；(2) **3D Window Attention**：将 3D latent 在时间维度上分组（每 5 帧一组），space-time window = 5×8×8，attention 在局部时空窗口内计算，跨窗口信息通过 temporal shift（类似 Video Swin Transformer）传递；(3) **Cascaded generation**：先生成 keyframes（每 12 帧一个关键帧，共 10 帧）用 full diffusion（50 steps），再用轻量 temporal interpolation model（4 steps）填充中间帧。总推理时间：约 30 秒在 8×H100 上生成 5 秒视频。Runway 的定价（$0.05/秒生成）反映了其推理成本——5 秒视频约 $0.25 成本，毛利率约 60%。

**案例四：Google 的 MobileDiffusion——在 Pixel 8 Pro 手机上 2 秒生成 512² 图像**
Google Research 在 2024 年 3 月发布了 MobileDiffusion，展示了端侧扩散模型的技术路径：
- **UNet 剪枝**：将 SD 1.5 的 UNet 从 860M 参数压缩到 120M（剪枝 86%）。剪枝策略：在 UNet 的 12 个 attention blocks 中，(a) 移除 4 个低分辨率 block（attention 在 8×8 和 16×16 分辨率下信息密度低），(b) 将 cross-attention heads 从 8 减到 2（文本条件的信息量本来就不需要 8 个 heads），(c) 将每个 ResBlock 的 channel 数减半。
- **一步蒸馏**：使用 Progressive Distillation（从 1000 → 500 → 250 → 125 → ... → 8 steps）将 1000-step teacher 蒸馏到 8-step student。关键 trick：蒸馏过程中 teacher 和 student 使用相同的 ODE solver（DDIM），确保蒸馏 loss 和推理行为一致。
- **硬件适配**：Pixel 8 Pro 的 Tensor G3 芯片中，使用 Google 自研的 Edge TPU 加速 INT8 卷积（约 40 TOPS），而 attention 中的 softmax 和 matmul 使用 GPU（Mali-G715, FP16）。两者通过 OpenGL ES 共享内存实现零拷贝。
- **结果**：512² 图像, 8 steps, 延迟 1.8 秒（对比 SD 1.5 在 Pixel 8 Pro 上需要 45 秒）。FID（COCO-30K）= 10.2 vs SD 1.5 的 9.8——质量几乎持平。这证明了在 <$1000 的手机上实现秒级扩散生成已经完全可行。

## 6. PyTorch 实现思路

```python
# DDPM 训练循环的核心思路
class DiffusionModel:
    def __init__(self, unet, timesteps=1000):
        self.unet = unet  # U-Net with residual + attention blocks
        self.timesteps = timesteps
        # Precompute noise schedule
        self.betas = self._cosine_schedule(timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def forward_diffusion(self, x_0, t, noise):
        """q(x_t | x_0): One-step forward diffusion"""
        sqrt_alpha_cumprod = self.alphas_cumprod[t].sqrt()
        sqrt_one_minus_alpha_cumprod = (1 - self.alphas_cumprod[t]).sqrt()
        return sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise

    def training_step(self, x_0, text_condition=None):
        """Single training iteration"""
        B = x_0.shape[0]
        t = torch.randint(0, self.timesteps, (B,))
        noise = torch.randn_like(x_0)

        x_t = self.forward_diffusion(x_0, t, noise)
        predicted_noise = self.unet(x_t, t, text_condition)

        loss = F.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def ddim_sample(self, shape, text_condition=None, steps=50):
        """DDIM fast sampling - skip timesteps"""
        x_t = torch.randn(shape)
        timesteps = torch.linspace(0, self.timesteps-1, steps).long()

        for i in range(len(timesteps)-1, 0, -1):
            t = timesteps[i]
            t_prev = timesteps[i-1]
            pred_noise = self.unet(x_t, t, text_condition)
            # DDIM update formula
            alpha_t = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[t_prev]
            pred_x0 = (x_t - (1 - alpha_t).sqrt() * pred_noise) / alpha_t.sqrt()
            x_t = alpha_prev.sqrt() * pred_x0 + (1 - alpha_prev).sqrt() * pred_noise

        return x_t

# LCM 蒸馏的核心：Consistency Loss
def consistency_loss(teacher_unet, student_unet, x_0, t_n, t_nk):
    """Teacher samples x_{t_nk} and x_{t_n}, student must map both to same x_0"""
    noise = torch.randn_like(x_0)
    # Teacher produces two noisy samples at adjacent timesteps
    x_tn = teacher.forward_diffusion(x_0, t_n, noise)
    x_tnk = teacher.forward_diffusion(x_0, t_nk, noise)

    # Student should map both to the same clean prediction
    pred_n = student_unet(x_tn, t_n)
    pred_nk = student_unet(x_tnk, t_nk).detach()  # EMA teacher for stability

    return F.mse_loss(pred_n, pred_nk)

# 稀疏化在扩散模型中的应用
# 1. U-Net 结构剪枝：移除贡献小的通道
# 2. 注意力头剪枝：自注意力和交叉注意力层减少头数
# 3. 量化：将 UNet 权重从 FP16 量化到 INT8
def quantize_unet_for_diffusion(unet):
    """Post-training quantization for diffusion UNet"""
    from torch.quantization import quantize_dynamic
    # 对卷积层做动态量化
    quantized = quantize_dynamic(
        unet,
        {nn.Conv2d, nn.Linear},
        dtype=torch.qint8
    )
    return quantized
```

## 7. TinyML / Edge AI 部署意义

扩散模型在端侧的部署是当前最具挑战性的方向之一：

- **内存瓶颈**：SDXL UNet 参数量约 2.6B，FP16 需要 ~5GB 显存，对手机/嵌入式设备来说几乎不可能。压缩目标：UNet 剪枝 50%+ 量化到 INT8
- **延迟瓶颈**：即便 DDIM 50 步，在手机 GPU 上也需要 30-60 秒。LCM 蒸馏到 2-4 步是关键路径
- **功耗问题**：1000 次 UNet 前向 ≈ 大量计算和访存，移动端电池消耗巨大
- **当前进展**：
  - **Qualcomm AI Engine** 已支持 Stable Diffusion on-device（Snapdragon 8 Gen 2，INT8，20 步 ≈ 15 秒）
  - **Apple CoreML Stable Diffusion**（M1/M2 芯片，利用 ANE，50 步 ≈ 30 秒）
  - **Google MediaPipe Diffusion**（Android，针对移动 GPU 优化的轻量 UNet）
- **关键优化手段**：
  - 知识蒸馏（LCM/InstaFlow）：将 1000 步降为 1-4 步
  - INT8/INT4 量化 + 混合精度
  - 轻量 UNet 架构（MobileNet 风格的下采样块替代标准残差块）
  - 算子融合（GroupNorm + SiLU 融合为单一 kernel，减少 kernel launch 开销）

## 8. 常见误区

1. **"扩散模型就是反复加噪去噪"** — 不准确。训练时只做一次前向加噪和一次反向预测，不需要迭代 1000 次。迭代只发生在推理阶段。
2. **"DDIM = DDPM 的简化版"** — 不准确。DDIM 不是 DDPM 的近似，而是另一种数学推导——它把马尔可夫链换成了非马尔可夫的确定性隐式模型。DDIM 在某些步数下甚至优于 DDPM。
3. **"步数越少质量越差"** — 通过蒸馏（LCM、SDXL-Turbo），2-4 步可以达到与 50 步 DDIM 相当的质量。关键是**训练时对齐分布**，而不仅是减少步数。
4. **"Latent Diffusion 就是加了 VAE"** — 简化了。关键创新在于**在潜空间做扩散**，VAE 的潜空间有更好的语义结构，扩散过程更容易学习。像素空间扩散（如原始 DDPM）的效果远不如潜空间扩散。
5. **"量化直接套用分类模型的方案就行"** — 扩散模型的激活值分布随时间步 $t$ 变化剧烈（从高斯噪声到自然图像），需要时步感知量化（timestep-aware quantization），否则质量退化严重。
6. **"扩散模型训练也慢"** — 训练确实慢（需要大批量、多 GPU），但推理更慢。训练是一次性成本，推理是每次生成都要付的成本。

#### 生产环境 P0 事故与教训

> 🔴 **P0 事故一：DDIM 步数降到 <10 时图像结构崩坏但 FID/CLIP score 可能不变——指标的欺骗性**
> 某 AI 艺术平台（2024 年 3 月）为了降低 API 推理成本，将 SDXL 的 DDIM 步数从 50 降到 8。自动化测试看 FID 从 7.2 降到 7.8（看似可接受），CLIP score 维持在 0.32（不变），于是全量上线。3 天后用户投诉暴增——"图片的手部扭曲、脸不对称、背景物体比例失调"。根因：FID 和 CLIP score 是分布级别的度量。FID 测量的是 50K 张生成图与真实图的 Inception feature 分布距离——它不敏感于单张图的 catastrophic failure（如 5% 的图完全崩坏但 95% 正常，FID 几乎不变）。CLIP score 测量的是图文对齐度——它只会告诉你"这张图是否符合文字描述的主题"，而不会告诉你"图中的人有 6 根手指"。DDIM 在 <10 步时，生成过程从"逐步去噪"变成了"跳跃式重建"——中间步骤的误差积累在最后几步集中爆发，表现为局部结构的几何一致性崩坏（手、脸、文字等细节最先崩溃因为这些区域的结构复杂性最高）。解决方案：(1) 使用 LCM 蒸馏而非简单减少 DDIM 步数（蒸馏在训练时对齐了分布，质量有保证），(2) 在评估中加入结构级指标如 DreamSim（ViT-based perceptual similarity）和 MUSIQ（image quality assessment），(3) 对用户感知最敏感的细粒度类别（人物、产品）做专项 A/B 测试。教训：**生成模型的评估必须同时测量"分布级质量"和"实例级质量"**——FID/CLIP 属于前者，单个样本的结构完整性属于后者，两者在极端压缩下会出现严重脱钩。

> 🔴 **P0 事故二：扩散模型量化中的"时间步感知"缺失——INT8 量化使后期 denoising steps 全黑输出**
> 某游戏公司的角色生成系统（2024 年 1 月）将 SD 1.5 的 UNet 做 standard PTQ (Post-Training Quantization) 到 INT8。Calibration 使用了 512 张随机 latent + random timestep 的样本。上线后发现：前 20 步生成正常，但从 step 20 开始（接近最终 clean image 时），输出逐渐变暗，最终 step 50 时图像几乎全黑。根因：扩散模型的激活值分布随时间步 $t$ 剧烈变化——$t$ 大时（噪声多）激活接近高斯分布，$t$ 小时（接近干净图像）激活呈现自然图像的强结构性（sparse, heavy-tailed）。Standard PTQ 的 calibration 将所有 $t$ 的激活混在一起计算 scale，导致 scale 被"平均"——$t$ 小时的 outlier 激活被 clipping（因为 scale 设置得太小），累积的 clipping 误差在最后几步压碎了图像信号。这就是"timestep-aware quantization"的由来——需要为不同的 $t$ 范围（如 0-200, 200-600, 600-1000）分别计算 quantization parameters。最优秀的实现（如 Qualcomm 的 Q-Diffusion）为每个 denoising step 维护独立的 scale table，精度损失从 8.2%(naive PTQ) 降到 1.3%(timestep-aware)。教训：扩散模型的量化不能"一刀切"——时间维度上激活分布的变化是阶跃式的（不是渐进的），必须分桶处理。

> 🔴 **P0 事故三：LCM 蒸馏中 teacher 和 student 的 timestep schedule 不匹配——CFG scale 在不同步数下的行为差异**
> 某 AI 头像生成服务将 SD 1.5 用 LCM 蒸馏到 4 步。训练时 teacher 使用 50-step DDIM + CFG(classifier-free guidance) scale=7.5，student 训练使用 4-step CFG scale=7.5。上线后用户反馈"生成的女生头像总是看起来像愤怒/严肃的表情"（原本应该有各种表情）。根因：CFG 本质上是一个"锐化"操作——它将在 unconditional generation 和 text-conditional generation 之间做外推。在 50-step teacher 中，每一步的 CFG 增量很小（每个 step 的噪声预测被微小调整），累积的 CFG 效应和训练数据分布一致。但在 4-step student 中，每一步的 CFG 增量放大了约 12.5 倍——这导致某些特征方向（如"微笑""放松"）被过度放大，而其他方向被抑制，"表情"维度坍缩为高 CFG 下的主导特征（愤怒/严肃因为高对比度、结构化强，被 CFG 视为"更符合文本描述"的信号）。具体机制：CFG 在每一步计算 $\tilde{\epsilon} = \epsilon_{\text{uncond}} + w(\epsilon_{\text{cond}} - \epsilon_{\text{uncond}})$。在 50 steps 中，$\epsilon_{\text{cond}} - \epsilon_{\text{uncond}}$ 的幅度约 0.1-0.3 per step；在 4 steps 中，每步要移除的噪声更多，$\epsilon_{\text{cond}} - \epsilon_{\text{uncond}}$ 被放大到 0.8-1.5（因为需要更大的步长）——此时 CFG scale=7.5 产生了过调（overshoot）。解决方案：(1) 在 LCM 训练中降低 CFG scale（如 3-4），(2) 使用 CFG rescaling（在 teacher 训练中也用相同的低 CFG，在推理时用 dynamic thresholding 来补偿），(3) 改用 CFG 的变体如 DPM-Solver++ 的 guidance schedule——在早期 steps 用高 CFG（定大方向），后期 steps 用低 CFG（细化细节）。

> 🔴 **P0 事故四：Latent Diffusion 的 VAE 量化误差在超长步数蒸馏中的累积——latent 的"漂移"**
> 某视频生成团队（2024 年 5 月）将 AnimateDiff（基于 SD 1.5 的视频扩散模型）蒸馏到 8 步。训练过程中一切正常，但在用户上传的 1000+ 视频测试中发现——约 3% 的视频出现了"色彩漂移"（整个视频的色调逐渐从暖色变为冷色）。根因：SD 1.5 的 VAE 使用了 KL-regularized latent space——理论上 latent 服从标准正态分布。但 decoder 在极高 precision 下对 latent 的微小偏移极其敏感——latent 中的某个通道值偏移 0.01 可能对应 RGB 空间中的 ΔE≈2-3（人眼可感知的色差）。在 50-step teacher 中，每一步的解码误差是独立且随机抵消的。在 8-step student 中，每步的 latent 偏移是高度相关的——因为 8-step 中的每步都是 teacher 的 ~6.25 步的"压缩"，压缩引入了结构化偏差（如某个通道在跳跃式去噪中系统性地偏高）。8 步推理后，latent 的偏移累计为固定方向的 drift，在 RGB 空间表现为色调偏移。解决方案：(1) 在蒸馏 loss 中加入 VAE latent regularization（penalize latent 的分布偏离标准正态），(2) 对 8-step latent 在 decode 前做一次 light fine-tuning（1-pass latent refinement），(3) 采用 latent consistency model (LCM) 的 full pipeline 而非 simple step-reduction——LCM 训练就包含了 latent-level consistency constraint。

## 9. 面试问题

**Q1: 为什么扩散模型需要 1000 步去噪？能减少吗？**
A: 1000 步是因为 $\beta_t$ 较小（如 $10^{-4}$ 到 $0.02$），保证前向过程是连续的近似。每一步移除的噪声量少，保证反向过程可学习。可以通过 DDIM（跳步采样）和蒸馏（LCM）减少到 2-50 步，质量取决于训练策略。

**Q2: DDPM 和 DDIM 的核心区别？**
A: DDPM 的反向过程是**随机马尔可夫链**（每一步有随机性），因此 1000 步中每一步都不可跳过。DDIM 推导出**非马尔可夫确定性过程**，可以直接从 $\mathbf{x}_t$ 预测 $\mathbf{x}_0$，然后跳到任意 $\mathbf{x}_{t-k}$，实现跳步采样。DDIM 的损失函数与 DDPM 完全相同，只是采样公式不同。

**Q3: Latent Diffusion 为什么要用 VAE 压缩？**
A: 像素空间的维度极高（512×512×3 ≈ 0.8M 维），直接在像素空间做扩散计算量大且语义学习困难。VAE 压缩到 64×64×4 ≈ 16K 维（约 50× 压缩），在潜空间中相邻点有语义相似性，扩散过程更容易学习。同时 VAE 的 KL 正则化使潜空间服从高斯分布，恰好与扩散过程的假设一致。

**Q4: 如何在移动端部署扩散模型？**
A: (1) LCM 蒸馏减少到 2-4 步 (2) UNet 剪枝移除 40-60% 通道 (3) INT8 量化 (4) 使用针对移动 GPU 优化的轻量 UNet 架构 (5) 算子融合减少 kernel launch (6) 针对具体芯片（高通 Adreno、Apple ANE）做后端优化。

**Q5: LCM 为什么能实现少步生成？**
A: LCM（Latent Consistency Model）通过将慢速教师模型（如 1000 步 DDPM）的知识蒸馏到快速学生模型。它学习一个**一致性函数** $f(\mathbf{x}_t, t) \rightarrow \mathbf{x}_0$，要求相邻时间步的映射结果一致。这样学生模型学会**从任意噪声水平直接预测干净图像**，从而一步或少数几步即可完成生成。

**Q6（高难度/FAANG Level）：DDIM 是如何从 DDPM 的马尔可夫链推导出来的？请从数学上证明 DDIM 的采样过程是确定性的，并解释为什么它可以"跳步"而 DDPM 不能。**
A: DDIM 和 DDPM 的数学关系是面试中经常要求"手推公式"的考点。

**DDPM 的马尔可夫假设**：$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) = \mathcal{N}(\mu_\theta(\mathbf{x}_t, t), \sigma_t^2\mathbf{I})$。关键约束是：$\mathbf{x}_{t-1}$ 的分布必须依赖 $\mathbf{x}_t$（通过 $\mu_\theta$）和随机噪声 $\sigma_t \boldsymbol{\epsilon}$。当 $\sigma_t \to 0$ 时，每一步都变成确定性的——但此时 Markov 链的假设不成立（因为 $p(\mathbf{x}_{t-1}|\mathbf{x}_t)$ 在 $\sigma_t=0$ 时退化为 delta 函数，不再是合法的概率分布）。这就是 DDPM 无法跳步的根本原因——跳步意味着跳过中间的 $\mathbf{x}_s$，这破坏了马尔可夫性质。

**DDIM 的非马尔可夫推导**：DDIM 直接假设一个更一般的反向过程族（indexed by a parameter $\sigma$），其中 DDPM 是 $\sigma_t = \sqrt{(1-\bar{\alpha}_{t-1})/(1-\bar{\alpha}_t)} \sqrt{1-\bar{\alpha}_t/\bar{\alpha}_{t-1}}$ 的特例。DDIM 选择 $\sigma_t=0$（确定性极限），得到：
$$\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\left(\frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{\bar{\alpha}_t}}\right)}_{\text{predicted }\mathbf{x}_0} + \sqrt{1-\bar{\alpha}_{t-1}}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$$

这个公式的关键性质是：$\mathbf{x}_{t-1}$ 的表达式**不依赖中间状态 $\mathbf{x}_{t+1}$**（与马尔可夫链不同）。它只依赖 $\mathbf{x}_t$ 和通过 $\boldsymbol{\epsilon}_\theta$ 预测的 $\mathbf{x}_0$。这意味着可以从 $\mathbf{x}_{1000}$ 直接跳到 $\mathbf{x}_{0}$——只要在公式中连续复合（composition）。更一般的跳步公式：
$$\mathbf{x}_{t-k} = \sqrt{\bar{\alpha}_{t-k}} \hat{\mathbf{x}}_0 + \sqrt{1-\bar{\alpha}_{t-k}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$$
其中 $\hat{\mathbf{x}}_0 = (\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}_\theta)/\sqrt{\bar{\alpha}_t}$。

**为什么 DDIM 的训练 loss 和 DDPM 一样**：因为 DDIM 的推理过程使用的 $\boldsymbol{\epsilon}_\theta$ 是在 DDPM 的 loss（$L_{\text{simple}}$）下训练的——这个 loss 只依赖于每个单一时间步的 $\mathbf{x}_t$ 和 $\boldsymbol{\epsilon}$ 的关系，而与该时间步在采样中的前后关系无关。换句话说，DDPM loss 是"时间步局部"的——它只需要模型在任意 $\mathbf{x}_t$ 上预测噪声，而不管 $\mathbf{x}_t$ 是如何到达的（马尔可夫 or not）。这就是 DDIM 可以完全复用 DDPM 训练权重的数学基础。

**DDIM vs DDPM 的速度-质量 trade-off 的精确解释**：
在 1000 步时，DDIM($\sigma=0$) 的生成质量通常略低于 DDPM($\sigma>0$)，因为 DDPM 的随机性提供了"探索"（多样性），而 DDIM 在 1000 步时过度确定。但在 <100 步时，DDIM 的确定性路径更"直"（从噪声到图像的 ODE trajectory 更平滑），而 DDPM 的随机路径在跳步时会"漂移"（因为每步的随机性累积在没有中间步纠正的情况下发散）。这表明 DDIM 和 DDPM 的优劣是步数依赖的——在计算资源充足时用 DDPM（多样性优势），在资源受限时用 DDIM（稳定性优势）。

**Q7（高难度/FAANG Level）：比较三种扩散模型加速方法的根本差异：DDIM（跳步采样）、LCM（一致性蒸馏）、SD-Turbo（对抗蒸馏）。从数学基础和训练成本两个维度分析。**
A: 三种方法代表了扩散模型加速的三个范式：

**DDIM（跳步采样）——零额外训练成本**
- **数学基础**：重写反向过程的参数化，从马尔可夫随机过程改为非马尔可夫 ODE。通过设置 $\sigma_t=0$ 消除随机性，允许在时间轴上做大步跳跃。
- **训练成本**：零。DDIM 的 $\boldsymbol{\epsilon}_\theta$ 就是 DDPM 训练的权重——不需要任何 fine-tuning。
- **质量上限**：约 10-20 步。步数进一步减少时质量急剧下降，因为在 <10 步时 $\boldsymbol{\epsilon}_\theta$ 的预测误差被大步长的 ODE integration 放大。
- **适合场景**：快速原型、batch generation（大量图像并行生成，单张速度可通过 batch 摊销）。

**LCM（Latent Consistency Model）——中等训练成本，质量天花板更高**
- **数学基础**：一致性函数理论。定义函数 $f(\mathbf{x}_t, t) \mapsto \mathbf{x}_0$，要求其在 ODE trajectory 上满足自洽性（self-consistency）：$f(\mathbf{x}_t, t) = f(\mathbf{x}_s, s)$ for any $t, s$ on the same trajectory。训练时，teacher model 产生轨迹上的两个相邻点 $\mathbf{x}_{t_n}, \mathbf{x}_{t_{n+k}}$，student model 被训练为将它们映射到同一 $\mathbf{x}_0$。最终，student 学到了整个 ODE trajectory 的"端点映射"——在推理时，一次 forward 就能从任意 $\mathbf{x}_t$ 跳到 $\mathbf{x}_0$。
- **训练成本**：中等。需要 teacher model 为每条训练数据跑 1-2 个完整采样轨迹（约 $500-1000$ extra forward passes per data point during training）。以 1M training images 为例，teacher inference cost 约 $1M \times 500 \times \text{cost_per_forward} \approx 500\text{M forward passes}$——在 64×A100 上约 3-5 天（$15K-25K）。
- **质量上限**：1-4 步可达 50-step DDIM 的 90-95% 质量（FID 差距 <1）。LCM-LoRA 方案（在已有 SD 权重上训练 LoRA 而非 full LCM）是目前性价比最高的方案——训练成本约 $500（单卡 A100 24h），推理 4 步在 RTX 3090 上约 0.5s。
- **适合场景**：互动式生成（用户每次 prompt 调整都实时反馈）、实时视频生成。

**SD-Turbo（对抗蒸馏 + Score Distillation）——训练成本最高，质量最好**
- **数学基础**：结合了 (a) Adversarial Distillation（用判别器监督 1-step student 的输出质量——对抗 loss 迫使输出"看起来真实"），和 (b) Score Distillation（teacher 的 score function 提供梯度方向——相对于 LCM 的一致性约束，score distillation 是"soft constraint"而非 hard constraint）。损失函数：$\mathcal{L} = \mathcal{L}_{\text{adv}} + \lambda \cdot \mathcal{L}_{\text{score}}$，其中 $\mathcal{L}_{\text{score}} = \mathbb{E}[\nabla_{\mathbf{x}_t} \log p_{\text{teacher}} \cdot \mathbf{x}_{\text{student}}]$。
- **训练成本**：最高。需要同时训练 student generator + discriminator + 保持 teacher 在 GPU 上做 inference。以 SDXL-Turbo 为例，Stability AI 使用了 512×H100 集群训练约 1 周（估计 $250K+$）。但 OpenAI 的 DALL-E 3 和 Stability 的 SD3 使用了类似技术，因为对闭源商业产品来说，训练成本是"一次性"的，推理成本的节省是"永久"的。
- **质量上限**：1 步生成质量 = 50-step DDIM（FID 完全持平甚至略好——因为对抗训练引入了额外的真实感约束）。SDXL-Turbo 在 1 步的 FID(COCO-30K) = 7.9，SDXL 50-step DDIM = 8.1——对抗蒸馏带来的"对抗真实感"补偿了少步带来的模糊。
- **适合场景**：对延迟有极端要求的产品（实时 AI 绘画、视频通话中的实时风格迁移）、需要 <100ms 延迟的场景。

**决策树**：
- 预算 $0, 质量要求中等 → DDIM 10-20 步
- 预算 <$500, 质量要求高 → LCM-LoRA（distill 到 4-8 步）
- 预算 >$50K, 延迟 <100ms → SD-Turbo（adversarial distillation to 1 step）

**Q8（超高难度/Fellow Level）：扩散模型的"去噪"和"生成"本质上是逆问题（inverse problem）。请从 score-based generative modeling 的视角解释：为什么扩散模型的"多步去噪"过程可以被蒸馏为"单步"而不违反信息论的根本限制？单步生成的极限（信息论下界）是什么？**
A: 这是连接扩散模型和 score-based SDE 理论的高阶问题。

**Score-Based 视角**：扩散模型的核心是估计数据分布的 score function $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$，其中 $p_t$ 是 $t$ 时刻的 perturbed distribution。SDE 的前向过程是 $d\mathbf{x} = -\frac{1}{2}\beta(t)\mathbf{x}dt + \sqrt{\beta(t)}d\mathbf{w}$（Ornstein-Uhlenbeck 过程），反向过程是 $d\mathbf{x} = [-\frac{1}{2}\beta(t)\mathbf{x} - \beta(t)\nabla_{\mathbf{x}}\log p_t(\mathbf{x})]dt + \sqrt{\beta(t)}d\tilde{\mathbf{w}}$。扩散模型的"1000 步去噪"本质上是用数值方法（Euler-Maruyama）求解这个反向 SDE——步数对应离散化精度。

**为什么蒸馏不违反信息论**：
蒸馏不是"绕过"反向 SDE，而是"重新参数化"它的解。一致性蒸馏（LCM）的原理是：反向 SDE 定义了一个从 $\mathbf{x}_T$ 到 $\mathbf{x}_0$ 的确定性的概率流 ODE (PF-ODE)（当 diffusion coefficient 取特定形式时）。该 ODE 的解是一个函数 $F: \mathbb{R}^d \times [0,T] \to \mathbb{R}^d$，满足 $F(\mathbf{x}_t, t) = \mathbf{x}_0$。LCM 直接学习这个 $F$ 函数——它不是"从噪声一步跳到图像"的暴力映射，而是学习了 PF-ODE 的**完整解算子**。这之所以可能，是因为 $F$ 是连续且足够光滑的（满足 Lipschitz 连续性），可以被神经网络以任意精度逼近（universal approximation theorem）。从信息论角度，$F$ 的输入和输出维度相同（都是 $\mathbb{R}^d$），不存在信息压缩——蒸馏仅仅是把"ODE numerical integration 的 $T/\Delta t$ 步"替换成"1 步 neural network forward pass"。

**单步生成的极限（信息论下界）**：
单步生成必须在"噪声向数据分布的单射映射"的可行域内操作。设 $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$（纯噪声），$\mathbf{x}_0 \sim p_{\text{data}}$（数据分布）。单步生成器 $G_\theta: \mathbb{R}^d \to \mathbb{R}^d$ 定义了从 $\mathcal{N}(0,I)$ 到 $p_\theta$（生成分布）的 push-forward measure。

信息论下界来自两个约束：
1. **Wasserstein 距离下界**（最优传输理论）：$\mathcal{W}_2(p_{\text{data}}, p_\theta) \geq \text{diam}(\text{supp}(p_{\text{data}})) \cdot \text{TV}(p_{\text{data}}, p_\theta)$。单步映射 $G_\theta$ 的 Lipschitz 常数受限于网络架构（ResNet 的 L≈10, ViT 的 L≈3），因此 $G_\theta$ 无法在高维空间中产生任意复杂的几何变换（如从高斯球面到 ImageNet manifold 的复杂弯曲）。这解释了为什么 SD-Turbo 需要 adversarial loss——判别器的梯度提供了超出单纯 score matching 的 geometric warping 信号。
2. **KL 散度下界**（data processing inequality）：$\text{KL}(p_{\text{data}} \| p_\theta) \geq \text{KL}(p_{\text{data}} \| p_T) - I(\mathbf{x}_T; \mathbf{x}_0)$，其中 $p_T = \mathcal{N}(0,I)$。这意味着单步生成的质量受限于噪声和数据之间的互信息 $I(\mathbf{x}_T; \mathbf{x}_0)$。对于 1000-step DDPM，每一步引入的噪声增量极小，$I(\mathbf{x}_T; \mathbf{x}_0)$ 被逐步"温和"地破坏——每一步仅损失少量信息。直接一步生成时，$\mathbf{x}_T$ 和 $\mathbf{x}_0$ 之间的互信息几乎为 0（因为 $T=1000$ 步的噪声累积使得 $\mathbf{x}_T \approx \mathcal{N}(0,I)$ 独立于 $\mathbf{x}_0$），单步映射需要从零信息重建全部结构——这是极端困难的。

**实践中的极限观察**：当前 SOTA（SD3-Turbo, PixArt-Sigma）的 1 步生成的 FID 约 5-8（ImageNet 256²），而 1000-step DDPM 约 2-3。这 3-5 个点的 FID 差距就是"单步生成的信息论代价"——它可以被进一步缩小（通过更大的模型、更多的对抗训练），但不太可能被完全消除，因为 $I(\mathbf{x}_T; \mathbf{x}_0)$ 在连续时间扩散中趋于 0 的速度受限于 diffusion coefficient 的 schedule。

**Q9（超高难度/Fellow Level）：请设计一个在消费级 GPU（RTX 4090 24GB）上实现"实时"文本到图像生成的完整 pipeline。目标：<200ms 端到端延迟生成 1024² 图像，同时保持 SDXL 的质量水平。** 
A: 这是 Stable Diffusion 社区目前的"圣杯"目标。以下是完整的工程设计：

**Pipeline 总览**：
文本编码(5ms) → UNet 前向(N×8ms) → VAE 解码(15ms) → 后处理(5ms)。若 N=4 steps，UNet 总耗时 32ms，pipeline 总计 57ms —— 加上 kernel launch + memory transfer overhead 约 10ms，pipeline 约 67ms，远在 200ms budget 内。但真正的瓶颈不是延迟而是显存和质量。

**核心组件选型**：

**1. Text Encoder（~5ms, VRAM 200MB）**
使用 CLIP ViT-L/14 text encoder（而非 SDXL 的双 CLIP——OpenCLIP ViT-bigG 太重）。在 4090 上 FP16 约 5ms。替代方案：Google T5-XXL encoder 虽然质量更好但 11B 参数太重——用 T5-Small（60M）做 95% 场景足够了。

**2. UNet（N×8ms, VRAM ~4GB）**
- 使用 SDXL-Turbo 的 UNet（一步蒸馏版本，4 步可达 50-step DDIM 质量）
- SVDQuant 4-bit 量化：VRAM 从 5.2GB → 1.3GB
- Nunchaku 推理引擎（SVD 分解 + low-rank + residual 混合并行 GEMM）
- CFG 使用 guidance schedule：step 1-2 用 CFG=7.5（定大方向），step 3-4 用 CFG=3.0（细化细节）——单步 UNet forward 减少 50%（因为低 CFG 时 unconditional path 和 conditional path 的差异更小，可用更小的 batch 或更轻量的 compute）。

**3. VAE Decoder（~15ms, VRAM ~1.5GB）**
这是最容易忽视的瓶颈。SDXL 的 VAE decoder 将 128×128×4 latent 解码为 1024×1024×3 RGB——包含 3 个上采样 block 和一个最终卷积。4090 上 FP16 约 50ms——占总延迟的 50%+！优化的关键：
- **TAESD (Tiny AutoEncoder)**：将 VAE decoder 从 80M params 压缩到 1.5M（使用 MobileNetV3 风格的深度可分离卷积）。在 4090 上仅需 1.8ms——但质量有损失（PSNR 从 32dB → 28dB）。对 1024² 图像，TAESD 的 artifact 在纹理区域可见。
- **混合方案（推荐）**：TAESD 做 fast preview（<2ms, 用于实时交互），当用户停止 typing 0.5 秒后，用 full VAE decoder 再做一次精细解码（50ms, 覆盖之前的 preview image）——用户感知到的延迟 <50ms（因为 preview 已经在屏幕上了）。
- **SVDQuant VAE**：将 VAE decoder 也做 4-bit 量化，VRAM 从 1.5GB → 0.4GB。FP16 的 50ms latency 降到 25ms（因 memory bandwidth 降低 2.5× on 4090 consumer tensor core）。

**4. VRAM 分配（24GB 总预算）**
- CLIP text encoder: 0.2GB
- UNet 4-bit weights: 1.3GB
- UNet activations (batch=1, 128×128 latent): 3.5GB
- VAE decoder 4-bit: 0.4GB
- VAE activations (1024×1024): 4.2GB
- CUDA context + misc: 2GB
- **Total**: 11.6GB —— 余量充足。甚至可以运行 batch=2 的 parallel CFG（conditional + unconditional path 并行）。

**5. 最小化 kernel launch overhead**
4090 的 kernel launch latency ~3μs。UNet 的跨注意力（cross-attention）有数十个小 matmul——总计 ~200 kernel launches per step, 4 步 = 800 launches ≈ 2.4ms overhead。解决方案：CUDA Graph capture——在 warmup 阶段 capture 整个 UNet forward graph，后续直接 replay，kernel launch overhead 降为 ~0.1ms。

**6. 端到端时序（实测估计）**
| 阶段 | 优化前 | 优化后 |
|------|--------|--------|
| Text encoding | 5ms | 5ms (无变化) |
| UNet × 4 steps | 200ms | 35ms (Turbo蒸馏 + SVDQuant + CUDA Graph) |
| VAE decoding | 50ms | 25ms (SVDQuant VAE) |
| Post-processing | 10ms | 10ms (无变化) |
| **Total** | **265ms** | **75ms** |

**剩余差距 —— 从 75ms 到 <50ms 的额外优化**：
- TAESD preview：让用户感觉 <10ms（preview image 在 10ms 内出现，3 秒后被 full decode 替换）
- Tensor parallelism across 2×4090：UNet 在两张卡上分片——但 4090 的 NVLink 带宽受限（不如 A100），TP 可能比单卡还慢
- 期待 NVIDIA 50-series 的硬件改进（Blackwell 的 FP4 support 可进一步提升 2-3×）

**结论**：在 RTX 4090 上实现 <200ms 的 SDXL 级 1024² 生成现在是完全可行的（SVDQuant + Nunchaku + LCM-LoRA 方案已经达到 ~75ms）。突破 <50ms 需要等待硬件的 memory bandwidth 进一步升级。

## 10. 本讲总结

扩散模型是当前生成式 AI 的核心技术，它将图像生成建模为"逐步去噪"的过程。DDPM 奠定了理论框架（1000 步随机去噪），Latent Diffusion 将操作空间从像素压缩到潜变量（使生成在高分辨率上可行），DDIM 通过去随机化实现跳步采样（压缩到 50 步），LCM 等蒸馏方法进一步压缩到 1-4 步。

从效率角度看，扩散模型的推理成本极高——每一步都需运行一次完整的 UNet（通常包含残差块 + 自注意力 + 交叉注意力），降低步数和压缩模型同等重要。稀疏化（UNet 结构剪枝、注意力头剪枝）和量化（时步感知 INT8）是部署到资源受限设备的关键技术。

将扩散模型部署到端侧（手机/嵌入式）仍是一个 open problem，但结合蒸馏、量化、轻量架构设计，已经在高通和 Apple 平台上取得了初步成果。这也是 TinyML 社区当前最活跃的研究方向之一。

## 11. 工业落地checklist

| 检查项 | 说明 | 不做的后果 |
|--------|------|-----------|
| DDIM 步数 < 10 时不能仅依赖 FID/CLIP score 评估——必须加入结构级指标 | 某 AI 艺术平台 SDXL 50→8 steps：FID 从 7.2→7.8（看似可接受）、CLIP score 不变，但用户投诉手部扭曲/脸不对称——FID 不敏感于 5% catastrophic failure | 自动化测试通过但用户投诉暴增，上线 3 天紧急回滚 |
| 扩散模型 INT8 量化必须使用 timestep-aware calibration——不同 t 的激活分布差异巨大 | 某游戏公司 standard PTQ：前 20 steps 正常，step 20+ 输出逐渐变暗至全黑——小 t 时的 outlier 激活被错误 clipping（calibration 将所有 t 混一起算 scale） | 量化模型生成全黑图像，"优化"结果完全不可用 |
| LCM 蒸馏时 CFG scale 在少步数下需降低（3-4 而非 7.5） | 某 AI 头像服务 LCM 4-step + CFG=7.5：每步 CFG 增量放大 12.5x → "微笑/放松"表情被抑制，"愤怒/严肃"成为主导 → 所有女生头像都像生气 | 产品上线后用户抱怨"为什么生成的头像都在瞪我" |
| VAE decoder 是 SDXL 推理的最大延迟瓶颈（占 50%+），必须用 TAESD 或 SVDQuant | RTX 4090 上 SDXL UNet 4-steps 仅 35ms，VAE decoder 却占 50ms——不做 VAE 优化，UNet 优化收益被减半 | UNet 从 200ms 压缩到 35ms 后瓶颈转移到 VAE，延迟仍 > 80ms |
| 扩散模型蒸馏必须保证 teacher 和 student 使用相同 ODE solver | Progressive Distillation 中 teacher 用 DDIM、student 用 DPM-Solver → 蒸馏 loss 和推理行为不一致 → student 质量远差于预期 | 蒸馏训练"成功"（loss 下降）但推理质量不升反降，白费训练资源 |
| 移动端部署扩散模型必须将 UNet + VAE + Text Encoder 分开做独立压缩 | Google MobileDiffusion 方案：UNet 剪枝 86% + 8-step 蒸馏 + Edge TPU INT8；各组件瓶颈不同，统一量化/剪枝率效果差 | 某组件过度压缩成为短板，整体延迟和质量均不达标 |
| SDXL 4-bit 量化必须使用 SVDQuant 而非纯 INT4，否则 consumer GPU tensor core 利用率低 | RTX 4090 的 consumer tensor core 上纯 INT4 GEMM 受限于 mma.sync warp 调度；SVDQuant 的 low-rank FP16 + residual INT4 并行方案额外加速 1.8x | 4-bit 量化后延迟仅降 1.6x（记忆带宽是瓶颈），远低于预期的 4x |
