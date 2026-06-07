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

## 10. 本讲总结

扩散模型是当前生成式 AI 的核心技术，它将图像生成建模为"逐步去噪"的过程。DDPM 奠定了理论框架（1000 步随机去噪），Latent Diffusion 将操作空间从像素压缩到潜变量（使生成在高分辨率上可行），DDIM 通过去随机化实现跳步采样（压缩到 50 步），LCM 等蒸馏方法进一步压缩到 1-4 步。

从效率角度看，扩散模型的推理成本极高——每一步都需运行一次完整的 UNet（通常包含残差块 + 自注意力 + 交叉注意力），降低步数和压缩模型同等重要。稀疏化（UNet 结构剪枝、注意力头剪枝）和量化（时步感知 INT8）是部署到资源受限设备的关键技术。

将扩散模型部署到端侧（手机/嵌入式）仍是一个 open problem，但结合蒸馏、量化、轻量架构设计，已经在高通和 Apple 平台上取得了初步成果。这也是 TinyML 社区当前最活跃的研究方向之一。
