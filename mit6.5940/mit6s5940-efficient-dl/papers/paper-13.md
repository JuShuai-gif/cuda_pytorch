# GAN Compression: Efficient Architectures for Interactive Conditional GANs

> Muyang Li et al., CVPR 2020

## 1. 论文解决什么问题

条件生成对抗网络（cGANs，如 pix2pix、CycleGAN、MUNIT、GauGAN）在图像到图像翻译、超分辨率、风格迁移等任务上展示了卓越的视觉质量。然而，cGAN 的生成器通常非常庞大（计算量可达数百 GOPs），推理延迟高（在 GPU 上都需要数秒），无法部署到边缘设备或实现实时交互。

直接将分类网络的压缩方法（剪枝、量化）应用于 GAN 生成器效果不佳，原因包括：(1) GAN 生成器的输出是图像而非单一标签，对细微的参数变化比分类网络敏感得多——轻微的剪枝可能导致视觉伪影（artifacts）；(2) 生成器和判别器在训练过程中相互博弈，使得直接微调剪枝后的生成器困难；(3) 缺少针对 GAN 生成器特性的压缩策略。

本文提出了 **GAN Compression**，一种专门针对 cGAN 生成器的压缩框架，在 **9-21× 压缩比**下保持视觉质量，使交互式条件生成在边缘设备上成为可能。

## 2. 核心方法

GAN Compression 采用三阶段压缩流水线：

### 2.1 第一阶段：Once-for-All 训练（教师模型）

首先生成一个"万能教师"（once-for-all teacher generator）。继承 Once-for-All（OFA）网络的理念：
- 训练一个超网生成器，其中每层包含多个不同 channel 数的候选子模块（如 original/0.75×/0.5×/0.25× channels）
- 使用渐进式收缩（progressive shrinking）训练策略：先训练完整模型，再逐步修剪 channel 数并微调
- 该教师模型中嵌入了大量不同大小的子生成器，可供后续步骤选择

这一步的关键优势是：只需要一次昂贵的生成器训练，便可用作所有后续压缩步骤的知识源。

### 2.2 第二阶段：通道剪枝 + 知识蒸馏（压缩）

在教师模型的指导下，通过"剪枝 + 蒸馏"两步压缩目标子生成器：

**通道剪枝（Channel Pruning）**：
- 对每层的卷积核进行通道级别的剪枝（结构化剪枝）
- 剪枝准则：使用 L1-norm（每个 channel 的权重绝对值之和），移除 L1-norm 最小的 channels
- 剪枝后的生成器在结构上是紧凑的（输出 channel 数减少，后续层输入 channel 数也对应减少）

**知识蒸馏（Knowledge Distillation）**：
- 将剪枝后的"学生"生成器与原始"教师"生成器的输出进行像素级和感知级对齐
- 蒸馏损失包括多种形式：
  - **逐像素 L1/L2 损失**：约束学生输出与教师输出在像素空间上接近
  - **感知损失（Perceptual Loss）**：通过预训练的 VGG-19 网络提取高层特征，约束学生和教师在感知特征空间上一致
  - **判别器特征匹配（Discriminator Feature Matching）**：将学生和教师的生成图像分别送入预训练判别器，约束判别器中间层特征一致

### 2.3 第三阶段：NAS 搜索最优通道配置

在剪枝和蒸馏后，使用 NAS（神经架构搜索）来确定每层的最优通道数分配。有了 Once-for-All 教师作为知识库，这一步可以高效完成：

- 从 OFA 教师中采样不同的通道配置
- 在少量验证图像上评估每个候选配置的生成质量（使用蒸馏后的轻量判别器）
- 使用进化算法在"质量-延迟"帕累托前沿上搜索

## 3. 关键公式（LaTeX）

**GAN 标准目标**：

$$
\mathcal{L}_{\text{GAN}}(G, D) = \mathbb{E}_{x,y}[\log D(x, y)] + \mathbb{E}_{x}[\log(1 - D(x, G(x)))]
$$

**总蒸馏损失**：

$$
\mathcal{L}_{\text{distill}} = \lambda_1 \mathcal{L}_{\text{pixel}} + \lambda_2 \mathcal{L}_{\text{perceptual}} + \lambda_3 \mathcal{L}_{\text{feature\_match}}
$$

其中：

- **像素损失**：$\mathcal{L}_{\text{pixel}} = \|G_{\text{teacher}}(x) - G_{\text{student}}(x)\|_1$
- **感知损失**：$\mathcal{L}_{\text{perceptual}} = \sum_{l} \|\phi_l(G_{\text{teacher}}(x)) - \phi_l(G_{\text{student}}(x))\|_2$，$\phi_l$ 为 VGG-19 第 $l$ 层特征
- **特征匹配损失**：$\mathcal{L}_{\text{feature\_match}} = \sum_{k} \|D_k(x, G_{\text{teacher}}(x)) - D_k(x, G_{\text{student}}(x))\|_2$，$D_k$ 为判别器第 $k$ 层

**通道剪枝准则**：

$$
I_c = \|W^{(l)}_{:,c,:,:}\|_1
$$

按 $I_c$ 从小到大的顺序剪除 channels。

**资源约束搜索（NAS 目标）**：

$$
\min_{\alpha} \mathcal{L}_{\text{perceptual}}(G_\alpha) \quad \text{s.t.} \quad \text{MACs}(G_\alpha) \leq T
$$

其中 $\alpha$ 为每层的通道选择参数，$T$ 为目标 MACs 预算。

## 4. 实验结论

- **压缩效果**（以 CycleGAN 为例）：
  - 生成器 MACs 压缩 **9-21×**
  - 模型大小压缩 **5-33×**（与 ResNet-based 基线对比）
  - 在 Jetson Xavier NX 上，延迟从 2.6 秒/帧降至 **0.15 秒/帧**（约 17× 加速）
  - 视觉效果：FID（Frechet Inception Distance）基本不降或微增 <5 个点
- **主要 GAN 架构的压缩结果**：
  - pix2pix (ResNet Gen): 20.4× MACs 压缩, FID 从 52.3 → 53.1（几乎无损）
  - CycleGAN (ResNet Gen): 10.6× MACs 压缩, FID 从 65.7 → 66.3
  - GauGAN (SPADE Gen): 9.2× MACs 压缩, FID 从 35.9 → 37.8
  - MUNIT: 21.3× MACs 压缩, FID 从 28.6 → 30.1
- **三种蒸馏损失的贡献**（消融实验）：
  - 仅像素损失：FID 退化 >20 点
  - 像素 + 感知损失：FID 退化约 10 点
  - 像素 + 感知 + 特征匹配：FID 退化 <5 点（完整方案）
- **相比分类网络压缩**：
  - 分类网络通道剪枝后仅需 cross-entropy fine-tuning
  - GAN 生成器剪枝后，仅用 GAN loss fine-tuning 会导致严重伪影（mode collapse / checkerboard artifacts），必须依赖蒸馏
- **交互式应用**：CycleGAN 压缩后在 Jetson AGX Xavier 上可实时运行（>20 FPS），支持摄像头输入实时风格迁移

## 5. 工业价值

- **使实时 GAN 成为现实**：在 AR/VR、实时视频特效、拍照翻译等场景中，GAN 压缩使得原本只能在服务器上离线运行的模型能在手机和边缘设备上实时运行
- **商业应用场景**：
  - **Snapchat/Instagram 滤镜**：实时风格迁移、人脸属性编辑
  - **Adobe Photoshop**：图像到图像翻译（草图→照片、白天→夜景）
  - **自动驾驶仿真**：实时生成不同天气/光照条件下的道路场景
  - **游戏开发**：实时纹理合成和风格化渲染
- **系统化的 GAN 压缩框架**：成为后续 GAN 压缩/加速工作的标准基线，包括 Content-Aware GAN Compression、DMAD（Differentiable Model Acceleration and Distillation）等
- **OFA 理念的扩展**：将 Once-for-All 范式从分类网络扩展到了 GAN 生成器，展示了"一次训练，多次压缩"在生成模型上的可行性

## 6. 与课程 lecture 的关系

- **Lecture 17（Efficient GANs）**：本文是 Lecture 17 的核心论文。Lecture 17 首先介绍 GAN 的基础知识（生成器 vs 判别器、条件 GAN、pix2pix/CycleGAN），然后引入 GAN 压缩的挑战（为什么分类网络的压缩方法不能直接用于 GAN），最后详细讲解 GAN Compression 的三阶段流水线（OFA 训练 → 剪枝+蒸馏 → NAS 搜索）。Lecture 17 也会介绍其他高效 GAN 方法如 FastGAN、MobileStyleGAN 等作为补充。

## 7. 我应该如何复现

1. **官方代码库**：
   - GAN Compression：`https://github.com/mit-han-lab/gan-compression`
   - 依赖：PyTorch >= 1.4, torchvision, dominate (用于 CycleGAN/pix2pix)

2. **核心实现骨架**：
   ```python
   class GANCompressor:
       def __init__(self, teacher_G, teacher_D):
           self.teacher_G = teacher_G
           self.student_G = prune_channels(teacher_G)
           self.vgg = vgg19(pretrained=True).features

       def distillation_loss(self, real_A, fake_B_student, fake_B_teacher):
           # Pixel loss
           loss_pixel = F.l1_loss(fake_B_student, fake_B_teacher)
           # Perceptual loss (VGG)
           feat_s = self.vgg(fake_B_student)
           feat_t = self.vgg(fake_B_teacher).detach()
           loss_perceptual = F.mse_loss(feat_s, feat_t)
           # Feature matching (discriminator)
           d_feat_s = self.D.get_features(real_A, fake_B_student)
           d_feat_t = self.D.get_features(real_A, fake_B_teacher).detach()
           loss_fm = sum(F.mse_loss(s, t) for s, t in zip(d_feat_s, d_feat_t))
           return loss_pixel + 10 * loss_perceptual + loss_fm

       def prune_channels(self, G, ratio=0.5):
           for name, module in G.named_modules():
               if isinstance(module, nn.Conv2d):
                   # L1-norm channel pruning
                   importance = module.weight.abs().sum(dim=(1,2,3))
                   keep = int(module.out_channels * ratio)
                   kept_idx = importance.topk(keep).indices
                   module.weight.data = module.weight.data[kept_idx]
                   ...
   ```

3. **简化复现路线**：
   - **Phase 1（理解 GAN 基础）**：用 `pytorch-CycleGAN-and-pix2pix` 仓库训练一个小型 CycleGAN（如 horse2zebra），熟悉生成器/判别器的训练循环和视觉质量评估（FID）
   - **Phase 2（通道剪枝）**：对训练好的 CycleGAN 生成器进行 L1-norm 通道剪枝（50%、75%），直接评估剪枝前后的视觉效果差异（不用蒸馏），观察伪影类型
   - **Phase 3（蒸馏）**：实现完整的三种蒸馏损失（pixel + perceptual + feature matching），对比仅剪枝 vs 剪枝+蒸馏的视觉效果
   - **Phase 4（NAS 搜索）**：在剪枝后的搜索空间上运行简单的 grid search（如每层 {0.25, 0.5, 0.75} 候选），找到质量最优的子生成器
   - **Phase 5（边缘设备推理）**：将压缩模型导出为 ONNX → TensorRT，在 Jetson Nano 上测量实际 FPS

4. **主要超参数**：
   - 蒸馏损失权重：$\lambda_1=1$ (pixel), $\lambda_2=10$ (perceptual), $\lambda_3=1$ (feature match)；感知损失权重最大，对视觉质量影响最显著
   - 学生生成器学习率：$2\times10^{-4}$（Adam），蒸馏 30-50 epochs
   - 通道剪枝比例：初始目标 50-75% MACs 减少
   - VGG 特征层：使用 relu1_2, relu2_2, relu3_3, relu4_3 四层
   - 判别器：使用原始教师判别器（不压缩）

5. **评估指标**：
   - **FID（Frechet Inception Distance）**：生成质量的主要量化指标（越低越好，<5 点变化通常不可见）
   - **LPIPS（Learned Perceptual Image Patch Similarity）**：与教师输出的感知相似度
   - **MACs / #Params / 延迟（ms）**：效率指标

6. **常见坑**：
   - 蒸馏时需要使用教师生成器的**推理模式输出**（teacher_G.eval()），而非训练模式
   - 感知损失中的 VGG 特征需要归一化（用 ImageNet 均值和标准差），否则数值范围不一致
   - 如果只做通道剪枝不调整后续层结构，会导致通道数不匹配；需要跟踪依赖图确保剪枝后的网络计算图有效
   - GAN 训练的不稳定性在压缩过程中可能被放大，建议监控 GAN loss 比值 $G_{\text{loss}} / D_{\text{loss}}$ 避免 mode collapse
   - 不同 GAN 模型对压缩的敏感度差异很大：CycleGAN > pix2pix > GauGAN（越复杂的生成任务压缩越困难）
