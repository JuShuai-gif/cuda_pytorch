# MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning

> Ji Lin et al., NeurIPS 2021

## 1. 论文解决什么问题

MCUNet（MCUNetV1）首次将 ImageNet 级深度学习推理带到商用量产 MCU（<256KB SRAM, <1MB Flash）上。但 MCUNetV1 存在一个关键瓶颈：**峰值内存（peak memory）**由第一层卷积的输入激活张量决定。对于典型的 ImageNet 224×224 输入，仅第一层的输入激活就需要 224×224×3×1byte ≈ 150KB（使用 int8 量化），在 256KB SRAM 下，留给后续层的剩余内存非常紧张。当输入分辨率提高到 320×320（对下游任务更友好）或使用更深的网络时，内存瓶颈变得更加严重。

MCUNetV2 解决了 **TinyML 推理中的峰值内存瓶颈问题**：如何在 MCU 有限的 SRAM 约束下，支持较大的输入分辨率和更深的网络拓扑。

## 2. 核心方法

### 2.1 基于分块（Patch-based）的推理

核心思想是**不在第一层一次性处理整个输入图像，而是将图像切割成重叠的小块（patches），逐块送入网络进行推理**。

具体流程：
1. 将输入图像 $X \in \mathbb{R}^{H \times W \times C}$ 按网格划分为 $P \times P$ 个重叠的 patches
2. 每个 patch 独立通过网络的初始若干层（conv stem），得到每个 patch 的中间特征图
3. 将所有 patches 的中间特征图按照空间位置拼接回完整特征图
4. 拼接后的特征图继续通过网络的剩余层

通过分块，峰值内存从 $O(H \times W)$ 降低到 $O(h \times w)$，其中 $h \times w$ 是单个 patch 的尺寸。

### 2.2 重叠分块与感受野再分配（Redistributed Receptive Field）

直接分块会导致 patch 边缘的信息丢失和块效应（blocking artifacts）。MCUNetV2 通过两种机制解决：

1. **重叠分块（Overlapping Patches）**：相邻 patches 之间保持一定比例的重叠（如 patch_size=48, overlap=16），确保边缘像素被多块覆盖

2. **感受野再分配（Receptive Field Redistribution）**：
   - 观察：当使用分块推理时，将网络的感受野（receptive field, RF）从"更大且稀疏"重新分配为"更小但密集"，使每个 patch 的感受野主要覆盖本 patch 内容，减少跨 patch 的远距离依赖
   - 具体做法：将网络浅层的大卷积核（如 7×7）替换为多层小卷积核（3×3），使 RF 更集中在 patch 内部
   - 数学上：保持总感受野大小不变，但通过减少大卷积核步长 + 增加层数来增大 patch 内像素的利用率

### 2.3 架构设计

MCUNetV2 的架构（包括分块策略、stem 层设计、patch size、overlap）通过 **TinyNAS** 自动搜索获得，而非手工设计。TinyNAS 根据目标 MCU 的 SRAM/Flash 约束和分块策略共同优化网络架构。

## 3. 关键公式（LaTeX）

**Patch 化操作**：

$$
X_{p}^{(i,j)} = X[(i-1)s_p+1 : (i-1)s_p + H_p, \; (j-1)s_p+1 : (j-1)s_p + W_p, \; :]
$$

其中 $(i,j)$ 为 patch 网格坐标，$H_p \times W_p$ 为 patch 尺寸，$s_p$ 为 patch 步长（$s_p < H_p, W_p$ 即有重叠）。

**峰值内存节省**：

$$
M_{\text{patch}} \propto H_p \times W_p \times C_{\text{mid}}
$$

vs

$$
M_{\text{full}} \propto H \times W \times C_{\text{mid}}
$$

例如当 $H=W=224$, $H_p=W_p=48$ 时：$M_{\text{patch}} / M_{\text{full}} \approx (48/224)^2 \approx 4.6\%$。

**特征图拼接**：

$$
F_{\text{merged}} = \text{ConcatGrid}\left(\{f_\theta(X_p^{(i,j)})\}_{i=1,j=1}^{P,P}\right)
$$

在拼接边界处使用平均池化或线性混合处理重叠区域：

$$
F_{\text{merged}}(x,y) = \frac{\sum_{p \in \mathcal{P}(x,y)} F_p(x - x_p, y - y_p)}{|\mathcal{P}(x,y)|}
$$

其中 $\mathcal{P}(x,y)$ 为覆盖像素 $(x,y)$ 的所有 patches 集合。

**计算开销（由于重叠导致的计算重复）**：

$$
\text{FLOPs}_{\text{patch}} = \text{FLOPs}_{\text{full}} \times \frac{\text{\#patches} \times \text{Area\_per\_patch}}{\text{Total\_Area}} \times \text{Overlap\_Factor}
$$

Overlap Factor 通常在 1.3-2.0× 之间（因为重叠区域被多次计算），但仍在 MCU 可接受范围内。

## 4. 实验结论

- **内存突破**：
  - MCUNetV2 将峰值内存降低 **4-8×**，使更深的网络和更大的输入分辨率（320×320）能在 <256KB SRAM MCU 上运行
  - 在 STM32F746（320KB SRAM）上，MCUNetV2 支持 320×320 输入，MCUNetV1 无法运行（内存不足）
- **精度提升**：
  - 在 ImageNet 上：MCUNetV2（320×320 输入，4.9M MACs, 168KB SRAM）精度 72.2%（Top-1），比 MCUNetV1 精度最高的模型（72.0%）高 0.2% 但内存更低
  - 大输入分辨率 + 分块推理使下游迁移任务（目标检测、语义分割）的精度显著优于低分辨率全图推理
- **下游任务效果**：
  - 在 Pascal VOC 目标检测上，MCUNetV2 作为 backbone 的精度比 MCUNetV1 高 **3-5 个 mAP 点**
  - 在 ADE20K 语义分割上，mIoU 提高 2-4 点
- **消融实验**：
  - 感受野再分配（RF redistribution）贡献了约 1.0% 的精度提升
  - 重叠分块（overlap > 0）比无重叠分块精度高 2-3%
  - 最优 patch size 约为 48-64（由 TinyNAS 自动搜索确定）
- **实测延迟**：MCUNetV2 在 STM32H743 MCU 上的推理延迟约 300-400ms，虽然比 V1 略高（因重叠计算），但仍在可部署范围（<1 秒）

## 5. 工业价值

- **将 TinyML 从分类扩展到感知任务**：分块推理使得目标检测、语义分割、姿态估计在 MCU 上成为可能——这些任务天然需要更高分辨率的输入
- **突破"第一层瓶颈"**：几乎所有 TinyML 模型都受限于第一层输入的内存，MCUNetV2 的分块方案具有通用性——可以与其他 TinyML 模型架构（MobileNetV3-Small、EfficientNet-lite 等）配合使用
- **推动 MCU AI 商业化**：
  - ARM 的 CMSIS-NN 库已在部分实现中支持分块卷积
  - STM32Cube.AI 工具链中引入了类似的分块机制
  - 智能门锁、可穿戴设备、无线传感器等需要高分辨率图像处理的 IoT 场景直接受益
- **内存与计算的权衡哲学**：MCUNetV2 展示了"用计算换内存"在资源极度受限场景下的有效性——分块带来 1.3-2.0× 额外 FLOPs，但换取 4-8× 内存减少

## 6. 与课程 lecture 的关系

- **Lecture 10（MCUNet）**：本文是 Lecture 10 的重要组成部分。课程从 MCUNetV1（TinyNAS + TinyEngine = 首次 ImageNet 推理上 MCU）讲起，然后引入 MCUNetV2 如何解决 V1 的峰值内存瓶颈（分块推理 + 感受野再分配）。Lecture 10 同时介绍 MCUNetV3（on-device training）作为延伸，形成 MCUNet 系列的完整故事线。

## 7. 我应该如何复现

1. **环境准备**：
   - MCUNet 官方代码：`https://github.com/mit-han-lab/mcunet`
   - 安装 TinyEngine 推理库（C/C++，用于 MCU 部署）
   - Python 依赖：PyTorch >= 1.7, torchvision, numpy

2. **核心实现要点**：
   ```python
   class PatchInference(nn.Module):
       def __init__(self, backbone, patch_size=48, overlap=16):
           self.backbone = backbone
           self.patch_size = patch_size
           self.stride = patch_size - overlap  # 控制重叠

       def forward(self, x):
           B, C, H, W = x.shape
           patches = F.unfold(x, kernel_size=self.patch_size,
                              stride=self.stride)
           # 将每个 patch 送入 stem
           patch_feats = self.backbone.stem(patches.reshape(...))
           # 拼接特征图
           feat_map = F.fold(patch_feats, output_size=(H, W), ...)
           # 继续通过 later stages
           return self.backbone.later_stages(feat_map)
   ```

3. **简化复现路线**：
   - **Phase 1（理解分块）**：在 CIFAR-10 上用 ResNet-18 实现简单的 non-overlapping 分块，验证分块前后的输出一致性（除了边缘）
   - **Phase 2（添加重叠）**：实现 overlapping patches，对比有/无重叠的精度差异
   - **Phase 3（RF 再分配）**：修改 stem 层，将 7×7 Conv 替换为 3×3 Conv × 3，观察对分块推理精度的影响
   - **Phase 4（MCU 部署）**：将训练好的 PyTorch 模型通过 TinyEngine（或 TensorFlow Lite Micro）部署到 STM32 开发板

4. **开发板推荐**：
   - **STM32F746G-DISCO**（320KB SRAM, 1MB Flash）：最常用的参考平台
   - **STM32H743**（1MB SRAM, 2MB Flash）：更高规格，可运行更大模型
   - **Raspberry Pi Pico**（264KB SRAM, 2MB Flash）：低成本替代方案

5. **关键配置参数**：
   - Patch size: 48-64（通过 TinyNAS 搜索确定）
   - Overlap: 16-24 pixels（够用就行，太大增加计算量）
   - Stem layers: 2-3 个 3×3 卷积层（替代单一的大卷积核）
   - 量化：int8 per-channel 量化（TinyEngine 支持高效 int8 推理）

6. **常见坑**：
   - 分块拼接时的特征值归一化：重叠区域需要按覆盖次数取平均，否则会亮度异常
   - TinyEngine 编译需要正确的 ARM GCC toolchain 版本（通常是 `arm-none-eabi-gcc`）
   - MCU 上的内存对齐要求：分配 tensor 时必须考虑字节对齐（如 CMSIS-NN 需要 4-byte 对齐）
   - 分块卷积在 MCU 上的实现与 PyTorch 模拟可能略有差异（数值精度）
