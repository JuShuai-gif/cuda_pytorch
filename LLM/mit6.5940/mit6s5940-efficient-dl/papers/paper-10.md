# HAQ: Hardware-Aware Automated Quantization with Mixed Precision

> Kuan Wang et al., CVPR 2019

## 1. 论文解决什么问题

模型量化是压缩深度学习模型的重要手段——将 32-bit 浮点权重的激活值降低到 8-bit、4-bit 甚至更低。但传统量化方法存在以下问题：(1) **统一精度量化（uniform precision）**对所有层使用相同的位宽，忽视了不同层对量化的敏感度差异——某些层（如第一层和最后一层）对精度敏感，需要更高位宽；其他层可以激进量化；(2) **手工调优位宽**需要大量专家经验和反复试错；(3) **FLOPs/模型大小不等于实际硬件效率**，不同硬件平台（GPU、CPU、FPGA、手机 DSP）对低精度运算的支持差异巨大。

本文提出了 **HAQ（Hardware-Aware Automated Quantization）**：使用强化学习自动为每层搜索最优位宽配置，同时将目标硬件的实际延迟/能耗反馈融入搜索过程，实现硬件感知的混合精度量化。

## 2. 核心方法

### 2.1 搜索空间与问题定义

将网络的每一层（卷积层、全连接层）的权重和激活值的位宽作为搜索变量。对于包含 $N$ 层的网络：
- 权重位宽：$b_{w1}, b_{w2}, ..., b_{wN} \in \{2, 3, 4, 5, 6, 7, 8\}$
- 激活位宽：$b_{a1}, b_{a2}, ..., b_{aN} \in \{2, 3, 4, 5, 6, 7, 8\}$

搜索空间大小约为 $(7^2)^N = 49^N$，对典型网络（N=50+），穷举搜索不可行。

### 2.2 强化学习智能体（DQN Agent）

使用 Deep Q-Network（DQN）作为搜索智能体。搜索建模为序列决策过程：
- **状态（State）**：当前层的索引、输入/输出通道数、kernel size、stride、前一层已分配的位宽
- **动作（Action）**：为当前层选择（权重位宽, 激活位宽）的组合
- **奖励（Reward）**：整个网络量化后的精度与目标硬件延迟/能耗的加权组合

DQN 的神经网络输入为状态向量，输出为每个可选动作的 Q 值。采用 $\epsilon$-greedy 策略进行探索。

### 2.3 硬件反馈环路

HAQ 的关键创新是将真实硬件反馈嵌入 RL 搜索循环：

1. **硬件模拟器**：对每一层，在目标设备上实际测量不同位宽下的延迟和能耗
2. **构建延迟/能耗查表**：将测量结果封装为查表，供 RL agent 奖励计算使用
3. **硬件约束**：将硬件约束（延迟阈值）转化为惩罚，约束搜索方向

这使得搜到的混合精度方案在真实硬件上能获得**实际加速**（而非仅理论加速）。

### 2.4 量化方案

采用均匀仿射量化（Uniform Affine Quantization）：

$$
x_q = \text{round}\left(\frac{x_r}{S}\right) + Z
$$

其中 $S$ 为 scale（步长），$Z$ 为零点（zero point）。搜索过程同时优化 $S$ 和 $Z$ 的确定策略（如 per-tensor vs per-channel）。

## 3. 关键公式（LaTeX）

**均匀量化定义**：

$$
x_q = \text{clamp}\left(\left\lfloor \frac{x_r}{S} \right\rceil + Z, q_{\min}, q_{\max}\right)
$$

其中 $S = \frac{\max(x_r) - \min(x_r)}{2^b - 1}$ 为步长，$b$ 为位宽。

**RL 的状态转移**：

$$
s_{t+1} = f(s_t, a_t) = (t+1, \text{layer}_{t+1}\text{.shape}, b_w^{(t)}, b_a^{(t)})
$$

**累计奖励（Q 值学习目标）**：

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \eta \left[R_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)\right]
$$

**搜索目标（多目标优化）**：

$$
\max_{b_w, b_a} \quad \text{Acc}(b_w, b_a) - \lambda \cdot \max\left(0, \frac{LAT(b_w, b_a)}{LAT_{\text{target}}} - 1\right)
$$

其中 $\text{Acc}$ 为量化模型的 Top-1 精度，$LAT$ 为实测延迟，$\lambda$ 为延迟约束惩罚系数。

**层敏感度度量**：

$$
\text{Sensitivity}_l = \frac{\text{Acc}_{\text{FP32}} - \text{Acc}_{l\text{-quantized-to-4bit}}}{\text{Acc}_{\text{FP32}}}
$$

敏感度高的层分配更高位宽，敏感度低的层分配更低位宽。

## 4. 实验结论

- **ImageNet 精度 vs 模型大小**：
  - HAQ 将 ResNet-50 从 4.09 MB（8-bit 统一量化）压缩至 **2.05 MB**（混合精度，部分层低至 4-bit），Top-1 精度从 76.1% 仅降至 75.3%（损失 0.8%）
  - 同模型大小下，HAQ 精度优于统一 4-bit 量化 **2-3 个百分点**
  - MobileNetV2：HAQ 平均位宽 4.6-bit，精度 71.1%（8-bit 为 71.9%），减少了约 40% 模型大小
- **硬件实测加速**：
  - 在 BitFusion（一种灵活的位宽可配置加速器）上，HAQ 混合精度方案相比 8-bit 统一量化有 **1.3-1.8×** 的实际加速和能效提升
  - 在 NVIDIA GTX 1080 Ti（仅支持 8-bit INT8 推理）上，HAQ 将 8-bit 方案的加速比提升到 2-3×（通过降低内存带宽需求）
- **层位宽分配规律**：
  - 第一层和最后一层几乎总是分配 8-bit（对信息编码和解码最敏感）
  - 深度可分离卷积层（如 MobileNetV2 的 DW conv）对量化非常鲁棒，常分配 4-bit 甚至更低
  - 1×1 卷积（Pointwise conv）的低位宽鲁棒性介于 DW conv 和 3×3 conv 之间
- **DQN vs 随机搜索 vs 进化算法**：DQN 搜索效率明显更高，在相同搜索步数内找到的配置精度优于其他方法
- **FLOPs 不等于延迟**：HAQ 明确显示了模型大小/FLOPs 与实际硬件延迟之间的非线性关系——某些层降低位宽减少的 FLOPs 有限但延迟改善显著（因为硬件利用率高），反之亦然

## 5. 工业价值

- **自动化量化部署**：将量化从手工调优变为自动化流程，降低在边缘设备上部署 AI 模型的门槛
- **硬件感知的设计理念**：FLOPs 指导的优化方向与实际硬件效率经常矛盾，HAQ 开创了将真实硬件反馈嵌入搜索环路的范式
- **混合精度推理的产业实践**：
  - NVIDIA TensorRT 支持 INT8 + FP16 混合精度推理
  - Qualcomm SNPE 提供了各层独立量化配置的能力
  - Apple Core ML 在内部使用各层感知的量化策略
- **推动了后续工作**：HAWQ（基于 Hessian 的混合精度量化）、ZeroQ、AdaQuant 等均受 HAQ 启发
- **与 NAS 的融合趋势**：HAQ 展示了 RL 搜索在"配置搜索"（非架构搜索）中的有效性，与 ProxylessNAS、Once-for-All 等工作共同构成了硬件感知自动优化的完整图景

## 6. 与课程 lecture 的关系

- **Lecture 06（Quantization II - Mixed Precision）**：本文是 Lecture 06 的核心论文。Lecture 05（Quantization I）介绍了量化基础知识（均匀量化、非均匀量化、量化感知训练），Lecture 06 在此基础上讨论混合精度——即"不同层可以用不同位宽"的理念。HAQ 是混合精度量化的代表工作，使用 RL 自动搜索各层位宽并融入硬件反馈。Lecture 06 还会比较 HAWQ（基于 Hessian 矩阵的敏感度）、混合精度量化与剪枝的关联等。

## 7. 我应该如何复现

1. **搭建量化框架**：
   ```python
   import torch.quantization as quant

   class MixedPrecisionQuantizer:
       def __init__(self, bitwidths_per_layer):
           self.bitwidths = bitwidths_per_layer

       def quantize_layer(self, layer, w_bit, a_bit):
           # 使用 fake quantization 进行 QAT
           w_scale = layer.weight.abs().max() / (2**(w_bit-1) - 1)
           w_q = torch.round(layer.weight / w_scale).clamp(...)
           ...
   ```

2. **用 PyTorch FX 实现灵活量化**：
   ```python
   import torch.quantization as quant
   # QuantStub/DeQuantStub + prepare_qat + convert
   # 各层独立的 qconfig 实现混合精度
   qconfig_dict = {
       "layer1": quant.QConfig(activation=..., weight=...),
       "layer2": quant.QConfig(activation=..., weight=...),
   }
   ```

3. **简化复现路线**：
   - **Phase 1（量化基础）**：在 CIFAR-10 上对 ResNet-18 实现 per-layer 的统一 8-bit QAT，验证精度恢复
   - **Phase 2（敏感度分析）**：遍历每层从 8-bit 降到 2-bit，绘制每层的"位宽-精度"曲线（层敏感度曲线），验证第一层/最后一层最敏感的规律
   - **Phase 3（RL 搜索）**：用简单 grid search 或随机搜索在 ResNet-18 上搜索混合精度方案，验证优于统一量化
   - **Phase 4（硬件感知）**：使用 `torch.utils.benchmark` 测量各层在不同位宽下的 GPU 延迟，将其作为 RL reward 的一部分

4. **开源参考实现**：
   - HAQ 官方代码：`https://github.com/mit-han-lab/haq`
   - PyTorch 官方 QAT 教程：`https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html`
   - torch2trt（NVIDIA 加速混合精度推理）：`https://github.com/NVIDIA-AI-IOT/torch2trt`

5. **关键超参数**：
   - RL agent：DQN with 2 hidden layers (128-64), replay buffer size=1000, $\epsilon$ decay from 1.0 to 0.1
   - 量化：$b=\{2,4,8\}$（简化搜索空间），per-channel weight quantization, per-tensor activation quantization
   - QAT epochs：各方案 fine-tune 约 5-10 个 epoch（不需要从头训练）
   - 奖励缩放：精度项 + $\exp(-\max(0, \text{latency}/\text{target}-1))$

6. **常见坑**：
   - 量化时不同层的 scale 范围差异巨大，需要对梯度做适当 clipping
   - DQN 搜索中，候选位宽组合的精度评估需要多轮平均（训练随机性导致单次估计方差大）
   - Batch Normalization 的 folding 在混合精度下需要特殊处理（不同层可能有不同的数值精度）
   - 硬件延迟测量需要排除其他进程干扰，建议单独测试
