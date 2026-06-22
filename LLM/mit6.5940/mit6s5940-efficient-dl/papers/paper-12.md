# On-Device Training Under 256KB Memory (MCUNetV3)

> Ji Lin et al., NeurIPS 2022

## 1. 论文解决什么问题

目前 TinyML 领域的所有工作都聚焦在**推理**（inference）——将训练好的模型部署到 MCU 上运行。但在实际场景中，仅做推理远远不够：(1) 边缘设备的数据分布与训练数据不同（domain shift），推理精度会逐渐退化；(2) 用户希望设备能根据个人数据持续适应（personalization）；(3) 隐私法规要求数据不能上传到云端再训练。因此，**在 MCU 级设备上直接进行训练（on-device training）**是刚需。

然而，训练需要的内存远超推理：反向传播需要保存中间激活值（activation memory）用于梯度计算，权重更新需要存储梯度本身，优化器状态（如 Adam 的 momentum/variance）更占用大量内存。在 256KB SRAM 的限制下，甚至 PyTorch 的最小训练脚本都远超内存上限。本文解决了**如何在 <256KB SRAM 的 MCU 上实现完整的神经网络训练**这一极端挑战。

## 2. 核心方法

MCUNetV3 通过一套训练系统（Tiny Training Engine, TTE）和两项关键算法创新，将训练内存从 >1GB 压缩到 <256KB：

### 2.1 量化感知缩放（Quantization-Aware Scaling, QAS）

训练时使用 8-bit 整数表示权重和激活值，但 8-bit 整数表示范围有限（$[-128, 127]$），训练过程中梯度通常非常小（$10^{-5}$ 量级），直接量化会导致梯度变为零，训练无法进行。

**QAS 的解决方案**：
- 在前向传播中对权重和激活值做 8-bit 量化（节省内存），但在反向传播中将梯度乘以一个精心设计的缩放因子（scaling factor），使梯度保持在 8-bit 可表示的范围
- 反向传播的梯度也使用 int8 存储，但使用不同 scale 策略——动态调整梯度的 scale 使其压缩到 $[-128, 127]$
- 权重的更新使用 int16 累加器，累积多步梯度后再取整写入 int8 权重

具体来说，QAS 维护以下量化-反量化流程：

$$
W_{\text{int8}} = \text{round}(W_{\text{fp32}} / s_W), \quad
g_{\text{int8}} = \text{round}(g_{\text{fp32}} / s_g)
$$

其中 $s_W$ 和 $s_g$ 为权重和梯度的量化步长，且 $s_g$ 在每个 training step 动态计算以确保 $g_{\text{fp32}}$ 范围被正确覆盖。

### 2.2 稀疏更新（Sparse Update）

完整训练需要在每次迭代中更新所有层的所有权重，但 MCU 上的 Flash 写入速度慢（每次写入都需要擦除和重编程整个扇区），全量更新不可行。

**稀疏更新的核心思想**：
- 并非所有层在所有训练迭代中都需要更新——某些层（特别是接近分类头的层）对微调更重要
- 每次迭代仅选择性地更新**部分层**或**部分权重**
- 选择策略基于梯度的 L2 范数：梯度模长大的层被优先更新（表明该层在当前 batch 上尚未收敛）

具体实现：
- 维护每层梯度的 running average（EMA）：$\bar{g}_l = \beta \bar{g}_l + (1-\beta) \|g_l\|_2$
- 每次迭代选择 $\bar{g}_l$ 最大的 top-K 层进行更新
- 被选中的层才做 Flash 写入，其余层保持权重不变

### 2.3 Tiny Training Engine (TTE)

将 QAS 和 Sparse Update 集成到 TinyEngine（推理引擎）中，扩展出一个完整的训练系统：
- **内存管理**：activation rematerialization（计算代替存储），in-place gradient computation
- **代码生成**：针对每种层类型生成融合了前向+反向+梯度缩放的 kernel
- **调度优化**：重新排列算子计算顺序以最小化峰值内存

## 3. 关键公式（LaTeX）

**QAS 的前向量化**：

$$
\hat{W} = s_W \cdot \text{clamp}\left(\text{round}(W / s_W), -128, 127\right)
$$

其中 $s_W = \frac{\max(|W|)}{127}$（symmetric quantization）。

**QAS 的反向量化梯度**：

$$
\hat{g} = s_g \cdot \text{round}(g / s_g), \quad s_g = \frac{\max(|g|)}{127}
$$

**稀疏更新选择机制**：

$$
\text{Update}_l^{(t)} = \begin{cases}
1, & \|\bar{g}_l^{(t)}\|_2 \geq \tau_t \\
0, & \text{otherwise}
\end{cases}
$$

其中 $\tau_t$ 为第 $t$ 步的阈值（保证总更新量在 Flash 写入预算内）。

**梯度 EMA**：

$$
\bar{g}_l^{(t)} = \beta \cdot \bar{g}_l^{(t-1)} + (1-\beta) \cdot \|g_l^{(t)}\|_2
$$

**内存统计（核心贡献）**：

$$
M_{\text{total}} = M_{\text{weight}} + M_{\text{activation}} + M_{\text{gradient}} + M_{\text{optimizer}} < 256 \text{KB}
$$

MCUNetV3 实现了 $M_{\text{weight}} \approx 50\text{KB}$, $M_{\text{activation}} \approx 80\text{KB}$, $M_{\text{gradient}} \approx 60\text{KB}$, $M_{\text{optimizer}} \approx 50\text{KB}$，总计约 240KB。

## 4. 实验结论

- **内存使用**：
  - MCUNetV3 在 MCU 上训练仅需 **~200KB** SRAM（预留部分给运行时栈和通信缓冲），而 PyTorch 在相同模型上训练需要 >1GB
  - 相比 naive 8-bit 训练尝试，QAS 单独贡献了约 **30% 的梯度精度提升**（避免梯度下溢导致训练停滞）
  - Sparse Update 将 Flash 写入量减少 **3-5×**，使训练在 Flash 寿命允许范围内
- **微调效果**：
  - 在 domain shift 场景下（比如从 ImageNet 到 IoT 传感器数据），MCUNetV3 微调 10 个 epoch 后精度恢复 **5-15 个百分点**（使用 QAS + Sparse Update）
  - 仅用 100 个 labeled 样本微调（few-shot 场景），精度提升 5-10 个百分点
  - 与全精度微调（在服务器上）对比，8-bit 训练精度差距通常在 **1-2% 以内**
- **个性化效果**：
  - 在 Wake Word Detection（唤醒词检测）任务上，MCUNetV3 使设备能根据特定用户的语音模式进行适配，FNR（漏检率）降低 20-30%
  - 在 Visual Wake Words（视觉唤醒词）任务上，个性化微调使精度提高了 5-10%
- **消融实验**：
  - 仅用 QAS（不含 Sparse Update）：精度恢复效果最好，但 Flash 写入量过大，无法在低端 MCU 上持续运行
  - 仅用 Sparse Update（不含 QAS）：Flash 写入 OK，但梯度下溢导致训练效果差
  - QAS + Sparse Update：二者互补，是最优组合

## 5. 工业价值

- **打开 TinyML 的下一个时代**：从"仅推理"走向"推理+训练"，使边缘 AI 设备具备了持续学习和自适应的能力
- **隐私保护的 AI**：on-device training 使得用户数据完全保留在本地，满足 GDPR、CCPA 等隐私法规的"数据不出设备"要求
- **降低标注成本**：设备可以根据用户交互隐式获得监督信号（如用户纠正预测），无需人工标注
- **实际场景应用**：
  - **智能家居**：智能音箱根据家庭成员的语音模式持续优化唤醒词检测
  - **可穿戴设备**：智能手表根据佩戴者的运动模式调整步数计数和心率预测模型
  - **工业传感器**：预测性维护模型根据具体机器的振动模式自适应调整
- **后续工作基础**：推动了 TinyTL（Tiny Transfer Learning）、POET（Privately On-device Edge Training）等工作

## 6. 与课程 lecture 的关系

- **Lecture 21（On-Device Training）**：MCUNetV3 是 Lecture 21 的核心论文。课程全面讲解了 on-device training 的三个核心挑战——内存、计算、存储（Flash 写入）——以及 MCUNetV3 如何通过 QAS 和 Sparse Update 系统性地解决这些问题。Lecture 21 也会介绍相关方法如 TinyTL（迁移学习中的快速适配）和 POET（隐私保护训练），将 MCUNet 系列的故事收束到完整的"在 MCU 上推理 + 训练"闭环。

## 7. 我应该如何复现

1. **官方开源仓库**：
   - MCUNetV3（包含 Tiny Training Engine）：`https://github.com/mit-han-lab/mcunet`
   - TinyEngine 推理+训练库（C/C++ 实现）：同一仓库的 `tinyengine/` 目录

2. **核心代码结构**（Python 训练框架模拟）：
   ```python
   class QuantizationAwareTraining(nn.Module):
       def forward(self, x):
           # QAS: quantize weights then forward
           w_q = self.quantize(self.weight)
           out = F.linear(self.quantize(x), w_q)
           return out

       def backward(self, grad_output):
           # QAS: scale gradient to int8 range
           grad_scaled = self.scale_grad(grad_output, 'int8')
           grad_weight = grad_scaled @ self.activation.T
           return grad_input, grad_weight

   class SparseUpdate:
       def __init__(self, update_ratio=0.3):
           self.update_ratio = update_ratio
           self.grad_ema = defaultdict(float)

       def select_layers(self, model):
           # 选择梯度模长最大的 top-K 层
           norms = {l: self.grad_ema[l] for l in model.layers}
           top_k = sorted(norms, key=norms.get, reverse=True)
           return top_k[:int(len(top_k) * self.update_ratio)]
   ```

3. **简化复现路线**：
   - **Phase 1（理解 QAS）**：在 CIFAR-10 上用小型 CNN 实现 int8 QAT（量化感知训练），对比 fp32、int8 w/o QAS、int8 w/ QAS 的训练曲线差异
   - **Phase 2（内存分析）**：用 `torch.cuda.memory_stats()` 或 `memory_profiler` 分析训练过程中权重、激活、梯度、优化器状态各自的内存占用
   - **Phase 3（稀疏更新）**：在 Phase 1 的 int8 QAT 基础上添加梯度的 EMA 追踪和稀疏更新策略，验证 Flash 写入量减少的效果
   - **Phase 4（MCU 部署）**：将训练好的 C 代码（通过 TinyEngine 生成）烧录到 STM32F746 开发板，在设备上实际运行训练循环

4. **硬件要求**：
   - 开发板：**STM32F746G-DISCO**（320KB SRAM）或 **STM32H743**（1MB SRAM）
   - 小数据集：建议先在 **CIFAR-10** (32×32) 或 **Speech Commands** 上验证，再扩展到 ImageNet
   - 编译器链：`arm-none-eabi-gcc` + CMSIS-NN 库

5. **关键超参数**：
   - 量化位宽：权重 int8, 激活 int8, 梯度 int8, 优化器 int16 (accumulator)
   - Sparse Update: update_ratio=0.2-0.5（根据 Flash 写入预算调整）
   - 学习率：int8 training lr 通常需要比 fp32 低 1.5-2×
   - Gradient EMA decay: $\beta=0.9$
   - 使用 SGD 而非 Adam（Adam 的 momentum/variance 额外占用 2× 参数量内存）

6. **常见坑**：
   - int8 梯度量化中，scale 计算必须在每个 step 重新进行——batch 之间的梯度分布可能剧烈变化
   - Flash 写入速度非常慢（~100μs/word），训练时间可能在数十秒到数分钟量级
   - 梯度 EMA 的更新阈值选择至关重要——过低导致几乎所有层被更新（稀疏更新退化），过高导致训练停滞
   - MCU 上的伪随机数生成（用于 sparse update 的随机探索）需要考虑功耗和确定性
