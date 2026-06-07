# 第六讲：量化 II — PTQ vs QAT、二值/三值量化与混合精度

## 1. 本讲核心问题

在上一讲介绍了量化基础之后，本讲深入探讨**如何将量化技术真正落地**。核心问题是：

- 训练好的浮点模型如何直接量化？**Post-Training Quantization (PTQ)** 的校准方法有哪些？为什么 PTQ 有时候会失败？
- 如果 PTQ 精度不够，**Quantization-Aware Training (QAT)** 如何通过模拟量化效应来挽救精度？
- 极限场景下，能否将权重压缩到 1 bit（二值）或 2 bit（三值）？**BinaryConnect / XNOR-Net** 如何工作？
- 不同层的敏感度不同，**混合精度量化**（Mixed Precision）如何为不同层分配不同位宽？
- 如何让量化参数（如 step size）变成可学习的？**LSQ (Learned Step Size Quantization)** 是怎么做到的？
- **自动混合精度 (AMP)** 在训练中如何加速？

## 2. 通俗解释

想象你是一个摄影师，手里有一张超高分辨率的 RAW 照片（FP32 模型），要把它印在不同大小的杂志上：

- **PTQ** 就是直接用 Photoshop 把 RAW 转成 JPEG，不再做任何调整。对于大多数照片，这足够好。但如果你要印到一个很小的邮票上（极低位宽），直接缩放可能会丢失重要细节（比如人脸看不清了），这就是 PTQ 失败的情况。
- **QAT** 就是在处理 RAW 照片时，你先在软件里模拟"这张照片印在邮票上会是什么效果"，然后根据这个模拟效果去调整原始 RAW 的处理参数，使得最终邮票上的效果尽可能好。QAT 在训练时就加入了量化的影响，所以模型"知道"自己会被量化，学会补偿。
- **二值量化** 就像是把照片变成纯粹的黑白两色——完全没有灰度。信息损失极大，但如果你只是一个"安全监控摄像头"，只需要判断有没有人（不需要看清脸），那黑白就够了，而且传输速度极快。
- **混合精度** 就像是这样：照片里的人脸部分保留更多细节（高精度），背景的蓝天白云可以压缩得更狠（低精度）。不同区域的信息重要度不同。
- **LSQ** 就像是让你的相机自动学习：曝光补偿调到多少最合适？不是手动设置的，而是通过拍摄大量照片，自动学出来的最佳参数。

## 3. 关键公式

### PTQ 校准方法

**MinMax 校准：**
对于给定位宽 b，量化范围由数据的最小最大值决定：

$$s = \frac{\max(|x_{max}|, |x_{min}|)}{2^{b-1} - 1}$$

$$x_q = \text{clamp}(\text{round}(x/s), -2^{b-1}, 2^{b-1} - 1)$$

**MSE 校准（更优）：**
寻找最优的 scale s 使得量化误差的均方误差最小：

$$s^* = \arg\min_s \mathbb{E}\left[\|x - s \cdot \text{clamp}(\text{round}(x/s), -2^{b-1}, 2^{b-1} - 1)\|^2\right]$$

**Percentile 校准：**
不取绝对 min/max，而是取第 p 个百分位（如 99.99%），把离群值截断：

$$s = \frac{\text{percentile}(|x|, p)}{2^{b-1} - 1}$$

### QAT 中的 FakeQuantize + STE

前向（模拟量化）：
$$x_{fake} = s \cdot \text{clamp}\left(\text{round}\left(\frac{x}{s}\right), -2^{b-1}, 2^{b-1} - 1\right)$$

反向（Straight-Through Estimator）：
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial x_{fake}} \cdot \mathbf{1}_{|x/s| \leq 2^{b-1}}$$

### 二值量化 (BinaryConnect)

前向用二值权重：
$$w_b = \text{sign}(w) = \begin{cases} +1, & w \geq 0 \\ -1, & w < 0 \end{cases}$$

反向 STE：
$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial w_b} \cdot \mathbf{1}_{|w| \leq 1}$$

### LSQ 的 Learned Step Size

将 scale s 作为可学习参数，梯度为：

$$\frac{\partial \hat{x}}{\partial s} = \begin{cases} -\frac{x}{s} + \text{round}\left(\frac{x}{s}\right), & |x/s| \leq 2^{b-1} \\ -\text{sign}(x) \cdot (2^{b-1} - 1), & \text{otherwise} \end{cases}$$

## 4. 公式背后的直觉

- **MinMax 问题**：直接取 min/max 虽然在数学上"完整覆盖"了所有值，但如果数据中有几个极端离群值（比如一个激活值是 100，而其他 99.9% 的值都在 [-1, 1]），那 scale 就会变得非常大，导致所有正常值都被量化到 0 附近，信息丢失。这就是为什么 MSE 和 Percentile 更优——它们允许截断少量极端值，换取绝大多数值的更高精度。

- **STE 的直觉**：`round()` 函数的梯度在几乎所有地方都是 0，无法训练。STE 假装 `round()` 就是恒等函数（梯度为 1），但对于超出量化范围的值，梯度为 0（因为 clamp 截断了）。这个"假装"虽然粗暴，但在实践中非常有效——就像你告诉学生"做错了也没关系，先假装你做得对，然后慢慢调整"。

- **LSQ 的洞见**：传统量化中，scale 是手动计算的固定值。但如果 scale 也是可学习的，模型就可以在训练过程中自然地找到最佳精度平衡。LSQ 的关键贡献是推导出 scale 的正确梯度公式，特别是处理了 round 函数的阶梯特性。

- **二值量化的威力**：用 ±1 替换浮点权重后，乘法变成了纯符号翻转（加/减法），这在实际硬件上快得惊人。想象你需要计算 1000 万次乘法——如果都是 ±1，硬件可以用一个 XOR 门就完成（比特操作），而不需要 32 位浮点乘法器。

## 5. 工业界用途

- **NVIDIA TensorRT**：工业级 PTQ 工具，支持 INT8 校准。内部使用 KL 散度校准（类似 Percentile 方法），在 BERT、ResNet-50 上实现 < 0.5% 精度损失。
- **PyTorch 的 `torch.quantization`**：支持 PTQ (`torch.quantization.quantize_dynamic`) 和 QAT (`torch.quantization.prepare_qat`)。PyTorch 2.0 的 `torch.compile` 也引入了自动量化支持。
- **TensorFlow Lite**：移动端部署标配 INT8 量化，支持训练后整数量化（weight + activation 都是 INT8）。
- **Qualcomm AI Engine**：手机上做 INT4 推理，支持 LSQ 风格的 learned quantization。
- **Hugging Face Optimum-Intel**：对 Transformer 模型做 INT8/INT4 量化，内置 SmoothQuant 等高级技术。
- **自动混合精度 (AMP)**：NVIDIA Apex 和 PyTorch 原生 `torch.cuda.amp`，在训练中自动将部分算子转为 FP16，部分保留 FP32（如 loss、batch norm），训练速度提升 2-3x 而精度几乎不变。
- **Binary Neural Networks**：在 FPGA 和超低功耗芯片上有实际应用，如 GAP8（GreenWaves）芯片原生支持二值网络推理。

## 6. PyTorch 实现思路

### PTQ with PyTorch

```python
import torch
import torch.quantization as quant

# 定义模型
model = torchvision.models.resnet18(pretrained=True)
model.eval()

# 设置量化配置：后端为 fbgemm（x86）或 qnnpack（ARM）
model.qconfig = quant.get_default_qconfig('fbgemm')

# 融合 Conv + BN + ReLU
model = quant.fuse_modules(model, [['conv1', 'bn1', 'relu']])

# 插入 Observer 并校准
model_prepared = quant.prepare(model)

# 使用校准数据集运行若干 batch，Observer 收集统计数据
for data, _ in calibration_loader:
    model_prepared(data)

# 转换为量化模型
model_quantized = quant.convert(model_prepared)
```

### QAT 实现思路

```python
# QAT 的关键：在训练循环中加入 FakeQuantize 模块
model.train()
model.qconfig = quant.get_default_qat_qconfig('fbgemm')
model_prepared = quant.prepare_qat(model)

# 正常训练，FakeQuantize 在前向时模拟量化，
# 反向时使用 STE 传播梯度
for epoch in range(epochs):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model_prepared(data)
        loss = criterion(output, target)
        loss.backward()  # FakeQuantize 使用 STE
        optimizer.step()

# 训练完成后转换
model_quantized = quant.convert(model_prepared.eval())
```

### LSQ 自定义实现（简化版）

```python
class LSQQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, nbits):
        # 量化
        x_int = torch.clamp(torch.round(x / scale), -2**(nbits-1), 2**(nbits-1)-1)
        ctx.save_for_backward(x, scale)
        ctx.nbits = nbits
        return x_int * scale  # 反量化

    @staticmethod
    def backward(ctx, grad_output):
        x, scale = ctx.saved_tensors
        nbits = ctx.nbits
        # STE for x
        grad_x = grad_output.clone()
        grad_x[(x / scale).abs() > 2**(nbits-1)] = 0
        # Gradient for scale (simplified)
        x_div_s = x / scale
        grad_s = (x_div_s - x_div_s.round()).clamp(-2**(nbits-1), 2**(nbits-1)-1)
        grad_s = (grad_s * grad_output).sum()
        return grad_x, grad_s, None
```

### AMP 训练

```python
# 使用 torch.cuda.amp 自动混合精度训练
scaler = torch.cuda.amp.GradScaler()

for data, target in train_loader:
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        output = model(data)
        loss = criterion(output, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## 7. TinyML / Edge AI 部署意义

- **PTQ** 是 Edge 部署的第一选择：不需要重新训练，校准只需少量数据（几百张图就够了），特别适合已经在云上训练好的模型直接压缩到端侧。
- **QAT** 是 PTQ 失败后的备选方案：当 PTQ 精度下降超过 2-3% 时，QAT 通常能挽回大部分精度。对 INT4 及以下位宽，QAT 几乎是必须的。
- **二值/三值网络** 在超低功耗 MCU 上意义重大：MCU 通常没有乘法器或乘法器很慢，二值网络把乘法全变成位操作，推理速度可提升 10-50x。
- **混合精度** 对应 TinyML 的"好钢用在刀刃上"：MCU 的内存非常有限，全模型 8 bit 可能装不下，但如果关键层用 8 bit、不敏感层用 4 bit，就能在给定内存预算下最大化精度。
- **LSQ** 在手机 NPU 上被广泛应用：手机芯片（如 Apple Neural Engine、Qualcomm Hexagon）支持可变位宽推理，LSQ 训练的 step size 可以直接映射到硬件量化参数。
- **AMP** 训练使得在边缘设备上做 on-device fine-tuning 成为可能：梯度用 FP16 既省内存又加速，在 Jetson Nano 等边缘设备上非常有价值。

## 8. 常见误区

1. **"PTQ 是无损的"**：错误。PTQ 总是有精度损失，只是大多数情况下损失很小（< 0.5%）。对于某些敏感模型（如小目标检测、超分辨率），PTQ 可能损失 3-5%。
2. **"量化到更低 bit 总是更好"**：不一定。虽然 4 bit 比 8 bit 省一半内存，但如果精度损失过大（比如 ImageNet top-1 掉 5%），那就不如用更小的 8 bit 模型。
3. **"STE 只是一个 hack，没有理论依据"**：虽然 STE 确实是一个近似，但近年研究表明 STE 实际上在做隐式的梯度校正，并且在凸优化问题中可证明收敛。
4. **"混合精度就是每层一个 bit 宽度"**：混合精度可以是 channel-wise（同一层不同通道不同位宽）、layer-wise、甚至是 filter-wise。粒度越细，优化越好，但搜索空间也越大。
5. **"AMP 和量化是一回事"**：不同。AMP 是训练加速技术（FP32 → FP16），量化主要是推理加速技术（FP32 → INT8/INT4）。AMP 不改变模型存储格式。
6. **"QAT 比 PTQ 总是更好"**：QAT 精度确实更好，但代价是需要重新训练。对于很多场景，PTQ 的精度损失已经可以接受，没必要花额外的训练成本。
7. **"校准数据集越大越好"**：校准数据集只需要能代表推理数据的分布即可。通常 100-1000 张图就足够。太大的校准集不会带来额外收益。

## 9. 面试问题

**Q1：PTQ 中，为什么 MSE 校准通常比 MinMax 校准更好？在什么情况下 MinMax 反而更好？**

MSE 校准通过最小化 L2 量化误差来寻找最优 scale，允许截断少量极端值以换取绝大多数值的更高精度。这在数据分布有"长尾"时特别有效。MinMax 更好的一种情况是：应用中要求严格的 range 保真度，不能有任何截断——例如音频信号处理中，clip 会导致可听失真。

**Q2：LSQ 和传统的 QAT（固定 scale）相比，核心区别是什么？为什么 LSQ 能取得更好的精度？**

传统 QAT 用 calibration 算出一个固定 scale，训练中不更新。LSQ 把 scale 变成一个可学习参数，模型在训练中同步优化权重和量化步长，最终收敛到更精确的"权重-量化"联合最优解。特别是在极低位宽（2-4 bit）下，合适的 scale 对精度影响巨大，LSQ 的优势尤为明显。

**Q3：BinaryConnect 前向用 sign(w)，反向用 STE，这样做为什么可行？sign 函数梯度为零，不会阻止训练吗？**

sign 函数确实是不可微的，其真正梯度处处为 0（除了 0 处无定义）。STE 的核心思想是"假装 sign 的梯度为 1"，从而让梯度"穿过"这个不可微操作。这在实践中有效的原因是：虽然每一步的梯度近似很粗糙，但在大批量、多步迭代的平均下，权重更新的方向大致正确。可以类比为"蒙着眼睛下山"——虽然看不清每一步的具体地形，但梯度的大方向引导你往下走。

## 10. 本讲总结

本讲是量化的进阶篇，从校准方法到训练策略，系统覆盖了从 PTQ 到 QAT 的完整技术栈：

- **PTQ 的核心是校准**：MinMax 简单但有长尾问题，MSE 和 Percentile 在实践中更优。校准数据集的质量和代表性是关键。
- **QAT 通过 FakeQuantize + STE 挽救 PTQ 无法企及的精度**，是极低位宽（≤4 bit）的必须步骤。
- **二值/三值网络**将极端量化推向极限，乘法变位操作，在 FPGA 和超低功耗 MCU 上有独特价值。
- **混合精度量化**承认"不是所有层都需要相同精度"，通过敏感度分析实现精度-效率的最优平衡。
- **LSQ**代表了量化技术的进化方向：让量化参数"可学习"，实现端到端的量化优化。
- **AMP**作为训练加速的工程实践，已经深度融入 PyTorch 训练流程。

一句话总结：量化不是简单的"除以一个数然后取整"，而是一个从校准、训练到部署的全链路优化问题——每一步的选择都直接影响最终精度。LSQ 的出现标志着量化从"手动工程"进入了"自动学习"时代。
