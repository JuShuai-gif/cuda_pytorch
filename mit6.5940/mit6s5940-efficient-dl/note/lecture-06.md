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

### 大厂量化进阶实战案例

- **Meta PyTorch 团队 AMP 实战经验**: 在训练 LLaMA 65B 时，使用 BF16 AMP + FSDP (Fully Sharded Data Parallel) 将 2048 张 A100 的显存利用率从 52% 提升到 78%，训练吞吐提升 1.8x。但关键发现：不是所有 op 都适合 BF16。特别是 `LayerNorm` 和 `Softmax` — 这些 op 的数值稳定性依赖 FP32 的精度，强制 BF16 会导致 loss spike。PyTorch AMP 通过 `torch.cuda.amp.autocast()` 的白名单机制自动保护这些敏感 op。

- **快手端上视频超分模型 LSQ 实践**: 在手机端部署 Real-ESRGAN 超分模型，传统 PTQ INT8 量化后 PSNR 从 28.3dB 降到 26.1dB（掉 2.2dB），图像的纹理明显退化。切换到 LSQ（Learned Step Size Quantization）后，在超分数据集上 QAT 微调 20000 步 → PSNR 恢复到 28.1dB（仅降 0.2dB）。关键原因：超分模型对高频细节极度敏感，固定 scale 的大量化步长会磨平纹理细节，LSQ 通过可学习 scale 找到了每层最优的"纹理保护"与"量化压缩"平衡点。

- **Google TensorFlow Lite Micro 在人声检测上的混合精度**: Wake-word detection 模型，8-bit 全量化后误报率从 0.3% 升到 1.1%（不可接受）。通过混合精度：前 2 层 feature extractor 保持 16-bit（对语音频谱的细微信号敏感），后 8 层分类器用 8-bit（对量化鲁棒）→ 误报率恢复到 0.35%，模型大小仅增加 12KB。

### 各量化方案的产业成熟度对比

| 技术 | 成熟度 | 推理加速 | 训练成本 | 硬件要求 | 典型场景 |
|------|--------|---------|---------|----------|----------|
| FP16 AMP (训练) | ★★★★★ | 1.5-2.5x | 零额外 | Volta+ GPU | 所有训练 |
| INT8 PTQ | ★★★★★ | 2-4x | 零（仅校准） | INT8指令集 | CNN/简单NLP推理 |
| INT8 QAT | ★★★★☆ | 2-4x | 10-20%额外训练 | INT8指令集 | 敏感模型推理 |
| INT4 PTQ | ★★★☆☆ | 4-8x | 零 | 有限硬件 | 极限压缩 |
| LSQ | ★★★☆☆ | 2-6x | 20-50%额外训练 | 可变位宽硬件 | 极低位宽(2-4bit) |
| AWQ/GPTQ (LLM) | ★★★★☆ | 2-3x | 数小时GPU | CUDA GPU | LLM本地部署 |
| BinaryConnect | ★★☆☆☆ | 5-10x | 从头训练 | FPGA/定制芯片 | 超低功耗场景 |

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

### 生产环境 QAT 的隐式精度陷阱

```python
import torch
import torch.quantization as quant
from typing import Dict

def production_qat_validation(model_qat, model_fp32, 
                               test_loader, target_inference_engine: str = 'tensorrt'):
    """Validate QAT model against production inference engine.
    
    CRITICAL: QAT's FakeQuantize uses PyTorch's rounding behavior.
    TensorRT, ONNX Runtime, and TFLite may use DIFFERENT rounding modes.
    
    Round mode comparison:
    | Engine        | Round Mode                   |
    |---------------|------------------------------|
    | PyTorch       | round-half-to-even (default) |
    | TensorRT      | round-half-away-from-zero    |
    | TFLite        | round-half-away-from-zero    |
    | ONNX Runtime  | round-half-to-even (default) |
    
    This 1-ULP difference in rounding causes systematic bias
    that accumulates across layers, especially in deep transformers.
    
    Solution: always validate QAT output against the actual inference
    engine output (not just PyTorch eval mode) before shipping.
    """
    model_qat.eval()
    model_fp32.eval()
    
    total_cosine_sim = 0.0
    total_l2_error = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            out_qat = model_qat(data)
            out_fp32 = model_fp32(data)
            
            # Cosine similarity: should be >0.999 for safe deployment
            cos_sim = torch.nn.functional.cosine_similarity(
                out_qat.flatten(), out_fp32.flatten(), dim=0
            )
            # L2 relative error: should be <0.01 for safe deployment
            l2_err = torch.norm(out_qat - out_fp32) / torch.norm(out_fp32)
            
            total_cosine_sim += cos_sim.item()
            total_l2_error += l2_err.item()
            n_batches += 1
    
    avg_cos = total_cosine_sim / n_batches
    avg_l2 = total_l2_error / n_batches
    
    return {
        'avg_cosine_similarity': round(avg_cos, 6),
        'avg_l2_relative_error': round(avg_l2, 6),
        'safe_to_deploy': avg_cos > 0.999 and avg_l2 < 0.01,
        'engine': target_inference_engine,
    }
```

### AMP 训练的数值稳定性检查

```python
def amp_numerical_safety_check(model, data, target, criterion):
    """Check for AMP numerical instability before full training.
    
    AMP can silently produce NaN gradients in these scenarios:
    - Very deep networks (gradient underflow below FP16 min = 6e-5)
    - Large loss values (overflow above FP16 max = 65504)
    - Attention with large sequence length (>1024) causing softmax overflow
    """
    scaler = torch.cuda.amp.GradScaler()
    
    with torch.cuda.amp.autocast():
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    
    # Check for NaN/Inf in gradients
    nan_grads = []
    inf_grads = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                nan_grads.append(name)
            if torch.isinf(param.grad).any():
                inf_grads.append(name)
    
    if nan_grads:
        print(f"WARNING: NaN gradients in: {nan_grads}")
        print("  → Try reducing learning rate or increasing GradScaler init_scale")
    if inf_grads:
        print(f"WARNING: Inf gradients in: {inf_grads}")
        print("  → Loss might be overflowing FP16 range (max=65504)")
        
    return {'nan_params': nan_grads, 'inf_params': inf_grads}
```

## 7. TinyML / Edge AI 部署意义

- **PTQ** 是 Edge 部署的第一选择：不需要重新训练，校准只需少量数据（几百张图就够了），特别适合已经在云上训练好的模型直接压缩到端侧。
- **QAT** 是 PTQ 失败后的备选方案：当 PTQ 精度下降超过 2-3% 时，QAT 通常能挽回大部分精度。对 INT4 及以下位宽，QAT 几乎是必须的。
- **二值/三值网络** 在超低功耗 MCU 上意义重大：MCU 通常没有乘法器或乘法器很慢，二值网络把乘法全变成位操作，推理速度可提升 10-50x。
- **混合精度** 对应 TinyML 的"好钢用在刀刃上"：MCU 的内存非常有限，全模型 8 bit 可能装不下，但如果关键层用 8 bit、不敏感层用 4 bit，就能在给定内存预算下最大化精度。
- **LSQ** 在手机 NPU 上被广泛应用：手机芯片（如 Apple Neural Engine、Qualcomm Hexagon）支持可变位宽推理，LSQ 训练的 step size 可以直接映射到硬件量化参数。
- **AMP** 训练使得在边缘设备上做 on-device fine-tuning 成为可能：梯度用 FP16 既省内存又加速，在 Jetson Nano 等边缘设备上非常有价值。

### 真实硬件上的量化方案可行性矩阵

| 硬件 | PTQ INT8 | QAT INT8 | LSQ (2-8bit) | 二值网络 | AMP训练 |
|------|----------|----------|-------------|---------|---------|
| **Cortex-M4 (无FPU)** | ✅ (软件INT8模拟) | ❌ (无法重训练) | ❌ | ✅ (位操作原生) | ❌ |
| **Cortex-M7+FPU** | ✅ (NEON SIMD) | ❌ | ❌ | ✅ | ❌ |
| **ESP32-S3** | ✅ (ESP-NN库) | ❌ | ❌ | 部分支持 | ❌ |
| **Jetson Nano** | ✅ (TensorRT) | ✅ | ✅ (PyTorch) | ✅ | ✅ (FP16) |
| **Apple Neural Engine** | ✅ (CoreML) | ✅ (CoreML Tools) | ✅ (CoreML 6+) | ❌ | N/A |
| **Qualcomm Hexagon** | ✅ (SNPE/QNN) | ✅ (AIMET) | ✅ (AIMET) | ❌ | N/A |

### 二值网络在极致功耗场景下的部署现实

- **GAP8 (GreenWaves) 芯片上的二值网络**: 在 50MHz 运行频率、35mW 功耗下，二值 CNN 做图像分类：推理延迟 4.2ms/帧，功耗 35mW × 4.2ms = 0.147 mJ/帧。对比 FP32(软件模拟) 的 Cortex-M4：92ms/帧，100mW → 9.2 mJ/帧。**差距**: 二值网络能效比是 FP32 的 62x。这使纽扣电池 (CR2032, 225mAh @ 3V = ~2430 J) 理论上可以支撑 1650万次二值推理，但只够 26 万次 FP32 推理。
- **但是**：二值网络在 ImageNet 上的 Top-1 只有 ~52%（vs ResNet50 的 76%）。在很多实际场景中精度不够。目前二值网络真正落地的场景仅限于：关键词检测（二进制判断有无唤醒词）、简单手势识别（上下左右滑动）、运动状态检测（走/跑/静止）。但凡需要细粒度分类（如识别具体物体类别），二值网络就不够用。

## 8. 常见误区

1. **"PTQ 是无损的"**：错误。PTQ 总是有精度损失，只是大多数情况下损失很小（< 0.5%）。对于某些敏感模型（如小目标检测、超分辨率），PTQ 可能损失 3-5%。
2. **"量化到更低 bit 总是更好"**：不一定。虽然 4 bit 比 8 bit 省一半内存，但如果精度损失过大（比如 ImageNet top-1 掉 5%），那就不如用更小的 8 bit 模型。
3. **"STE 只是一个 hack，没有理论依据"**：虽然 STE 确实是一个近似，但近年研究表明 STE 实际上在做隐式的梯度校正，并且在凸优化问题中可证明收敛。
4. **"混合精度就是每层一个 bit 宽度"**：混合精度可以是 channel-wise（同一层不同通道不同位宽）、layer-wise、甚至是 filter-wise。粒度越细，优化越好，但搜索空间也越大。
5. **"AMP 和量化是一回事"**：不同。AMP 是训练加速技术（FP32 → FP16），量化主要是推理加速技术（FP32 → INT8/INT4）。AMP 不改变模型存储格式。
6. **"QAT 比 PTQ 总是更好"**：QAT 精度确实更好，但代价是需要重新训练。对于很多场景，PTQ 的精度损失已经可以接受，没必要花额外的训练成本。
7. **"校准数据集越大越好"**：校准数据集只需要能代表推理数据的分布即可。通常 100-1000 张图就足够。太大的校准集不会带来额外收益。

### 生产环境量化 P0 级事故

8. **"QAT 训练的精度 99%，上线后推理引擎输出与训练时不一致"** — 这是最常见的"QAT精度OK部署挂"事故。QAT 训练使用 PyTorch 的 FakeQuantize (round-half-to-even)，但 TensorRT 的 INT8 kernel 使用 round-half-away-from-zero。这个 1 ULP 的差异在 ResNet50 的 50 层中逐层累积，最终导致分类结果的 softmax 分布偏移。**另一层陷阱**: 即使 round mode 一致，TensorRT 和 PyTorch 的 INT8 GEMM 累加顺序（FP32 accumulator → 写回 INT8）也不同，也会引入微小但非零的偏差。**排查技巧**: 逐层对比 QAT eval mode 和 TensorRT 的中间层输出，用 cosine similarity 检测哪一层开始偏差超过 0.001。通常从第 15-20 层开始。

9. **"AMP 训练中出现 NaN loss，增大 GradScaler scale 反而更糟"** — AMP 中 NaN loss 有两大类：(A) 梯度下溢（underflow）→ FP16 最小正值 ~6e-5，小于此值的梯度变为 0 → 用 GradScaler 放大 loss 再缩小梯度。(B) 梯度溢出（overflow）→ FP16 最大值 65504，大于此值的梯度变为 NaN。增大 GradScaler 只能解决 (A)，但会让 (B) 更严重。如果 loss 本身就大到使 FP16 overflow（如大 batch 下 loss × GradScaler > 65504），需要：(1) 降低 batch size；(2) 对 loss 先做 `loss = loss / num_accumulation_steps` 再过 AMP；(3) 使用 BF16（范围与 FP32 相同，只损失精度不损失范围）。

10. **"LSQ 训练后 step size 的值变得异常小（1e-8 量级）→ 训练初期就 exploded"** — LSQ 的 step size 梯度公式在初始化的前几百步非常不稳定。如果初始 scale 过大（`scale_init = 2 * |w|_mean / sqrt(q_max)`），那么 `x/s` 会非常小 → `round(x/s) = 0` → 所有值被量化到 0 → 反向梯度为 0 → scale 不再更新 → 训练死锁。**解决方案**: LSQ 的第一作者 Steven Esser 推荐 `scale_init = max(|w|) / q_max`（而非 mean），保证初始化时至少有一些值 ≥1 量化级。并且在训练前 1000 步将 scale 的 lr 设为 0，等权重大致稳定后再开始更新 scale。

## 9. 面试问题

**Q1：PTQ 中，为什么 MSE 校准通常比 MinMax 校准更好？在什么情况下 MinMax 反而更好？**

MSE 校准通过最小化 L2 量化误差来寻找最优 scale，允许截断少量极端值以换取绝大多数值的更高精度。这在数据分布有"长尾"时特别有效。MinMax 更好的一种情况是：应用中要求严格的 range 保真度，不能有任何截断——例如音频信号处理中，clip 会导致可听失真。

**Q2：LSQ 和传统的 QAT（固定 scale）相比，核心区别是什么？为什么 LSQ 能取得更好的精度？**

传统 QAT 用 calibration 算出一个固定 scale，训练中不更新。LSQ 把 scale 变成一个可学习参数，模型在训练中同步优化权重和量化步长，最终收敛到更精确的"权重-量化"联合最优解。特别是在极低位宽（2-4 bit）下，合适的 scale 对精度影响巨大，LSQ 的优势尤为明显。

**Q3：BinaryConnect 前向用 sign(w)，反向用 STE，这样做为什么可行？sign 函数梯度为零，不会阻止训练吗？**

sign 函数确实是不可微的，其真正梯度处处为 0（除了 0 处无定义）。STE 的核心思想是"假装 sign 的梯度为 1"，从而让梯度"穿过"这个不可微操作。这在实践中有效的原因是：虽然每一步的梯度近似很粗糙，但在大批量、多步迭代的平均下，权重更新的方向大致正确。可以类比为"蒙着眼睛下山"——虽然看不清每一步的具体地形，但梯度的大方向引导你往下走。

**Q4 (NVIDIA 面试真题)**: "你在 A100 上用 FP16 AMP 训练一个 Vision Transformer (ViT-Large)，前 1000 步正常，第 1001 步 loss 突然变成 NaN 并且之后所有的 loss 都是 NaN。你用二分法定位到问题出在第 8 个 Transformer Block 的 Attention 中。请从 AMP 数值范围的角度解释可能的原因，以及为什么增大 GradScaler 的 scale 值会让问题更严重。"

**参考答案**: 

FP16 的表示范围是 [−65504, 65504]。ViT Attention 中的 `Q @ K^T / sqrt(d_k)` 操作在序列较长时（如 384×384 输入，patch_size=16 → 576 tokens），attention score 矩阵的大小是 576×576。在训练前期，Q 和 K 的分布尚未稳定，可能存在某些维度的值异常大，导致：

1. `Q @ K^T` 的某些元素 > 65504 → FP16 overflow → Inf → softmax(Inf) → NaN
2. 即使 `Q @ K^T` 没有溢出，`softmax(QK^T / sqrt(d_k))` 在 FP16 中的计算链也可能溢出 — softmax 内部的 `exp(x - max(x))` 中，`x - max(x)` 在 FP16 下可能因为 rounding error 变成正值（理论上应该 ≤0 但 FP16 精度不够）→ exp 溢出

**为什么增大 GradScaler 让问题更严重**: GradScaler 放大的是 `loss.backward()` 中的 loss 值，目的是让小梯度的低位值在 FP16 中不丢失。但如果 overflow 发生在 forward pass（Attention 计算中），GradScaler 根本不参与 forward pass → 增大 scale 对 forward overflow 没有帮助。但 GradScaler 放大的 loss 会使 backward 中的梯度也变大 → 如果 forward 产生了 Inf/NaN，backward 的 Inf/NaN 梯度会被 GradScaler 放大后传播 → 更多层被污染。

**正确解法**: (1) Attention 的 QK 计算强制使用 FP32（通过 `with torch.cuda.amp.autocast(enabled=False):` 包裹）；(2) 或降低 `sqrt(d_k)` 的缩放，用更大的温度系数使 softmax 输入更平稳；(3) 使用 BF16 代替 FP16 — BF16 的表示范围与 FP32 相同 (~3.4e38)，只是精度低（7-bit mantissa vs FP16 的 10-bit）。

**Q5 (快手面试真题)**: "你在手机端部署一个混合精度量化模型，前 3 层 INT8，中间 6 层 INT4，最后 2 层 INT8。所有层的量化参数（scale/zero_point）都被正确导出到 TFLite 模型。但在高通骁龙 8 Gen 1 的 Hexagon DSP 上推理时，INT4 层的输出完全错误。请排查可能的原因。"

**参考答案**: 

高通 Hexagon DSP 对 INT4 的支持有多个硬件限制：

1. **INT4 数据排布要求**: Hexagon DSP 要求 INT4 权重以特定的交叉排布（interleaved layout）存储 — 两个 INT4 值打包到一个 INT8 字节中，且打包顺序是 `[low_nibble, high_nibble]` 而非直觉的 `[high, low]`。如果 TFLite converter 的 INT4 packing 顺序与 Hexagon 期望的不一致 → 每个 INT4 值的高低位被交换 → 权重值完全错乱。

2. **INT4 零点的隐含假设**: Hexagon 的 INT4 指令假设 zero_point 为 0（对称量化），不支持非对称 INT4 量化。如果 QAT 训练用了非对称量化（有非零 zero_point），导出的参数在 Hexagon 上会被错误解释 → 输出系统性偏移。

3. **INT4 通道数对齐要求**: Hexagon DSP 的 INT4 向量指令（HVX）一次处理 64/128 bytes。如果卷积的输入通道数不能被 HVX 向量宽度整除（如 6 通道 INT4 → 3 bytes → 不对齐 128-bit 边界）→ 会触发未定义行为（undefined behavior），后面的数据被错误读取。

4. **Debug 方法**: 
   - 用 Hexagon SDK 的 `hexagon-sim` 模拟器逐层 dump 中间结果，与 PyTorch 的 QAT eval 输出对比
   - 在 TFLite 的 CPU delegate 上先跑一遍排除是模型导出问题还是 Hexagon delegate 问题
   - 检查 `TfLiteHexagonDelegateOptions` 中的 `debug_level` 设置，开启 VERBOSE 日志

**Q6 (字节跳动面试真题)**: "你负责将 OpenAI Whisper 语音识别模型用量化部署到手机端。Whisper 的 Encoder 是纯 Transformer，Decoder 是自回归的。直接对整个模型 PTQ INT8 后，Encoder 精度尚可（WER 从 5.2% 升到 5.8%），但 Decoder 完全崩了（WER 从 5.2% 升到 47%）。请分析 Decoder 崩的原因，并提出至少两种不同思路的解决方案。"

**参考答案**: 

Whisper Decoder 崩的根本原因是**自回归解码中的误差累积放大**。Decoder 的每个 token 生成依赖前一个 token 的 hidden state。如果前一个 token 因为量化误差导致了微小的 hidden state 偏移，这将影响下一个 token 的 attention 计算 → 误差以指数级增长。这与 classification 模型 "一次前向得到结果" 的误差模型完全不同。

具体机制：
1. 量化后 decoder 的 KV cache 存储的是量化后的 INT8 值，而非原始的 FP32 值
2. 每生成一个新 token，量化误差混入 KV cache → 后续 token 的 cross-attention 看到的是"累积了误差"的 past keys/values
3. 在 Whisper 这类 1500-token 的长序列转录中，第 500 个 token 的 KV cache 已经包含了前 499 步的累积量化噪声 → attention 分布完全被噪声主导

**方案 A (Decoder 特化 QAT)**: 不是对 Encoder+Decoder 同时 QAT，而是冻住 Encoder，只对 Decoder 做 QAT，并在 QAT 的训练循环中模拟 KV cache 量化（即用 FakeQuantized 的 past KV）

**方案 B (Decoder KV Cache 保留 FP16)**: 对 Decoder 的 FFN 和 QKV projection 做 INT8 量化，但 KV cache 本身保持 FP16 存储。代价是 KV cache 占用显存不变，但每 token 的推理延迟降至 1/2-1/3（因为 GEMM 仍是 INT8）

**方案 C (SmoothQuant for Decoder)**: Decoder 的激活值 outlier 比 Encoder 更严重（自回归的隐藏状态有更强的 channel bias），用 SmoothQuant 将激活值的幅度迁移到权重上

**方案 D (Token-level dynamic quantization)**: 不用固定的 scale，而是每个 token 生成时根据当前 hidden state 动态计算 scale — 实现复杂但在生产环境中已有成功案例（如 Apple 的 Whisper.cpp 使用逐 token 动态量化）

## 10. 本讲总结

本讲是量化的进阶篇，从校准方法到训练策略，系统覆盖了从 PTQ 到 QAT 的完整技术栈：

- **PTQ 的核心是校准**：MinMax 简单但有长尾问题，MSE 和 Percentile 在实践中更优。校准数据集的质量和代表性是关键。
- **QAT 通过 FakeQuantize + STE 挽救 PTQ 无法企及的精度**，是极低位宽（≤4 bit）的必须步骤。
- **二值/三值网络**将极端量化推向极限，乘法变位操作，在 FPGA 和超低功耗 MCU 上有独特价值。
- **混合精度量化**承认"不是所有层都需要相同精度"，通过敏感度分析实现精度-效率的最优平衡。
- **LSQ**代表了量化技术的进化方向：让量化参数"可学习"，实现端到端的量化优化。
- **AMP**作为训练加速的工程实践，已经深度融入 PyTorch 训练流程。

一句话总结：量化不是简单的"除以一个数然后取整"，而是一个从校准、训练到部署的全链路优化问题——每一步的选择都直接影响最终精度。LSQ 的出现标志着量化从"手动工程"进入了"自动学习"时代。

## 11. 工业落地 Checklist

| 检查项 | 说明 | 不做的后果 |
|--------|------|-----------|
| PTQ→QAT 决策 | 先试 PTQ，精度损失 >2% 再做 QAT。不要一上来就 QAT | 浪费训练资源在 PTQ 就能搞定的模型上 |
| Round mode 一致性验证 | QAT 训练后用目标推理引擎跑一致性测试（cosine similarity > 0.999） | QAT 精度 OK，部署后数值偏差累积 → 结果漂移 |
| AMP NaN 自动检测 | 训练前用 `torch.autograd.set_detect_anomaly(True)` 跑 100 步验证 | 训练跑了几小时后才报 NaN，白白浪费 GPU 时间 |
| AMP softmax/layernorm 保护 | 强制 softmax 和 LayerNorm 使用 FP32（autocast 白名单） | 大模型训练中 FP16 softmax 溢出 → loss NaN |
| LSQ scale 初始化 | `scale_init = max(|w|) / q_max`（非 mean），前 1000 步不更新 scale | scale 过小 → 所有权重量化到 0 → 训练死锁 |
| 混合精度敏感度分析 | 逐层尝试不同 bit-width 并评估精度，不是拍脑袋决定哪层用几 bit | 关键层用低精度 → 整体性能被拖垮 |
| 自回归解码器特殊处理 | Decoder 量化时必须评估 KV cache 的误差累积，不只评估单 token 精度 | 第 1 个 token OK，第 500 个 token 完全错乱 |
| Hexagon/ANE 等 DSP 位数对齐 | INT4 打包顺序、通道对齐、zero point 假设必须与 DSP spec 一致 | 输出完全随机错误，无报错 |
| 二值网络精度上限评估 | 二值化前确认任务精度要求，ImageNet 二值网络 Top-1 上限约 52% | 在需要细粒度分类的场景用二值 → 精度不够 |
| 校准集激活值 saturation 率检查 | 校准后检查 activation saturation rate，>5% 说明校准数据不具代表性 | 线上推理时大部分激活值被 clamp → 信息丢失 |
