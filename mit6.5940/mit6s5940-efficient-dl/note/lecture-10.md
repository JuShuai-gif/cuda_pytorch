# 第十讲：MCUNet — TinyML 系统与模型-引擎协同设计

## 1. 本讲核心问题

当深度学习要跑到一个内存只有 256KB SRAM、2MB Flash 的微控制器（MCU）上时，传统方法全部失效——连一个标准的 MobileNet 都装不下。MCUNet 通过**模型-引擎协同设计 (Model+Engine Co-Design)**解决了这个问题。本讲的核心问题：

- **MCU 上的极端硬件约束**是什么？SRAM < 512KB, Flash < 2MB 意味着什么？为什么在 MCU 上做推理是全栈问题而非单纯压缩模型？
- **什么是 TinyNAS？** 它如何在一个受硬内存约束的搜索空间中自动发现最优架构？
- **基于 Patches 的推理 (Patch-based Inference)** 如何解决 MCU 看不到完整图像的问题？
- **TinyEngine** 为什么不能直接用 TFLite Micro？如何针对 MCU 做"极致优化"？
- **MCUNetV2 的重新分配 (Redistribution)** 如何进一步突破内存瓶颈？什么是"receptive field redistribution"？
- **模型-引擎协同设计 (Co-Design)** 的哲学是什么？为什么比"先设计模型再优化引擎"更好？

## 2. 通俗解释

想象你有一个**火柴盒大小的单片机（MCU）**，你想让它实时识别人脸、听懂关键词。这个火柴盒只有：

- **SRAM (256KB)**：相当于你的**"便签纸"**——你只能同时记两三行数字在上面，做完一步就必须擦掉再记下一步。这对应的是推理时中间结果（激活值）的临时存储空间。
- **Flash (2MB)**：相当于你的**"口袋小本本"**——所有知识的"字典"都放在这里，但翻本子比在便签纸上写慢很多。这对应的是模型权重的永久存储空间。
- **CPU (ARM Cortex-M4, 几十到几百 MHz)**：相当于你的**"计算速度"**——慢，做一次乘法约需零点几微秒。没有任何并行计算能力（没有 GPU）。

MCUNet 的解决方案，用一个故事来比喻：

你是一个野外探险家，背包只有 2 升（Flash），口袋只有 250 毫升（SRAM），步行速度只有 3km/h（CPU）。你要辨认 1000 种野生动植物（ImageNet 分类）。

- **TinyNAS**：你不能带一本《世界动植物百科全书》（ResNet-50），太重了（25MB+）。你也不能自己手写一本——你没有这个生物学知识。于是你雇了一个 AI 助手：它分析你的背包和口袋尺寸，自动写出一本"刚好放得下、够用且最准"的迷你手册。这就是 TinyNAS——在严格的硬件预算下，自动搜索最优模型架构。

- **Patch-based Inference**：你的火柴盒太小了，一次只能放手掌大的一张照片。如果你要辨认一头大象，你不能一次看到整头大象！于是你采取了**"瞎子摸象"策略**：先把照片切成 4 块（patches），每块单独推理，然后把结果汇总。你摸到鼻子（第一块）→ 觉得像蛇；摸到腿（第二块）→ 觉得像柱子；综合起来 → 判断是大象。Patch-based 推理允许模型在极小的 SRAM 里处理任意大的输入——代价是精度可能略有损失（因为缺乏全局上下文）。

- **TinyEngine**：你写迷你手册的字迹（计算图）非常讲究。普通词典的排版（TFLite Micro）太浪费纸张了——同样的内容，经过 TinyEngine 的"重新排版"，多塞进了 30% 的内容。TinyEngine 做的事情是：针对 ARM Cortex-M 芯片的每一个特性（SIMD 指令、内存层级、cache line 对齐）做了极致的手工优化，把所有不必要的开销全砍掉了。

- **MCUNetV2 + Redistribution**：你发现了一个更强的技巧——你不是均匀地把知识分配在整本手册里，而是把"最重要的识别线索"放在最前面几页，这样你大部分时候只需要翻前 10 页就够了。这就是 receptive field redistribution：调整网络的感受野分布，让前端层捕捉全局信息（增大 kernel/stride），后端层做精细判断——在给定内存预算下提高信息利用效率。

## 3. 关键公式

### MCU 内存模型

闪存约束（模型大小）：
$$\text{Model Size} = \sum_{l=1}^{L} |W_l| \times \text{bitwidth} \leq \text{Flash Budget}$$

SRAM 约束（运行时内存峰值）：
$$\text{Peak Memory} = \max_{l} \left(\text{Input Activation}_{l} + \text{Output Activation}_{l} + \text{Weight Buffer}_{l}\right) \leq \text{SRAM Budget}$$

对于标准卷积层：
$$\text{Peak Memory}_l = H_l W_l C_{in} + H_l W_l C_{out} + K^2 C_{in} C_{out}$$
（假设 batch size = 1，int8 量化，通道维度）

### TinyNAS 搜索目标

在内存硬约束下最大化精度：
$$\max_{a \in \mathcal{A}} \text{Acc}(a)$$
$$\text{s.t. } \text{PeakMemory}(a) \leq \text{SRAM}, \quad \text{ModelSize}(a) \leq \text{Flash}$$

### Patch-based 推理的精度损失

完整图像推理的 logits 为 f(I)，patch 推理为 f_patch(I)：
$$\text{Gap} = \|\text{softmax}(f(I)) - \text{softmax}(\text{ensemble}(f_{patch}(I)))\|$$

patch 数量 P vs 精度：
$$\text{Acc} \propto 1 - e^{-\beta P}$$

当 P 足够大时（通常 P ≥ 4），精度接近完整图像推理。

### MCUNetV2 的 receptive field redistribution

重新分配各层的感受野（RF）和 stride：
$$\text{RF}_l = \sum_{i=1}^{l} \left((k_i - 1) \prod_{j=1}^{i-1} s_j\right) + 1$$

通过增大早期层的 kernel size 或 stride，使得：
$$\text{RF}_{early} \gg \text{RF}_{early}^{baseline}, \quad \text{而} \quad \text{PeakMemory}_{early} \text{ 增幅可控}$$

## 4. 公式背后的直觉

- **内存峰值决定生死**：很多研究者只关注模型总大小（Flash），但 MCU 推理的致命瓶颈是**SRAM 的运行时峰值**。想象你有一个窄走廊——你的身高体重（总参数）没问题，但在某个节点你需要同时挤过一张大桌子和一把大椅子（某层的输入+输出+权重同时驻留在 SRAM 中），这就是峰值。MCUNet 的搜索空间强制排除任何违反峰值约束的候选架构。

- **为什么直接的 Uniform Shrinking 不行**：直观上，把 MobileNet 的宽度乘子（width multiplier）设为 0.1 就装得下了？但实验证明，这样做的精度极差（ImageNet top-1 只有 ~40%，随机猜测是 0.1%）。原因：太小（太窄）的网络失去了"表示瓶颈"——每一层只能传递极少量的信息，最终全丢了。TinyNAS 发现的架构不是"更窄的 MobileNet"，而是**不同形状的架构**——比如更深的、但通道分布更精细的网络。

- **Patch-based 推理的权衡**：将一张 224×224 的图切成 4 个 112×112 的 patch，每个 patch 单独推理。优点：峰值内存降到原来的约 1/4（因为激活尺寸小了）。缺点：(1) 失去了 patch 边界处的空间信息（但通过重叠切割可缓解）；(2) 无法利用全局上下文（比如你看到的 patch 可能是一只猫的耳朵，但看不到猫的身体）；(3) 4 次推理 = 4 倍延延迟。对于"是否需要即时响应"的应用，这点要权衡。

- **Redistribution 的巧妙之处**：传统 CNN 的感受野是逐层递增的。但在资源极度受限的情况下，按照传统分配方式，网络在早期层的感受野会很小，等到拥有足够大的感受野时，已经消耗了大量计算。Redistribution 的思路是**把感受野的"增长"向前移动**——让早期层就获得较大的感受野（通过更大的 stride 或 dilation），从而在给定的总计算量下，让网络更快地"看到"全局信息。这类似于拍照时先调好焦距再按快门，而不是按完快门再回去修图。

- **协同设计 (Co-Design) 的深层原因**：在深度学习部署中，一个问题常被忽视：**模型的设计影响引擎的效率，引擎的能力也反过来限制模型的设计**。例如，如果一个引擎对 3x3 depthwise conv 有特殊优化（SIMD + loop unrolling），那 NAS 在搜索时就应该"奖励"使用这个操作——反过来，如果 NAS 发现某个操作精度好但引擎不支持，工程师就应该去引擎里加上这个操作的优化实现。Co-Design 正是认识到这两者是"互相塑造"的，应该一起优化。

## 5. 工业界用途

- **STMicroelectronics STM32**：MCUNet 被部署在 STM32F746 (320KB SRAM, 1MB Flash) 和 STM32H743 (1MB SRAM, 2MB Flash) 上，实现了在 256KB SRAM 内运行 ImageNet 1000 类分类。
- **Keyword Spotting (KWS)**：Google 的 "Hey Google" 唤醒词检测运行在 MCU 上，MCUNet 的 TinyNAS + TinyEngine 组合可用于为特定唤醒词自动搜索最优 KWS 模型架构，支持多唤醒词同时检测。
- **Person Detection**：ARM 维护的 TensorFlow Lite Micro 示例 "person_detection" 使用 MCU 做人员检测——250KB 模型，在 ARM Cortex-M4 上 200ms 推理一次。MCUNet 的 co-design 方法可以进一步将延迟压到 100ms 以下。
- **Predictive Maintenance**：工厂里的振动传感器使用 MCU 推理异常检测模型。电池供电，需要超低功耗。MCUNet 派生的模型可以在 50KB 以内实现 >95% 的故障检测准确率。
- **农业 IoT**：安装在田间的 MCU 传感器分析图像（叶面病害检测、果实计数），通过太阳能供电。由于没有 WiFi，所有推理必须在端侧完成。MCUNet 的 patch-based 推理支持处理高分辨率图像（如 640×480）。
- **可穿戴设备**：智能手表上的心率异常检测、跌倒检测。MCU 功耗 < 1mW，而一个简单的蓝牙数据传输就耗 10mW+。能本地推理 = 更省电。

## 6. PyTorch 实现思路

### TinyNAS 搜索空间设计（概念代码）

```python
class MCUConstrainedSearchSpace:
    """为 MCU 定制的搜索空间：只包含 memory-efficient 操作"""

    def __init__(self, sram_budget=256*1024, flash_budget=1*1024*1024):
        self.sram_budget = sram_budget  # bytes
        self.flash_budget = flash_budget  # bytes

        # 候选操作：全是 MCU 友好的
        self.ops = [
            'MBConv_k3_e3',   # MobileNetV2 inverted bottleneck, kernel=3, expand=3
            'MBConv_k5_e3',
            'MBConv_k3_e4',
            'MBConv_k5_e4',
            'MBConv_k3_e6',
            'MBConv_k5_e6',
        ]

        # 每层的候选输出通道数（离散值，方便在 MCU 上对齐）
        self.channel_choices = [8, 16, 24, 32, 40, 48, 56, 64, 80, 96]

        # stride 选择（决定分辨率下降位置）
        self.stride_configs = [(1,1,1), (2,1,1), (1,2,1), (2,2,1), (1,2,2)]

    def estimate_peak_memory(self, architecture):
        """估算给定架构的 SRAM 峰值使用量（简化版）"""
        peak = 0
        h, w = 224, 224  # 输入分辨率（假设整图推理）
        for layer in architecture:
            if layer.stride == 2:
                h, w = h//2, w//2
            # 估算：input activation + output activation + weight buffer
            # 假设 INT8，batch_size=1
            input_mem = h * w * layer.in_channels  # INT8 -> 1 byte per element
            output_mem = h * w * layer.out_channels
            weight_mem = layer.kernel_size**2 * layer.in_channels * layer.out_channels
            peak = max(peak, input_mem + output_mem + weight_mem)
        return peak

    def estimate_model_size(self, architecture):
        """估算 Flash 占用"""
        total = 0
        for layer in architecture:
            total += layer.kernel_size**2 * layer.in_channels * layer.out_channels
        return total

    def is_valid(self, architecture):
        """检查架构是否满足硬件约束"""
        return (self.estimate_peak_memory(architecture) <= self.sram_budget and
                self.estimate_model_size(architecture) <= self.flash_budget)
```

### Patch-based 推理的 PyTorch 实现

```python
def patch_based_inference(model, image, patch_size=112, overlap=8):
    """
    将大图切分成 patches，逐 patch 推理后 ensemble
    适用于 MCU 模拟——每个 patch 的 activation memory 很小
    """
    _, _, H, W = image.shape
    patches = []
    positions = []

    # 切分 patches，带 overlap
    for y in range(0, H - overlap, patch_size - overlap):
        for x in range(0, W - overlap, patch_size - overlap):
            y_end = min(y + patch_size, H)
            x_end = min(x + patch_size, W)
            # 如果最后一块不完整，向左/上移动起始位置
            y_start = max(0, y_end - patch_size)
            x_start = max(0, x_end - patch_size)
            patch = image[:, :, y_start:y_end, x_start:x_end]
            patches.append(patch)
            positions.append((y_start, x_start, y_end, x_end))

    # 逐 patch 推理
    patch_logits = []
    for patch in patches:
        with torch.no_grad():
            logits = model(patch)
            patch_logits.append(F.softmax(logits, dim=1))

    # Ensemble：平均所有 patch 的 softmax 概率
    final_probs = torch.stack(patch_logits).mean(dim=0)

    return final_probs
```

### MCUNetV2 Redistribution 概念模拟

```python
class RedistributedNetwork(nn.Module):
    """
    MCUNetV2 风格的网络：早期层放大感受野
    对比传统设计：
    - 传统：stride 均匀分布
    - Redistributed：早期更大的 stride/kernel，快速建立感受野
    """

    def __init__(self, num_classes=1000):
        super().__init__()
        # 传统设计：第一层 stride=2, kernel=3
        # MCUNetV2 设计：第一层 stride=2, kernel=5 (更大感受野)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
        )

        # 快速下采样阶段：用更大的 stride 快速建立感受野
        self.stage1 = nn.Sequential(
            InvertedResidual(16, 24, stride=2, expand_ratio=6),  # 快速下采样
            InvertedResidual(24, 24, stride=1, expand_ratio=3),
        )

        # 精细特征提取阶段：较慢的下采样
        self.stage2 = nn.Sequential(
            InvertedResidual(24, 40, stride=2, expand_ratio=6),
            InvertedResidual(40, 40, stride=1, expand_ratio=3),
            InvertedResidual(40, 40, stride=1, expand_ratio=3),
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Conv2d(40, 160, 1),
            nn.BatchNorm2d(160),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(160, num_classes, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)
```

### 内存峰值估算工具

```python
def estimate_memory_usage(model, input_shape=(1, 3, 224, 224)):
    """
    估算模型在推理时的峰值内存使用（简化的 PyTorch 模拟）
    在实际部署中，需要更精确的 per-op 内存分析
    """
    peak = 0
    x = torch.randn(input_shape)

    def hook_fn(module, input, output):
        nonlocal peak
        # 粗略估算：输入 + 输出 + 参数（INT8 近似）
        input_mem = input[0].numel() if isinstance(input, tuple) else input.numel()
        output_mem = output.numel()
        param_mem = sum(p.numel() for p in module.parameters())
        total = (input_mem + output_mem + param_mem)  # bytes if INT8
        if total > peak:
            peak = total

    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook_fn))

    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

    return peak
```

## 7. TinyML / Edge AI 部署意义

- **MCUNet 是 TinyML 的一次"范式转变"**：在此之前，MCU 上的 DL 主要是简单的 KWS（关键词检测）和手势识别（2-4 类）。MCUNet 证明了即使是 1000 类 ImageNet 分类也可以在 256KB SRAM 内完成。
- **Patch-based 推理打破了"输入分辨率 = 内存瓶颈"的魔咒**：MCU 现在可以处理任意大小的输入图像——只要愿意接受更多 patch（更多延迟）。这对于需要高分辨率判断但延迟要求不严格的应用（如农业病害检测）至关重要。
- **模型-引擎协同设计是 TinyML 的"系统思维"**：只优化模型 → 引擎跑得慢；只优化引擎 → 模型太烂。MCUNet 两者一起做，达成了 1+1 > 2 的效果。这种哲学现在被整个 TinyML 社区接受。
- **Redistribution 策略的通用性**：虽然 MCUNetV2 是为 MCU 设计的，但 redistribution 的思想（合理分配网络早期的感受野和计算）也可以在更大的模型上应用——它挑战了"stride 均匀分布"的传统设计惯例。
- **功耗的隐含优势**：MCU 上的模型推理功耗通常 < 1mW，而数据传输（蓝牙/WiFi）功耗是 10-100mW。能够在设备上本地推理 = 可以关闭通信模块，续航延长 10x+。这对电池供电的 IoT 设备至关重要。

## 8. 常见误区

1. **"MCU 上的模型就是 MobileNet 缩到很小"**：不是。直接缩小 MobileNet（width multiplier = 0.1）的精度极差。适合 MCU 的架构和适合手机 GPU 的架构在形状上有本质区别——MCU 偏好更深的、通道分布更精细的、操作更少样的架构。
2. **"TFLite Micro 已经是 MCU 最优推理引擎了"**：不是。TFLite Micro 是一个通用引擎，目标是支持大多数操作和架构。TinyEngine 是一个专用引擎，只为 TinyNAS 搜索到的操作子集做极致优化——包括 custom loop ordering、in-place depthwise conv、专门的 memory layout。TinyEngine 在 ImageNet 模型上比 TFLite Micro 快 2-5x。
3. **"Patch-based 推理免费"**：4 个 patch = 4 次推理 = 4 倍延迟。对于需要实时响应的应用（如 keyword spotting），需要权衡 patch 数量和响应延迟。
4. **"全部用 INT8 就不用担心内存了"**：INT8 确实把内存压缩到 FP32 的 1/4，但对于 256KB SRAM 来说，即使 INT8 也可能不够。MCUNet 的搜索空间不仅考虑参数量，还考虑了**激活内存的峰值**——而这个值主要由输入分辨率和通道数决定，与量化位宽关系较小。
5. **"Co-design 就是 '先搜模型再优化引擎'"**：不对。Co-design 是**同时**优化模型和引擎。TinyNAS 在搜索模型时，使用的 reward 不仅包含精度，还包含 TinyEngine 对该架构的实际推理延迟和内存使用——这两者是耦合的。
6. **"MCU 不能跑任何视觉任务"**：MCUNet 已经证明了 MCU 可以跑 ImageNet 1000 类分类、VWW (Visual Wake Words) 人员检测、以及多类目标分割。MCU 的极限远未被完全探索。

## 9. 面试问题

**Q1：MCUNet 的核心创新是什么？为什么它被称为 "Model+Engine Co-Design"？**

MCUNet 的核心创新是认识到 MCU 推理不是单纯的模型压缩问题，而是一个**全栈系统优化问题**。具体来说：(1) TinyNAS 在一个受硬内存约束（SRAM 和 Flash 硬上限）的搜索空间中自动搜索最优架构——传统 NAS 只考虑精度，TinyNAS 的每个候选架构都必须通过内存可行性检查；(2) TinyEngine 是专为 TinyNAS 搜索的操作子集定制的推理引擎，通过 loop optimization、in-place depthwise conv、memory layout 优化等技术实现了远超 TFLite Micro 的性能；(3) 关键的是，这两个组件不是独立设计的——TinyNAS 在搜索时使用 TinyEngine 的实际性能数据作为评估标准。这就是 "Co-Design"：模型和引擎互相塑造、联合优化，而非各自为战。结果是在 256KB SRAM 的 MCU 上实现了 ImageNet 级别的分类。

**Q2：在 MCU 上，为什么 Patch-based Inference 是必要的？它有什么代价？**

MCU 的 SRAM 大小通常不足以存放一张高分辨率图像（如 224×224×3 = 150KB）的全部激活值，更不用说加上权重和中间结果了。Patch-based inference 将输入切分成多个小块，每个 patch 的激活内存远小于整张图的激活内存，从而满足 MCU 的 SRAM 限制。代价是：(1) 延迟增加——N 个 patch 意味着 N 次推理；(2) 精度损失——每个 patch 缺乏全局上下文（比如一个 patch 可能只包含一只猫的耳朵，没有身体线索）；(3) 重叠区域的计算冗余。不过，通过适当的 overlap 和 ensemble（平均 softmax 概率），精度损失通常可以控制在 1-2 个百分点以内。

**Q3：TinyEngine 相对于 TFLite Micro 做了哪些关键优化，为什么能获得数量级的加速？**

TinyEngine 的关键优化包括：(1) **In-place depthwise convolution**：标准的 depthwise conv 需要分配单独的输入和输出 buffer，TinyEngine 在满足条件时让输出直接覆盖输入，节省了 SRAM 并减少了内存带宽；(2) **Loop ordering optimization**：针对 ARM Cortex-M 的 cache hierarchy 和 SIMD 宽度（128-bit NEON），重新排列卷积循环的嵌套顺序（如先遍历输出通道而不是输入通道），提高 cache 命中率；(3) **Operator fusion**：将连续的算子（如 Conv + BN + ReLU）融合为一个 kernel，消除中间 buffer 的分配和读写；(4) **Specialized memory layout**：为深度可分离卷积设计特殊的内存布局（如 CHW vs HWC 针对不同层），使得 SIMD 加载更高效。这些优化是**手工针对 ARM Cortex-M 指令集精心调校的**，而 TFLite Micro 追求通用性，无法做到这种级别的底层优化。

## 10. 本讲总结

MCUNet 是 Efficient DL 课程最具代表性的案例——它展示了"系统优化"的思维高度：

- **MCU 的约束不是"小"的问题，而是"根本性地不同"的问题**——没有 GPU、没有大内存、没有成熟的软件栈。直接把为 GPU 设计的模型缩小的策略完全失败。
- **TinyNAS** 证明了在极端约束下，自动搜索的架构性能远超手工设计和 uniform shrinking。**约束驱动的搜索空间设计**是关键。
- **Patch-based 推理**解决了 MCU 的输入分辨率瓶颈，以延迟换精度，使得 ImageNet 级别的视觉任务在 256KB SRAM 内成为可能。
- **TinyEngine** 通过针对特定硬件的极致手工优化，实现了比通用框架（TFLite Micro）2-5x 的推理加速。这不是"更好的编译器"，而是"为 MCU 手工打造的推理引擎"。
- **MCUNetV2 的 Redistribution** 揭示了架构设计的另一个自由度：感受野的分布策略。不是均匀增长，而是"好钢用在刀刃上"——把计算资源集中到最需要的地方。
- **Co-Design 哲学**是本讲最重要的思想：在资源极度受限的系统中，单个组件的优化不足以突破瓶颈，**必须让模型、引擎、硬件三者相互适应、联合优化**。

一句话总结：MCUNet 给你的启示不是"怎么把模型做小"，而是"在资源受限的系统里，你必须重新思考一切——从架构到引擎到推理策略"。这不是压缩，而是一次完整的系统重构。
