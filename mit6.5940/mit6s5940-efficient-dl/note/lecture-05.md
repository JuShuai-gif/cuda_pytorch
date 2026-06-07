# Lecture 05-06: 量化 (Quantization) - 用"低像素"表示神经网络

## 1. 本讲核心问题

> FP32 的权重太"奢侈"了。能不能用 INT8 甚至 INT4 来存？精度损失多少？推理能快多少？

## 2. 通俗解释

**生活类比 — 像素 vs 矢量图**：

一张照片你用 4800万像素存储（FP32），细节丰富但文件大。如果你把它缩放成 800×600（INT8），虽然稍模糊但人眼基本分不出，文件却小了很多。

量化的本质就是：**用更少的比特数表示权重和激活值，牺牲微小精度换取巨大的存储和速度收益**。

但关键在于——哪里能模糊、哪里不能模糊？这取决于数据的分布。

## 3. 关键公式

### 3.1 为什么低精度运算更省电？

来自 Horowitz (2014) 的经典数据：

| 运算 | 能耗 (pJ) | 相对FP32 |
|------|-----------|----------|
| INT8 ADD | 0.03 | 1/30x |
| INT8 MULT | 0.2 | 1/18x |
| FP16 ADD | 0.4 | 1/2x |
| FP32 ADD | 0.9 | 1x |
| FP32 MULT | 3.7 | 1x |
| SRAM Read (8KB) | 10 | - |
| DRAM Read | 1300 | - |

> **关键洞察**: 内存访问能耗远大于计算能耗！量化不仅减少计算，更重要的是减少内存访问。

### 3.2 线性量化的核心公式

**量化** (Float → Int):
$$q = \text{clamp}\left(\text{round}\left(\frac{r}{S}\right) + Z, \ q_{min}, \ q_{max}\right)$$

**反量化** (Int → Float):
$$r = S \cdot (q - Z)$$

其中：
- $S$ = Scale（缩放因子）：$S = \frac{r_{max} - r_{min}}{q_{max} - q_{min}}$
- $Z$ = Zero Point（零点偏移）：$Z = \text{round}(q_{min} - \frac{r_{min}}{S})$
- $r$ 是浮点值，$q$ 是量化后的整数值

**核心思想**: 把浮点数的范围 $[r_{min}, r_{max}]$ 线性映射到整数范围 $[q_{min}, q_{max}]$。

### 3.3 量化粒度

| 粒度 | Scale/ZeroPoint 个数 | 精度 | 开销 |
|------|---------------------|------|------|
| Per-Tensor | 整个张量1组 | 最低 | 最小 |
| Per-Channel | 每个通道1组 | 较高 | 较小 |
| Per-Group | 每N个元素1组 | 最高 | 较大 |

**Per-Channel 为什么重要？**

不同通道的权重分布差异可能很大。用同一个 scale 会导致某些通道信息丢失严重。

### 3.4 量化矩阵乘法

量化后的计算（带bias）：

$$y_{int} = \frac{S_w S_x}{S_y} \left( W_{int} X_{int} + b_{int} \right)$$

其中 $b_{int}$ 需要特殊处理：
$$S_{bias} = S_w \cdot S_x$$
$$b_{int} = \text{round}(b / S_{bias})$$

### 3.5 PTQ vs QAT

- **PTQ (Post-Training Quantization)**: 训练完成后，直接量化。无需重训练。简单但精度可能下降明显。
- **QAT (Quantization-Aware Training)**: 训练/微调时模拟量化效果（fake quantization），让模型"学会"在低精度下表现好。

## 4. 公式背后的直觉

### Scale 到底在干什么？

想象你要把一个杯子里的水倒进另一个大小不同的杯子：
- **原杯子**: 浮点数，可以装很多水（值域大）
- **目标杯子**: INT8，只装256滴水（-128到127）
- **Scale** = 换算比例：原杯子100ml对应目标杯50格 → Scale=2ml/格
- **Zero Point** = 校准：原杯0ml对应目标杯-128格 → Z=-128

### 为什么量化会损失精度？

量化误差 (Quantization Error):

$$\text{MSE} \approx \frac{1}{12} \cdot \Delta^2, \quad \Delta = 2S$$

- $\Delta$ 是两个量化级之间的间隔
- 8-bit → 256级 → $\Delta$ 较小 → 精度损失可接受
- 4-bit → 16级 → $\Delta$ 较大 → 精度损失明显
- 2-bit → 4级 → 几乎只有 +最大/0/-最大 → 严重损失

### K-Means 量化 vs 线性量化

- **线性量化**: 等间距量化，简单快速，硬件友好（适合 INT8 推理）
- **K-Means 量化**: 按数据分布聚类，非等间距，压缩率更高但推理时需要查表

## 5. 工业界用途

| 场景 | 量化方案 | 工具 |
|------|----------|------|
| 手机端推理 | INT8 PTQ | TensorFlow Lite, CoreML |
| 云端推理优化 | INT8/FP8 QAT | TensorRT, ONNX Runtime |
| LLM本地部署 | W4A16 (AWQ/GPTQ) | llama.cpp, vLLM |
| MCU推理 | INT8 per-tensor | TinyEngine, TFLite Micro |
| 训练加速 | FP16/BF16混合精度 | PyTorch AMP, NVIDIA APEX |

### 实际案例

- **BERT 量化**: INT8 PTQ → 2x加速, <1%精度损失 (TensorRT)
- **LLaMA 7B INT4**: AWQ量化 → 可在手机/笔记本运行
- **ResNet50 INT8**: TensorRT部署 → 3x加速 vs FP32

### 大厂量化实战数字

- **美团搜索推荐 BERT 量化**: BERT-Base (110M params) INT8 量化后推理延迟从 12ms 降至 5.4ms (加速 2.2x)，显存从 440MB 降到 110MB。但直接在中文搜索排序任务上 PTQ，NDCG@10 从 0.832 跌到 0.807（掉 3%）。通过 QAT（在搜索数据上微调 5000 步）恢复到 0.829。**教训**: NLP 模型的激活值分布比 CNN 不规则得多，PTQ 的直接校准精度不如分类任务，搜索排序场景对精度极度敏感。

- **快手端上视频分类模型 INT8 量化**: 使用 ncnn INT8 量化 MobileNetV3，延迟从 FP32 的 89ms 降至 31ms（加速 2.9x），功耗从 1.2W 降到 0.38W（降低 68%）。关键发现：ncnn 的 INT8 inference 使用了 ARM NEON `SMLAL` 指令（一次做 4 个 INT16→INT32 的乘累加），实际吞吐比 FP32 用 NEON `FMLA` 高 4.2x，比理论 2x 好得多 — 因为 INT8 的 NEON 指令是专门的 SIMD 优化路径，FP32 的 FMLA 管线有气泡（bubble）。

- **字节跳动抖音推荐模型量化路线**: FP32 → FP16 (AMP) → INT8 PTQ → INT8 QAT → INT8 + Per-Channel。每个阶段的效果：
  - FP32→FP16: 延迟降低 38%，精度无损（因为 AMP 训练已经做了适配）
  - FP16→INT8 PTQ: 延迟再降 35%，Top-1 降 0.2%
  - INT8 PTQ→QAT: 精度恢复 0.1%，代价是 2 天额外训练
  - Per-Tensor→Per-Channel: 精度再提升 0.15%，延迟无变化

### 量化部署成本分析（字节跳动推荐场景）

一个典型的短视频推荐模型推理集群（假设每日 50 亿次推理，FP32 ResNet50 等效计算量）：

| 配置 | GPU 卡数 | 单卡日成本 | 年总成本 | 精度 |
|------|---------|-----------|---------|------|
| FP32 | 2000 | \$35 | \$25,550,000 | 76.1% |
| FP16 | 1200 | \$35 | \$15,330,000 | 76.1% |
| INT8 PTQ | 700 | \$35 | \$8,942,500 | 75.9% |
| INT8 QAT | 700 | \$35 | \$8,942,500 | 76.0% |
| INT8+剪枝 | 400 | \$35 | \$5,110,000 | 75.6% |

> **关键结论**: FP32→INT8+剪枝，年度 GPU 成本从 2555 万美元降至 511 万美元，节省 80%（超过 2000 万美元/年）。对于推荐系统等对精度容忍度较高的场景，这是几乎无痛的成本优化。对于安全攸关场景（如支付风控），宁可多花 2000 万也要保证精度。

## 6. PyTorch 实现思路

### 6.1 基础线性量化

```python
import torch
import torch.nn as nn

def linear_quantize(tensor, bits=8):
    """对张量进行线性量化"""
    qmin, qmax = -(2 ** (bits - 1)), (2 ** (bits - 1)) - 1
    
    rmin, rmax = tensor.min(), tensor.max()
    scale = (rmax - rmin) / (qmax - qmin)
    zero_point = qmin - torch.round(rmin / scale)
    zero_point = zero_point.clamp(qmin, qmax).to(torch.int32)
    
    # 量化
    q = torch.clamp(torch.round(tensor / scale) + zero_point, qmin, qmax)
    q = q.to(torch.int8)
    
    # 反量化
    r = scale * (q.float() - zero_point.float())
    
    return q, scale, zero_point, r

# 测试
w = torch.randn(4, 4) * 3  # 模拟权重, std=3
q, s, z, r = linear_quantize(w, bits=8)
error = (w - r).abs().mean()
print(f"Weight: {w}")
print(f"Quantized: {q}")
print(f"Reconstructed: {r}")
print(f"Mean Abs Error: {error:.6f}, Scale: {s:.6f}, ZeroPoint: {z}")
```

### 6.2 量化感知训练 (QAT)

```python
class FakeQuantize(nn.Module):
    """模拟量化的模块, 用于 QAT"""
    def __init__(self, bits=8):
        super().__init__()
        self.bits = bits
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.zero_point = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, x):
        qmin, qmax = -(2**(self.bits-1)), (2**(self.bits-1))-1
        # Fake量化: 前向使用量化值, 反向传播使用STE(Straight-Through Estimator)
        if self.training:
            x_q = torch.clamp(torch.round(x / self.scale) + self.zero_point, qmin, qmax)
            x_r = self.scale * (x_q - self.zero_point)
            # STE: 反向传播时梯度直通
            return x + (x_r - x).detach()
        else:
            # 推理时真正量化
            x_q = torch.clamp(torch.round(x / self.scale) + self.zero_point, qmin, qmax)
            return self.scale * (x_q - self.zero_point)

class QATConv2d(nn.Module):
    """带 QAT 的卷积层"""
    def __init__(self, conv, weight_bits=8, act_bits=8):
        super().__init__()
        self.conv = conv
        self.weight_quant = FakeQuantize(weight_bits)
        self.act_quant = FakeQuantize(act_bits)
    
    def forward(self, x):
        w = self.weight_quant(self.conv.weight)
        x = self.act_quant(x)
        return nn.functional.conv2d(
            x, w, self.conv.bias,
            self.conv.stride, self.conv.padding,
            self.conv.dilation, self.conv.groups
        )
```

### 6.3 INT8 推理模拟

```python
def calibrate_activation_ranges(model, dataloader, num_batches=10):
    """校准激活值范围 (用于 PTQ)"""
    act_ranges = {}
    hooks = []
    
    def hook_fn(name):
        def fn(module, input, output):
            if name not in act_ranges:
                act_ranges[name] = {'min': float('inf'), 'max': float('-inf')}
            act_ranges[name]['min'] = min(act_ranges[name]['min'], output.min().item())
            act_ranges[name]['max'] = max(act_ranges[name]['max'], output.max().item())
        return fn
    
    # 注册 hook
    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # 在校准数据集上运行
    model.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            model(data)
    
    # 移除 hooks
    for h in hooks:
        h.remove()
    
    return act_ranges
```

### 校准数据选择策略 — 比量化算法本身更重要

校准数据是 PTQ 中最容易被低估的环节。校准数据的选择直接决定量化的成败：

```python
def calibration_data_strategy(dataloader, num_calibration_samples: int = 1024):
    """Production-grade calibration data selection strategy.
    
    CRITICAL RULES:
    1. Calibration data MUST match the distribution of inference-time data.
       Using ImageNet to calibrate a model deployed on surveillance footage
       → activation ranges mismatch → silent accuracy degradation on dark scenes.
    
    2. Sample diversity matters more than sample count.
       200 diverse samples > 10,000 homogeneous samples.
       Include: different lighting conditions, object scales, backgrounds.
    
    3. Always include some "worst-case" samples (edge cases):
       - Extreme lighting (very dark / very bright)
       - Rare object poses and occlusions
       - These samples define your activation range's upper bound
       → using only "easy" samples gives tight ranges → quantization error on 
         real-world hard cases is amplified 5-10×.
    
    4. For NLP models: calibration data should cover all sequence lengths
       that appear in production (short queries AND long documents).
       Padding-only tokens create spurious zero-activations that skew ranges.
    """
    # Strategy: stratified sampling from production traffic
    # 80% randomly sampled from recent production logs
    # 20% hand-picked edge cases (dark, blurry, occluded, rare categories)
    
    # Key check: activation range stability
    # Run calibration multiple times with different random seeds.
    # If the computed scale varies by >10%，your calibration set is too small
    # or not representative enough.
    pass

# REAL-WORLD BUG CASE:
# A team calibrated INT8 quantization on ImageNet validation set,
# achieved <0.3% accuracy drop. Deployed the model.
# Result: model accuracy on user-uploaded photos dropped 8%.
# Root cause: ImageNet photos are professionally shot (good lighting, centered objects).
# User photos are blurry, tilted, with random lighting.
# The activation ranges during inference were 3x wider than calibration predicted.
# Solution: re-calibrated using 1000 random user-uploaded photos from production logs.
```

## 7. TinyML / Edge AI 部署意义

**量化是 TinyML 的命脉**：

- MCU 的 SRAM 只有几百KB → 必须 INT8 才能装下模型
- 许多 MCU 没有 FPU（浮点运算单元）→ 必须整数量化才能推理
- ARM Cortex-M 的 SIMD 指令（如 SMLAD）一次处理4个INT16 → 量化后天然利用SIMD
- 能耗：INT8 MAC 能耗约为 FP32 的 1/20 → 电池设备至关重要

### 真实 MCU 量化部署的生存条件

- **ARM Cortex-M4 (STM32F407, 192KB SRAM, 168MHz, 无 FPU)**: 关键词检测模型 DS-CNN，INT8 量化后权重 14KB + 激活值 buffer 22KB = 36KB 内存占用。如果是 FP32，模型 56KB + 激活值 88KB = 144KB → 剩余仅 48KB 给音频采集 DMA buffer（需要 32KB）和应用代码 → 勉强可跑但无内存安全边际。INT8 后剩余 156KB → 可以加入回声消除算法（额外需要 40KB buffer）。

- **ESP32-S3 (512KB SRAM, 240MHz, Xtensa LX7)**: 人脸检测模型，INT8 per-channel 量化 → 模型 320KB + Tensor Arena 150KB = 470KB。但 ESP-IDF 的 Wi-Fi + BLE 协议栈已占用 ~180KB SRAM → 如果模型用了 FP32（模型 1.2MB）→ 直接超 SRAM 上限 3 倍，连程序都加载不了。ESP32 的 PSRAM (8MB) 可以存模型权重，但权重从 PSRAM 搬运到 SRAM 的延迟是 SRAM 内读取的 15x → 推理会慢一个数量级。

- **没有 FPU 的 MCU 如何推理**: ARM Cortex-M0/M0+、RISC-V RV32IMC 等没有浮点运算单元的 MCU 上，执行一次 FP32 乘法需要 ~70-100 个 CPU 周期（纯软件模拟 IEEE 754）。而 INT8 乘法只需 1 个周期（硬件乘法器）。这就是为什么在这些 MCU 上，量化不是"优化"而是"能否运行"的准入条件。

### 量化精度的真实约束

- **MCUNet on STM32F746 (320KB SRAM)**: INT8 量化后的模型：权重 168KB + 激活值峰值 196KB + TinyEngine runtime buffer 60KB = 424KB 理论需求 > 320KB 物理容量。通过"activation memory planning"（激活值缓冲区复用），将峰值从 196KB 压缩到 138KB → 总需求 366KB → 仍超 46KB。最后通过 int16→int8 的中间精度动态切换（非瓶颈层用 int8，瓶颈层用 int16 pooling），在不增加激活值的前提下解决了精度问题。

## 8. 常见误区

1. **"量化就是截断小数点"** — 错！量化包含 scale/zero_point 校准，是精密工程
2. **"INT8量化精度损失 <1% 是普遍的"** — 对 CNN 成立；对 Transformer/LLM 直接 INT8 可能掉>5%
3. **"QAT 一定比 PTQ 好"** — QAT 需要重训练，成本高；PTQ 对很多模型已经够用
4. **"Per-Tensor 和 Per-Channel 量化差别不大"** — 对激活值可能不大，但对权重量化，per-channel 可显著降低误差
5. **"量化后的推理一定更快"** — 需要硬件支持 INT8 指令（如 ARM NEON, NVIDIA INT8 TC）才能真正加速

### 生产环境量化 P0 级事故

6. **"量化前忘记 fuse BatchNorm → BN 的 scale 放大量化误差 10 倍"** — 这是量化部署中最常见的 P0 级 bug。训练时的 Conv-BN-ReLU 在推理时需要 fuse 为单个操作：`y = Conv_fused(x) = γ/σ * Conv(x) + (β - γ·μ/σ)`。如果没有 fuse，量化是分别对 Conv 输出（范围小而均匀）和 BN 输出（被 γ/σ 缩放后范围变了）做的，导致 BN 的 scale 因子直接放大 Conv 的量化误差。实际案例：ResNet50 在 TensorRT 中，未 fuse BN 时的 INT8 精度 Top-1 从 76.1% 跌到 68.4%（损失 7.7%）。fuse 后恢复到 75.9%（仅损失 0.2%）。

7. **"ONNX 导出 dynamic batch 设置错误 → 生产环境 batch=1 推理 OOM"** — 在 PyTorch 中导出 ONNX 时设置 `dynamic_axes={'input': {0: 'batch'}}`。如果 TensorRT 在 build engine 时用了 `opt_batch_size=32, max_batch_size=64`，但生产环境实际请求是 batch=1 → TensorRT 按 max_batch_size 预分配了 64-batch 的显存 → OOM。**正确做法**: `min_batch=1, opt_batch=1, max_batch=64`，让 TensorRT 为 batch=1 优化，通过 dynamic batching 在 runtime 攒 batch。

8. **"INT8 校准用了不具代表性的数据 → 激活值 range 被低估 → 线上长尾 case 崩"** — 校准集只包含"容易的"正常样本 → 得到的 scale 范围很窄 → 遇到困难样本时，大量激活值被 clamp 到 q_max/q_min → 信息丢失。最灾难的场景：校准用的是白天图片，但模型实际部署在夜间安防场景 → 红外补光下的激活值范围是白天的 3-5 倍 → 80% 的激活值溢出（saturation）。**排查方法**: 检查量化后的模型在推理时有多少比例的激活值达到了 q_max 或 q_min — 如果 >5%，说明校准不足，scale 范围太小。

9. **"为省事用了 per-tensor 量化 → per-channel 只需多存几十个 scale 值，但精度提升巨大"** — 一个 64 通道的卷积层，per-tensor 只存 2 个 float（scale + zero_point），per-channel 存 128 个 float。128 × 4 = 512 bytes，相对于几十 MB 的模型体量可以忽略。但对 MobileNet depthwise conv 这类单通道权重差异大的结构，per-tensor 到 per-channel 可以挽回 3-5% 的 Top-1 精度。如果因为觉得"开销大"而拒绝 per-channel → 这是在用 0.001% 的存储开销换取 5% 精度损失，属于典型的"捡芝麻丢西瓜"。

## 9. 面试问题

**Q1**: "为什么量化到 INT8 通常不掉精度，但 INT4 就容易掉精度？"

**A1**: 
- INT8 提供 256 个量化级别，对于大多数权重分布来说足够精细
- INT4 只有 16 个级别，量化间隔 $\Delta$ 变大 → 量化误差按 $\Delta^2/12$ 增大 → 指数级精度损失
- 另外，INT4 的硬件支持也不普及

**Q2**: "Per-Channel 量化为什么比 Per-Tensor 好？原理是什么？"

**A2**: 卷积层的不同输出通道（卷积核）可能捕捉完全不同的特征，权重分布差异很大（有的通道值大、有的小）。Per-Tensor 用一个 scale 会被"大值通道"拖累 → "小值通道"的量化解不够精细。Per-Channel 每个通道独立 scale → 解决了这个问题。

**Q3**: "SmoothQuant 是如何解决 LLM 激活值量化难题的？"

**A3**: LLM 的激活值存在巨大的异常值（outliers, 比平均值大100倍），导致直接量化激活值精度很差。SmoothQuant 的巧妙之处：将激活值中异常值的"幅度"通过数学等价变换转移到权重上（W × X = (W·diag(s)) × (diag(s)^{-1}·X)），权重一般分布均匀更适合承受额外range。

**Q4 (NVIDIA 面试真题)**: "你给客户做 PTQ INT8 量化。ResNet50 在 ImageNet 上 Top-1 只降了 0.3%，客户很满意。一个月后客户投诉说在夜间安防场景准确率暴跌 12%。请从量化误差传播的角度解释为什么会发生，并给出三套不同复杂度的解决方案。"

**参考答案**: 

**根因分析**: 这是一个典型的"校准数据分布偏移"问题。原始校准用了 ImageNet（白天自然光场景），得到的激活值 range 被低估。夜间红外补光场景下，激活值的分布完全不同：
- 红外图片的像素值集中在低灰度区域（暗），但部分高反射区域（人脸、车牌）有极高的像素值 → 激活值 range 是白天的 3-5 倍
- INT8 量化中，超出 scale 范围的值被 clamp 到 q_min/q_max → 这些激活值的信息完全丢失
- 量化误差逐层传播：第 3 层的 2% clamp rate → 第 6 层累积到 8% → 第 10 层累积到 25% → 分类头看到的是严重失真特征

**方案 A（快速的 1 天修复）**: 从夜间安防场景抓取 2000 帧真实图片作为校准集，重新跑 PTQ。用 Percentile(99.99) 校准容忍少量极端值。精度可恢复到损失 1-2%。

**方案 B（中期的 1 周方案）**: QAT 在混合数据集（白天+夜间各50%）上微调 5000 步。让模型学习在量化约束下同时适应两种光照条件。精度损失 <0.5%。

**方案 C（长期的架构级方案）**: 
- 用混合精度：前 3 层（对光照敏感的特征提取层）保留 FP16，中间层 INT8，最后分类头 INT8。前 3 层 FP16 只增加约 5% 的额外推理时间。
- 或引入光照自适应的动态量化：在推理时检测输入的平均亮度，动态选择为白天/夜间场景训练的量化参数。

**Q5 (字节跳动面试真题)**: "你们团队用 INT8 QAT 训练了一个 BERT 模型，在验证集上精度与 FP32 持平。部署到 TensorRT 后，用相同的测试数据跑，精度低了 1.8%。你排查了所有训练和推理 pipeline，发现 QAT 训练中 FakeQuantize 用的 round mode 是 'round-half-to-even'，而 TensorRT 的 INT8 kernel 用的是 'round-half-away-from-zero'。请解释为什么这个差异会导致 1.8% 的精度损失。"

**参考答案**: 

`round(2.5)`: 
- round-half-to-even: → 2 (IEEE 754 标准)
- round-half-away-from-zero: → 3 (常见硬件实现)

FakeQuantize 的前向计算（`round(x/s)`）中，round mode 差异导致：
1. **边界值的系统偏差**: 每个 `.5` 边界值在 QAT 中被 round-down 到偶数，在 TensorRT 中被 round-up。BERT 的大量 attention score（集中在 softmax 之后的 [0, 1] 区间）经过 scale 变换后，恰好有大量值落在 `.5` 附近 → 系统性的 ±1 量化偏差。
2. **累积效应**: BERT 有 12 层 Transformer，每层的量化偏差逐层累积。第 1 层的 ±1 偏差在 attention 中被 softmax 放大，传导到第 2 层已经变成 ±3 偏差，到最后一层累积为显著的 shift。
3. **softmax 的非线性放大**: attention score 的 softmax 是对数值敏感的非线性映射。例如：scores [2.0, 2.5, 3.0] → softmax → [0.19, 0.31, 0.50]。如果 2.5 被 round mode 偏差成 2 或 3 → softmax 输出变成 [0.24, 0.18, 0.58] 或 [0.16, 0.42, 0.42] → attention 权重分配完全不同 → 下游 token 的表示被改变。

**解决方案**: 
1. 在 QAT 训练时，FakeQuantize 中用与 TensorRT 一致的 round mode（可通过 `torch.round()` 的 behavior 或自定义 kernel）
2. 或：在 QAT 训练完成后，用目标推理引擎（TensorRT）跑一遍 calibration 验证集，对比 PyTorch 输出的一致性（而非仅看最终 accuracy number）

## 10. 本讲总结

量化的三个核心理解：
1. **公式**: $r = S(q - Z)$ — 用两个参数在浮点和整数之间架桥梁
2. **粒度**: Per-tensor < Per-channel < Per-group ← 精度递增，开销也递增
3. **流程**: PTQ（快但可能不准）→ QAT（慢但更准）→ 根据场景选择

量化回答了："能不能用更少的比特存储参数？"配合剪枝（更少的参数），两者叠加 = 超高效模型。

本讲的量化知识是后续 LLM 量化（AWQ, GPTQ, SmoothQuant）和 TinyML 部署的基础。

## 11. 工业落地 Checklist

| 检查项 | 说明 | 不做的后果 |
|--------|------|-----------|
| BN 融合 | 量化前必须 fuse Conv+BN 到单个操作，否则 BN 的 scale 会放大量化误差 | 精度损失 7-10%（ResNet50从76.1%跌到68.4%） |
| 校准数据分布匹配 | 校准集必须覆盖线上推理的所有场景（光照/角度/类目/序列长度） | 夜间长尾 case 准确率暴跌 12% |
| 校准数据多样性 | 至少包含 20% hard cases（极端光照、遮挡、罕见类目），不只选 easy samples | 激活值 range 被低估，线上覆盖率不足 |
| 激活值 saturation 检查 | 上线前检查量化模型推理时 activation 达到 q_max/q_min 的比例，>5% 说明校准不足 | 信息丢失随时间累积，最终结果偏差 |
| Per-Channel 量化 | 对卷积层权重量化默认用 per-channel（512 字节开销可忽略），depthwise conv 必须 per-channel | 精度损失 3-5% 仅因省了 512 字节 |
| Round mode 一致性 | QAT 训练中的 round mode 必须与目标推理引擎一致（round-half-to-even vs round-half-away-from-zero） | 训练精度 OK 但部署后精度掉 1.8%（BERT案例） |
| INT8 指令集验证 | 确认目标硬件支持 INT8 SIMD(NEON)或INT8 Tensor Core(NVIDIA) | 量化后推理反而比 FP32 慢 |
| 动态量化 fallback | 对没有 INT8 指令的硬件，准备 FP16/FP32 fallback 路径 | 用户设备不支持 INT8 → app crash 或黑屏 |
| Scale/ZP 精度 | Scale 用 FP32 存储（不要用 FP16），否则 scale 本身的量化误差会传导至所有权重 | 双重量化误差 → 精度雪崩 |
