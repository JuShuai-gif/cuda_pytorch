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

## 7. TinyML / Edge AI 部署意义

**量化是 TinyML 的命脉**：

- MCU 的 SRAM 只有几百KB → 必须 INT8 才能装下模型
- 许多 MCU 没有 FPU（浮点运算单元）→ 必须整数量化才能推理
- ARM Cortex-M 的 SIMD 指令（如 SMLAD）一次处理4个INT16 → 量化后天然利用SIMD
- 能耗：INT8 MAC 能耗约为 FP32 的 1/20 → 电池设备至关重要

## 8. 常见误区

1. **"量化就是截断小数点"** — 错！量化包含 scale/zero_point 校准，是精密工程
2. **"INT8量化精度损失 <1% 是普遍的"** — 对 CNN 成立；对 Transformer/LLM 直接 INT8 可能掉>5%
3. **"QAT 一定比 PTQ 好"** — QAT 需要重训练，成本高；PTQ 对很多模型已经够用
4. **"Per-Tensor 和 Per-Channel 量化差别不大"** — 对激活值可能不大，但对权重量化，per-channel 可显著降低误差
5. **"量化后的推理一定更快"** — 需要硬件支持 INT8 指令（如 ARM NEON, NVIDIA INT8 TC）才能真正加速

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

## 10. 本讲总结

量化的三个核心理解：
1. **公式**: $r = S(q - Z)$ — 用两个参数在浮点和整数之间架桥梁
2. **粒度**: Per-tensor < Per-channel < Per-group ← 精度递增，开销也递增
3. **流程**: PTQ（快但可能不准）→ QAT（慢但更准）→ 根据场景选择

量化回答了："能不能用更少的比特存储参数？"配合剪枝（更少的参数），两者叠加 = 超高效模型。

本讲的量化知识是后续 LLM 量化（AWQ, GPTQ, SmoothQuant）和 TinyML 部署的基础。
