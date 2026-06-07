# Lecture 02: 深度学习基础 - 从神经元到效率指标

## 1. 本讲核心问题

> 在开始"压缩"之前，我们先搞清楚要压缩的是什么。如何精确衡量一个模型的"成本和性能"？

## 2. 通俗解释

**生活类比**: 你要搬家（部署模型），得先知道：
- 你有多少东西要搬（参数数量）
- 搬一次要多累（计算量/FLOPs）
- 搬一次要多久（延迟/Latency）
- 搬家的车能装多少（内存/带宽）
- 油费要多少（能耗）

没搞清这些就乱扔东西，可能把电视扔了却留了一堆纸箱子。

## 3. 关键公式

### 3.1 一图看懂各项指标

| 指标 | 定义 | 单位 | 类比 |
|------|------|------|------|
| **#Params** | 模型参数总数 | 个/M | 搬家物品总量 |
| **Model Size** | 模型占用存储空间 | MB/GB | 搬家箱子体积 |
| **FLOPs** | 浮点运算次数 | M/G/T | 搬一次消耗的体力 |
| **MACs** | 乘加操作次数 | M/G/T | ~2×FLOPs |
| **Latency** | 单次推理时间 | ms | 单趟搬运耗时 |
| **Throughput** | 单位时间处理量 | samples/s | 一天能搬多少趟 |
| **Peak Memory** | 推理时内存峰值 | MB/GB | 最大车厢占用 |
| **Energy** | 单次推理能耗 | mJ | 单趟油费 |

### 3.2 各层参数量计算

**全连接层**:
$$\text{Params} = C_{in} \times C_{out}$$

**卷积层**:
$$\text{Params} = K_h \times K_w \times C_{in} \times C_{out}$$

**Transformer Attention**:
$$\text{Params}_{QKV} = 3 \times d_{model} \times d_{model}$$

### 3.3 FLOPs 计算

**全连接层**:
$$\text{FLOPs} = 2 \times C_{in} \times C_{out}$$

**卷积层**:
$$\text{FLOPs} = 2 \times K_h \times K_w \times C_{in} \times C_{out} \times H_{out} \times W_{out}$$

> 乘以2是因为每个 MAC 算作两个 FLOP（一次乘+一次加）

## 4. 公式背后的直觉

**为什么区分 FLOPs 和 Latency？**

想象你要搬100个箱子。FLOPs说的是"总共需要搬100次"。但Latency说的是"在一条窄走廊里搬100次"和"在宽阔的路上一趟搬10个"的差别。

- **FLOPs**: 理论工作量（纯计算）
- **Latency**: 实际耗时（计算 + 内存搬运 + 调度开销）

> 同一个模型的 FLOPs 不变，但在不同硬件上 Latency 可以差10倍！

## 5. 工业界用途

| 场景 | 关键关注指标 | 原因 |
|------|-------------|------|
| 云端推理服务 | Throughput | 批处理最大化，服务成千上万请求 |
| 手机端实时AR | Latency | 每帧<16ms，否则用户头晕 |
| 智能手表 | Energy + Memory | 电池小，内存<512KB |
| 自动驾驶 | Latency (P99) | 尾延迟不能超标，安全攸关 |
| 训练集群 | Bandwidth + Memory | 模型可能100B+参数 |

## 6. PyTorch 实现思路

### 完整模型分析脚本

```python
import torch
import torch.nn as nn
import time
import numpy as np

def count_parameters(model):
    """统计参数量和模型大小"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = total * 4 / (1024**2)
    return total, trainable, size_mb

def measure_latency(model, input_shape, device='cpu', warmup=10, repeat=100):
    """测量推理延迟"""
    model = model.to(device).eval()
    dummy = torch.randn(*input_shape).to(device)
    
    # Warmup
    for _ in range(warmup):
        _ = model(dummy)
    
    # 如果是GPU，同步
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(repeat):
        _ = model(dummy)
        if device == 'cuda':
            torch.cuda.synchronize()
    end = time.perf_counter()
    
    avg_latency = (end - start) / repeat * 1000  # ms
    return avg_latency

# 使用示例
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 10)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = TinyCNN()
total, trainable, size_mb = count_parameters(model)
latency_cpu = measure_latency(model, (1, 3, 32, 32), 'cpu')

print(f"参数量: {total:,} | 模型大小: {size_mb:.2f} MB")
print(f"CPU推理延迟: {latency_cpu:.2f} ms")
```

## 7. TinyML / Edge AI 部署意义

**理解指标 = 理解硬件约束**：

- MCU (如 ARM Cortex-M4): SRAM < 512KB, Flash < 2MB, 算力 ~200 MOPS
- 手机 NPU: 内存 4-8GB, 算力 5-15 TOPS
- GPU (A100): 内存 80GB, 算力 312 TFLOPS (FP16)

差距是 10^4 - 10^6 倍的！所以必须知道模型的每个参数和每个 FLOP "值多少钱"。

## 8. 常见误区

1. **"参数量少 = 速度快"** — 不一定。Depthwise Conv 参数少但内存访问密集，在GPU上可能更慢
2. **"FLOPs = Latency"** — 错。内存访问、kernel launch、流水线停顿都是延迟来源
3. **"看平均数就够了"** — 错。实时系统关心的是 P99/P99.9 尾延迟
4. **"CPU和GPU推理速度直接除以性能倍数"** — 错。小 batch 下 CPU 可能反而更快（无 kernel launch 开销）

## 9. 面试问题

**问题**: "给你一个 ResNet50，如何快速估计它能在目标设备上跑多快？"

**参考思路**:
1. 导出 ONNX，用 onnxruntime 或 TensorRT 跑 benchmark
2. 关注的不只是总时间，还有逐层分析（哪个层是瓶颈）
3. 看内存占用峰值：能否装进设备的可用内存？
4. 考虑 batch size：设备通常 batch=1，GPU 的大 batch 优势丧失
5. 检查算子兼容性：目标设备是否支持所有操作？

## 10. 本讲总结

在深入剪枝、量化等高级技术之前，必须掌握**模型的"体检报告"**：
- 参数量/模型大小 → 存储成本
- FLOPs/MACs → 理论计算量
- Latency/Throughput → 真实性能
- Peak Memory → 内存成本
- Energy → 功耗成本

每个指标在工业界有不同的优先级，取决于你的部署场景。
