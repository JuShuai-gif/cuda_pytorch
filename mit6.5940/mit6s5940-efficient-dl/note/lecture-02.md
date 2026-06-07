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

### 真实生产案例与成本

- **OpenAI GPT-4 推理集群**: 每百万 token 输出约消耗 \$0.06（FP16推理），如果所有请求都用 FP32 则成本翻倍到 \$0.12，按每日 1000 亿 token 计算，FP16 相比 FP32 每日节省 \$600万。这是为什么 FP16/BF16 推理是 LLM 部署的绝对标准。
- **特斯拉 Autopilot FSD**: 自动驾驶芯片 FSD Chip 上 2 颗 NPU 共 144 TOPS（INT8），功耗仅 72W。如果运行 FP32 需要 500+ TOPS 等效算力，功耗将超 300W → 显著影响续航。INT8 量化使车载推理成为可能。
- **腾讯微信语音识别**: ARM 端上的流式 ASR 模型，从 FP32 到 INT8 量化 → 推理延迟从 340ms 降至 85ms（加速 4x），功耗从 820mW 降到 210mW。每年节省的服务器带宽成本约 ¥800万（更多在端侧完成，不再上传音频到云）。

### 指标优先级的业务视角

相同模型在不同业务下，优化的指标完全不同：

| 业务场景 | 第一优先级 | 第二优先级 | 为什么 |
|----------|-----------|-----------|--------|
| 短视频推荐（信息流） | Throughput | Memory | 海量请求，GPU 利用率决定成本 |
| 直播实时美颜 | Latency | Energy | 每帧 16ms 硬 deadline，超时就掉帧 |
| AR 眼镜导航 | Energy | Memory | 电池容量仅 300mAh，功耗超 500mW 则续航 < 2h |
| 自动驾驶紧急制动 | P99.99 Latency | — | 200ms 延迟 = 5.5m 制动距离差距（100km/h） |

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

### 生产环境 Latency 测量的正确姿势

```python
import torch
import time
import numpy as np
from typing import List, Tuple

def production_latency_benchmark(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    device: str = 'cuda',
    warmup: int = 50,
    repeat: int = 500,
) -> dict:
    """Production-grade latency measurement that avoids common pitfalls.
    
    Critical rules for accurate GPU timing:
    1. ALWAYS use torch.cuda.Event for GPU timing (CPU-side `time.perf_counter`
       does NOT wait for GPU to finish — it measures host time, not device time)
    2. ALWAYS synchronize after each iteration (not just at end) to prevent
       overlapping executions that inflate throughput numbers
    3. Skip the first several warmup iterations (CUDA JIT compilation,
       cudnn auto-tune, memory allocation warmup)
    4. Report p50, p99, p99.9, not just mean — tail latency kills user experience
    5. Use separate CUDA streams if you need to measure concurrent execution
    """
    model = model.to(device).eval()
    dummy = torch.randn(*input_shape, device=device)
    
    # Use CUDA events for device-accurate timing
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    
    # Warmup: more iterations on GPU to trigger all auto-tuning
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(dummy)
    torch.cuda.synchronize()
    
    timings: List[float] = []
    with torch.no_grad():
        for _ in range(repeat):
            starter.record()
            _ = model(dummy)
            ender.record()
            torch.cuda.synchronize()
            timings.append(starter.elapsed_time(ender))
    
    timings = np.array(timings)
    return {
        'mean_ms': round(np.mean(timings), 3),
        'std_ms': round(np.std(timings), 3),
        'p50_ms': round(np.percentile(timings, 50), 3),
        'p99_ms': round(np.percentile(timings, 99), 3),
        'p99.9_ms': round(np.percentile(timings, 99.9), 3),
        'min_ms': round(np.min(timings), 3),
        'max_ms': round(np.max(timings), 3),
    }

# Common error: measuring latency with CPU clock on GPU operations
# WRONG:
# start = time.perf_counter()
# _ = model(dummy)  # GPU may still be executing!
# elapsed = time.perf_counter() - start  # This measures HOST time, not DEVICE time
```

## 7. TinyML / Edge AI 部署意义

**理解指标 = 理解硬件约束**：

- MCU (如 ARM Cortex-M4): SRAM < 512KB, Flash < 2MB, 算力 ~200 MOPS
- 手机 NPU: 内存 4-8GB, 算力 5-15 TOPS
- GPU (A100): 内存 80GB, 算力 312 TFLOPS (FP16)

差距是 10^4 - 10^6 倍的！所以必须知道模型的每个参数和每个 FLOP "值多少钱"。

### 各级硬件 FLOPs/能耗深度对比

| 硬件平台 | 算力 | 内存带宽 | 典型功耗 | 1 TOPS 成本 (能耗) | 1 TOPS 成本 (价格) |
|----------|------|----------|----------|-------------------|-------------------|
| **Cortex-M4 (STM32F4)** | 0.0002 TOPS | 0.1 GB/s | 100mW | 500,000 mJ | \$0.1 |
| **Cortex-M7 (STM32H7)** | 0.001 TOPS | 0.5 GB/s | 300mW | 300,000 mJ | \$0.5 |
| **Jetson Nano** | 0.47 TOPS | 25.6 GB/s | 5W | 10,638 mJ | \$99/卡 |
| **手机 NPU (A16)** | 17 TOPS | 51.2 GB/s | 3W | 176 mJ | N/A |
| **RTX 4090** | 82.6 TOPS (FP16) | 1008 GB/s | 450W | 5,448 mJ | \$1,599/卡 |
| **H100 (FP8)** | 1979 TOPS | 3350 GB/s | 700W | 354 mJ | \$30,000/卡 |

### 关键洞察：内存墙 (Memory Wall)

在 MCU 级别的硬件上，**读取 1 个 FP32 权重的能耗 > 执行 1 次 MAC 运算的能耗**：
- SRAM 读取 (8KB): 10 pJ
- INT8 MAC: 0.2 pJ
- 差距: **50 倍**！

也就是说，在 MCU 上运行神经网络，瓶颈从来不是"算不快"，而是"读太贵"。这也解释了为什么剪枝（减少权重数量）和量化（减少每个权重的比特数）在边缘设备上效果如此显著 — 它们减少的是内存访问，而非仅仅减少计算。每次剪掉一个权重，你省下的是"从 Flash/SRAM 搬运它到 ALU"的能耗，这比省下的计算能耗多 10-50 倍。

## 8. 常见误区

1. **"参数量少 = 速度快"** — 不一定。Depthwise Conv 参数少但内存访问密集，在GPU上可能更慢
2. **"FLOPs = Latency"** — 错。内存访问、kernel launch、流水线停顿都是延迟来源
3. **"看平均数就够了"** — 错。实时系统关心的是 P99/P99.9 尾延迟
4. **"CPU和GPU推理速度直接除以性能倍数"** — 错。小 batch 下 CPU 可能反而更快（无 kernel launch 开销）

### 生产环境高频事故

5. **"用 time.perf_counter() 测 GPU 延迟"** — 这是生产环境最常见的低级错误。`time.perf_counter()` 是纯 CPU 侧计时器，它测量的是"CPU 把任务提交给 GPU"的时间，而不是"GPU 执行完任务"的时间。用 `time.perf_counter()` 测 GPU 推理 → 你会以为延迟是 0.3ms，实际 GPU 花了 3ms。**必须用 `torch.cuda.Event` 或 `torch.cuda.synchronize()` 做 GPU 侧计时**。

6. **"Model size (MB) = total_params × 4"** — 这忽略了 optimizer states（训练时）、gradients（训练时）、中间激活值（推理时）和 CUDA context 开销。推理时的峰值内存通常 = 权重 + 激活值 + runtime buffer + cuDNN workspace，而非仅仅权重大小。例如 ResNet50 权重仅 100MB，但 batch=32 时峰值内存可达 1.2GB。

7. **"FLOPs 小了 50%，推理成本就省 50%"** — 云厂商按 GPU 使用时长（instance-hours）计费，不是按 FLOPs。如果你的模型 FLOPs 减半但 latency 不变（因为 memory-bound），你仍然需要同样多的 GPU 实例，成本不变。只有真正降低了 latency/提高了 throughput，才能减少实例数 → 降低成本。

## 9. 面试问题

**Q1**: "给你一个 ResNet50，如何快速估计它能在目标设备上跑多快？"

**参考思路**:
1. 导出 ONNX，用 onnxruntime 或 TensorRT 跑 benchmark
2. 关注的不只是总时间，还有逐层分析（哪个层是瓶颈）
3. 看内存占用峰值：能否装进设备的可用内存？
4. 考虑 batch size：设备通常 batch=1，GPU 的大 batch 优势丧失
5. 检查算子兼容性：目标设备是否支持所有操作？

**Q2 (NVIDIA 面试真题)**: "你在 A100 上跑一个 ResNet50，发现 batch=1 时 GPU 利用率只有 12%。但 batch=64 时利用率达到 92%。请从 GPU 微架构角度解释为什么，并给出优化方案。"

**参考答案**: A100 有 108 个 SM（流式多处理器），每个 SM 有 4 个 warp scheduler。当 batch=1 时：
- 单个 batch 的矩阵乘法尺寸太小（比如最后一层 2048×1000），无法填满所有 SM — 大量 SM 空转
- 每个 kernel launch 的 overhead（~5-10μs）在 batch=1 时占比巨大
- 内存带宽利用不充分：A100 HBM2e 带宽 2TB/s，但 batch=1 时的实际带宽利用率不到 20%（连续请求太少）

**优化方案**: 
1. Dynamic batching — 在服务端攒多个请求一起推理（代价是增加排队延迟）
2. Kernel fusion — 把多个小 kernel 融合成一个大 kernel（如 Conv+BN+ReLU），减少 kernel launch 开销
3. MPS (Multi-Process Service) — 让多个推理进程共享 GPU context，提升 SM 利用率
4. 用 TensorRT 的 tactical engine builder 做 auto-tuning（它会自动尝试不同的 kernel 实现和 launch 策略）

**Q3 (字节跳动面试真题)**: "你需要在手机上部署一个实时视频分割模型（30fps, 每帧 33ms）。你的第一次实现延迟是 48ms/帧。你有两套优化方案：(A) 对模型做 INT8 量化，预期延迟降至 28ms。(B) 用更高效的预处理管线，将输入处理从 12ms 降到 4ms，总延迟降至 40ms。你会选哪个？为什么？"

**参考答案**: 选 A。原因：

1. **INT8 量化是"结构性优化"**，它不仅降低延迟还降低功耗（INT8 MAC 能耗是 FP32 的 1/20），可以延长手机电池续航。这是方案 B 永远做不到的。

2. **预处理优化边际收益递减** — 你已经把预处理从 12ms 优化到 4ms（67% 减少），但下一次再优化的空间很小（最多再省 4ms）。而量化之后还能进一步剪枝、蒸馏，优化天花板更高。

3. **风险权衡** — 量化有精度损失风险（通常 <1%），但预处理优化无精度风险。如果你需要先上线一个版本保证业务，B 是短期安全方案。但如果目标是长期迭代，A 是正确方向。**面试官真正想听到的是：你可以通过 A+B 组合（量化 + 多线程并行预处理管线）达到比单独任何一个更好的效果。**

但也要清醒：如果量化后精度损失 >3%（某些敏感分割模型），那 A 方案就不可行了。关键在于上线前做好精度评估。

**Q4 (快手 面试真题)**: "你负责一个 1000 万 DAU 的短视频 app 的推荐模型推理服务。目前 3000 台 GPU 服务器，每台日成本 \$40。CTO 要求明年成本降 40% 但不影响推荐质量。请设计一个完整的优化路线图。"

**参考答案**: 

1. **阶梯优化 (0-3 个月，快速见效)**: 
   - 所有模型 INT8 量化 + TensorRT 部署 → 吞吐提升 2-3x → GPU 需求降至 ~1200 台 → 年省 \$1300万
   - Dynamic batching 合并小请求 → 再提升 20-30% 吞吐

2. **中阶优化 (3-6 个月，架构升级)**:
   - 通道剪枝 + 知识蒸馏 → 模型参数量降低 60%+ → 再减少 30% GPU
   - 热点模型改为缓存计算（embedding lookup 的 top-k 相似度预先计算）

3. **高阶优化 (6-12 个月，系统级)**:
   - Cascade 推理：轻量模型先过一遍（过滤 80% 的简单 case），困难 case 才用大模型 → 总计算量降低 50%
   - 早停 (Early exit) 机制：简单样本在浅层就输出结果，不跑完整模型

4. **必须同步做的事**: 
   - 建立精度回归监控：每次量化/剪枝后自动跑线上流量回放对比
   - P99 延迟不升高 → 用户体验不受影响
   - 分层 A/B 实验：1%→10%→50%→100% 逐步放量

**总结**: 通过这个四阶段优化，可以在 12 个月内将 GPU 从 3000 台降至约 1200 台，且推荐质量不降低。关键是"量化→剪枝→系统级"的递进策略，而非一步到位。

## 10. 本讲总结

在深入剪枝、量化等高级技术之前，必须掌握**模型的"体检报告"**：
- 参数量/模型大小 → 存储成本
- FLOPs/MACs → 理论计算量
- Latency/Throughput → 真实性能
- Peak Memory → 内存成本
- Energy → 功耗成本

每个指标在工业界有不同的优先级，取决于你的部署场景。

## 11. 工业落地 Checklist

| 检查项 | 说明 | 不做的后果 |
|--------|------|-----------|
| GPU 侧计时验证 | 用 `torch.cuda.Event` 或 `torch.cuda.synchronize()` 计时，严禁用 `time.perf_counter()` 测 GPU 延迟 | 报告 0.3ms 延迟，实际 3ms，误导优化方向 |
| 尾延迟监控 | 上线后必须监控 P99/P99.9 延迟，不能只看 mean/median | 平均 10ms 但 P99.9 500ms → 用户体验灾难 |
| 峰值内存安全边际 | 推理峰值内存 < 硬件可用内存 × 80% | 偶发 OOM，进程被 kill，无日志可查 |
| 激活值内存核算 | 估算推理时激活值 + 权重 + runtime buffer 的总和，不仅看权重大小 | 权重 100MB 但激活值 800MB → 内存超限 |
| 预热充分 | GPU 推理前至少 50 次 warmup，触发所有 JIT 编译和 cudnn auto-tune | 首次推理延迟是稳定状态的 10-50x → 超时 |
| 算子兼容性矩阵 | 逐 op 交叉检查目标推理引擎的支持列表 | 模型导出成功但包含不支持的 op → 推理 crash 或 CPU fallback |
| 校准数据分布匹配 | 量化校准数据必须与线上推理数据同分布 | 离线精度 OK 但线上长尾 case 崩 |
| 内存带宽 vs 计算瓶颈分类 | 确定模型是 compute-bound 还是 memory-bound，对症优化 | memory-bound 模型去优化 FLOPs 是无效投入 |
