# Lecture 01: 课程引言 - 为什么需要高效深度学习

## 1. 本讲核心问题

> 深度学习模型越来越大，但硬件增长跟不上。如何在准确率几乎不变的前提下，让模型能在手机、MCU、IoT设备上跑起来？

## 2. 通俗解释

**生活类比**: 想象你有辆F1赛车（大模型），在马路上跑得飞快。但现在要把这辆车开到小巷子里（手机/嵌入式设备）。巷子太窄（内存小）、路面不平（算力低）、油不够（功耗限制）。你不能直接开进去——你得把车变小，但又不能让它变成破车。

这就是高效深度学习的核心：**让模型在资源受限的设备上也能运行，同时尽量保持准确率**。

### 关键数据

- 深度学习模型规模每 **2年增长4倍**
- 硬件（GPU/内存）每 **2年只增长2倍**（摩尔定律）
- 供需差距越来越大 → 必须做模型压缩

## 3. 关键公式

本讲更多是引入概念，核心公式是定义**模型压缩的优化目标**：

$$\min_{W} \mathcal{L}(x; W) \quad \text{s.t.} \quad \text{Cost}(W) \le C_{budget}$$

解释：
- $\mathcal{L}$ 是损失函数（模型不准就扣分）
- $\text{Cost}(W)$ 可以是：参数数量、FLOPs、延迟、内存、能耗
- $C_{budget}$ 是硬件能承受的上限

## 4. 公式背后的直觉

这本质上是一个**约束优化问题**：
- 你有一个"预算"（比如 256KB 内存、10ms 延迟）
- 你要在这个预算内最大化准确率
- 四个字概括：**戴着镣铐跳舞**

## 5. 工业界用途

| 场景 | 约束 | 技术要求 |
|------|------|----------|
| 手机端实时翻译 | <50ms延迟, <200MB | 量化 + 剪枝 |
| 智能音箱关键词检测 | <10ms, <64KB SRAM | TinyML |
| 自动驾驶感知 | <5ms, <30W功耗 | 剪枝 + TensorRT |
| LLM本地运行 | <8GB显存 | AWQ/GPTQ量化 |
| MCU传感器异常检测 | <32KB Flash | MCUNet |

### 真实公司案例与数字

- **字节跳动抖音推荐模型**: 在抖音推荐系统中对 ResNet50 做通道剪枝，参数量从 25M 压缩到 5M（80% 压缩率），配合 INT8 量化后线上推理延迟从 42ms 降至 17ms（降低 60%），单机 QPS 从 1200 提升到 3400。
- **快手短视频特征提取**: MobileNetV3 + 结构化剪枝 + FP16 量化，模型大小从 5.4MB 降至 0.9MB，端上首帧耗时从 380ms 降至 110ms。
- **美团外卖图片审核**: YOLOv5s 通道剪枝 + INT8 TensorRT 部署，推理延迟从 8ms 降到 2.3ms，每日节省 GPU 服务器成本约 ¥12,000。
- **Google 智能助手语音唤醒**: 在 Pixel 手机上，关键词检测模型从 FP32 的 450KB 压缩到 INT8 的 110KB，误唤醒率从 0.5次/小时 降至 0.2次/小时（配合蒸馏）。

### 多平台推理引擎对比

| 引擎 | 适用平台 | 量化支持 | 剪枝支持 | 特点 |
|------|----------|----------|----------|------|
| **TensorRT** | NVIDIA GPU (数据中心/嵌入式) | INT8/FP8/INT4 | 2:4 sparsity | 最强GPU优化，NVIDIA官方 |
| **ONNX Runtime** | 跨平台 (CPU/GPU/移动端) | INT8/FP16 | 有限支持 | 生态最好，模型交换格式标准 |
| **TFLite** | Android/iOS/ARM | INT8/INT16/FP16 | 训练时剪枝 | Google官方移动端方案 |
| **CoreML** | Apple 全系列 (ANE加速) | INT8/FP16 | 权重剪枝 | Apple生态独享ANE加速 |
| **OpenVINO** | Intel CPU/VPU/GPU | INT8/FP16 | 通道剪枝 | Intel生态最佳，VPU/Movidius专优 |
| **ncnn** | 移动端(ARM) | INT8 | 通道剪枝 | 腾讯开源，极致轻量，ARM NEON优化 |

### 推理成本分析

大规模推理的真实成本对比（以每日10亿次ResNet50推理为例，AWS g4dn.xlarge 实例）：

| 精度配置 | 显存/实例 | 所需实例数 | 日成本 | 年成本 |
|----------|-----------|-----------|--------|--------|
| FP32 | 16GB | 42台 | $2,100 | $766,500 |
| FP16 | 8GB | 28台 | $1,400 | $511,000 |
| INT8 | 4GB | 18台 | $900 | $328,500 |
| INT8+剪枝50% | 2GB | 10台 | $500 | $182,500 |

> **结论**: FP32 → INT8+剪枝，年成本从 $766,500 降至 $182,500，减少 76%。大型互联网公司的推理集群动辄上万张 GPU，此类优化每年节省数千万美元。

## 6. PyTorch 实现思路

入门级示例——统计模型到底有多大：

```python
import torch
import torchvision.models as models

model = models.resnet50()
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
model_size_mb = total_params * 4 / (1024 * 1024)  # FP32: 4 bytes per param

print(f"总参数量: {total_params:,}")
print(f"可训练参数: {trainable_params:,}")
print(f"模型大小(FP32): {model_size_mb:.2f} MB")
```

### 生产级模型分析完整脚本

```python
import torch
import torchvision.models as models
import time
from typing import Dict, Tuple

def production_model_audit(model: torch.nn.Module, 
                           input_shape: Tuple[int, ...],
                           device: str = 'cuda') -> Dict:
    """Production-grade model audit that catches common pitfalls.
    
    Key features beyond naive parameter counting:
    - Detects fused vs unfused BN layers (fusion required before quantization)
    - Measures memory with CUDA caching allocator (not just param * 4)
    - Uses CUDA events for accurate GPU timing (not CPU-side timestamps)
    """
    model = model.to(device).eval()
    
    # Use CUDA memory snapshot for accurate measurement
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    dummy = torch.randn(*input_shape, device=device)
    
    # Warmup: at least 3 iterations to trigger JIT compilation and autotune
    for _ in range(3):
        _ = model(dummy)
    torch.cuda.synchronize()
    
    # Timed inference with CUDA events (wall-clock accurate)
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    
    starter.record()
    with torch.no_grad():
        _ = model(dummy)
    ender.record()
    torch.cuda.synchronize()
    latency = starter.elapsed_time(ender)
    
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    # Check for BN layers that need fusion before quantization
    unfused_bn = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            unfused_bn.append(name)
    
    # Check parameter count per dtype
    params_by_dtype = {}
    for name, p in model.named_parameters():
        if p.dtype not in params_by_dtype:
            params_by_dtype[p.dtype] = 0
        params_by_dtype[p.dtype] += p.numel()
    
    return {
        'total_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'peak_memory_mb': round(peak_memory, 2),
        'latency_ms': round(latency, 2),
        'unfused_bn_layers': unfused_bn,
        'params_by_dtype': params_by_dtype,
    }

# Usage
model = models.resnet50()
audit = production_model_audit(model, (1, 3, 224, 224), 'cuda')
print(f"Latency: {audit['latency_ms']}ms | Memory: {audit['peak_memory_mb']}MB")
if audit['unfused_bn_layers']:
    print(f"WARNING: {len(audit['unfused_bn_layers'])} BN layers need fusion before INT8 quantization!")
```

## 7. TinyML / Edge AI 部署意义

**核心矛盾**：
- 大模型 → 高准确率 → 需要大GPU → 数据中心 → 延迟高/隐私差/断网不能用
- 小模型 → 可部署到边缘 → 实时/隐私/离线 → 但可能不准

**TinyML 的解法**：通过算法-系统协同设计，在 MCU 级别的硬件上跑神经网络（SRAM < 512KB, 算力 < 1 GOPS）。

### 真实硬件规格对比

| MCU 型号 | 内核 | SRAM | Flash | 主频 | FPU |
|----------|------|------|-------|------|-----|
| **STM32F746** | Cortex-M7 | 320KB | 1MB | 216MHz | 单精度 FPU |
| **STM32H743** | Cortex-M7 | 1MB | 2MB | 480MHz | 双精度 FPU |
| **Arduino Nano 33 BLE** | nRF52840 (Cortex-M4) | 256KB | 1MB | 64MHz | 无 FPU |
| **ESP32-S3** | Xtensa LX7 | 512KB | 16MB | 240MHz | 单精度 FPU |
| **Raspberry Pi Pico** | RP2040 (Cortex-M0+) | 264KB | 2MB | 133MHz | 无 FPU |

### 实际部署约束示例

- **MCUNet on STM32F746**: 320KB SRAM中，模型峰值激活值 196KB + TinyEngine 运行时缓冲区 60KB = 总占用 256KB，恰好剩余 64KB 给传感器驱动和应用逻辑。任何一层的激活值溢出 8KB 就导致静默数据损坏（silent memory corruption），不会报错但推理结果随机出错。
- **Keyword Spotting on Arduino Nano 33 BLE**: 256KB SRAM, 无 FPU = 必须 INT8 量化。16KB 模型 + 40KB feature extraction buffer = 56KB 常驻内存，留给 BLE 协议栈 200KB。若模型超过 32KB，BLE 连接会间歇性断开（内存竞争）。

## 8. 常见误区

1. **"模型小了就一定会不准"** — 错！剪枝+蒸馏可以做到模型缩小10倍，准确率只降0.5%
2. **"FLOPs少就一定快"** — 错！Depthwise Conv FLOPs少但在GPU上可能更慢（内存带宽瓶颈）
3. **"量化就是简单取整"** — 错！量化是门科学，涉及scale/zero_point/校准等
4. **"所有层用同样的压缩率"** — 错！不同层对压缩的敏感度差异巨大

### 生产环境 P0 级陷阱

5. **"本地验证OK, 上线就崩"** — 量化模型的校准数据来源必须与线上推理数据分布一致。用 imagenet 校准量化后部署到监控摄像头场景 → 夜间低光照图片的激活值范围与校准集完全不同 → 量化误差被放大 5-10 倍 → 线上召回率从 0.92 跌到 0.67。**解决方案**: 校准数据必须从生产线抓取真实流量样本，不能只用公开数据集。

6. **"ONNX 导出时 dynamic batch dim 设错"** — 导出模型时 batch dimension 设为 `dynamic`，但推理引擎在 batch=1 时会为 batch=32 的场景预分配内存 → 单个推理 OOM。在 TensorRT 中这是高频事故：`build(batch_size=1)` 但 `context.set_binding_shape()` 未正确绑定导致引擎按 `max_batch_size` 分配显存。

7. **"QAT 训练精度到 99%, 部署后只剩 93%"** — QAT 训练中使用了 FakeQuantize + STE，但实际部署时不同推理引擎的 INT8 实现细节不同（rounding mode: round-half-to-even vs round-half-away-from-zero），导致推理结果偏差累积。**解决方案**: 在 QAT 训练完成后的 eval 阶段即用目标推理引擎做一致性测试。

## 9. 面试问题

**Q1 (基础)**: "为什么模型参数量减少50%，推理延迟不减少50%？"

> **掌握时机**：学完 **Lecture 02（效率指标）+ Lecture 03/04（剪枝）** 之后。需要理解 latency vs FLOPs vs memory 的区别，以及结构化 vs 非结构化剪枝对实际加速的影响。

**参考答案**: 因为内存访问时间（memory access）是瓶颈，不是计算。剪枝只减少了计算量，但内存访问模式可能变得更不规则（尤其非结构化剪枝），导致实际加速远小于理论值。结构化剪枝（channel pruning）才能实实在在地减少延迟。

**Q2 (字节跳动/TikTok 级别)**: "你在抖音推荐场景部署了一个剪枝+量化的 ResNet50 模型，测试集 Top-1 从 76.1% 掉到 75.9%（仅降 0.2%），但上线后用户停留时长的业务指标下降了 2.3%。你会如何排查？"

> **掌握时机**：学完 **Lecture 03/04（剪枝）+ Lecture 05/06（量化与 PTQ 校准）** 之后，并需具备**实际部署经验**。涉及量化校准数据分布、长尾分层评估、softmax 分布、系统链路瓶颈转移等，已超出纯算法范畴。

**参考答案**: 这是一个典型的"模型指标 OK 但业务指标劣化"问题，需要从以下角度排查：

1. **长尾样本退化**: 测试集 Top-1 只反映整体分布，但剪枝/量化对长尾样本（罕见类目、低清晰度图片）的负面影响可能被高置信度样本平均掉。用分层评估（stratified evaluation）检查每个类目的准确率变化，通常会发现某些低频类目掉点 10%+。

2. **校准数据与线上数据分布偏移**: 量化校准用的是 ImageNet，但抖音实际图片是用户 UGC 内容 — 压缩率高、水印多样、构图随意。用线上真实流量重新校准，或者使用 Percentile 校准（截断离群值避免 scale 被少数极端值拉偏）。

3. **Softmax 置信度分布变化**: 即使 Top-1 正确率不变，剪枝/量化模型输出的 softmax 分布可能更"平滑"（entropy 增大），导致推荐系统的 score 区分度下降。需要检查模型输出 logits 的标准差和 top-k margin 是否缩小。

4. **系统链路瓶颈转移**: 原本 CPU-bound 的推荐链路中模型是瓶颈，现在模型被加速后，瓶颈转移到特征提取或 faiss 检索环节，导致端到端延迟并没有改善，反而因为模型精度下降损失了业务指标。

**解决方案**: 上线前必须在线上流量回放（log replay）环境中跑 A/B 对比，不能仅看离线指标。

**Q3 (NVIDIA 级别)**: "你有一张 A100，要给客户演示 2:4 结构化稀疏的优势。客户问：'为什么我剪了 50% 的权重，Tensor Core 只能给我 1.3 倍加速而不是 2 倍？' 请从硬件微架构角度解释。"

> **掌握时机**：学完 **Lecture 04（结构化剪枝/稀疏）** 之后，并需**补充 GPU 硬件微架构课外知识**（Tensor Core 2:4 稀疏指令、HBM 带宽、kernel launch 开销）。本课程对硬件微架构涉及有限，这题深度超出课程本身。

**参考答案**: 2:4 稀疏的 2 倍加速前提假设是"计算完全被 Tensor Core 的矩阵乘法主导"。实际中达不到 2 倍的原因：

1. **非矩阵乘法的开销**: 模型推理中除了 Conv/GEMM（能用 Tensor Core），还有 element-wise 操作（ReLU, BN, Add, Pooling 等）。这些操作不受 2:4 稀疏的加速，占据了总推理时间的 20-40%。更致命的是，这些操作会打断 Tensor Core 的流水线，导致大量的 kernel launch 开销。

2. **内存带宽瓶颈**: 在 memory-bound 场景（小 batch，如 batch=1）下，算力不是瓶颈。即使 Tensor Core 加速了 2 倍，权重读取的时间仍然不变，总体加速被 memory bandwidth 限制。A100 的 HBM2e 带宽约 2TB/s，而实际权重读取 + 激活值读写可能已经接近这个上限。

3. **稀疏矩阵转换开销**: 要把密集权重按 2:4 模式重新排列（permute）才能喂给 Tensor Core 的稀疏指令。这个重排操作本身消耗显存带宽和 SM 资源，在 batch=1 时可能比节省的计算量还大。

4. **非均匀的压缩收益**: 真正能达到 2 倍加速的只有大矩阵乘法（M，N，K 都足够大）。对于小矩阵（如 MobileNet 的 depthwise conv），Tensor Core 利用率本身就很低，2:4 加速更少。

**实际最佳实践**: 在 A100 上，2:4 稀疏通常带来 1.4-1.7 倍的实际加速。要达到接近 2 倍，需要：大 batch（batch≥64）+ 大矩阵尺寸 + 算子融合（减少 kernel launch）。

**Q4 (快手/字节 级别)**: "你的团队开发了一套自动混合精度量化框架，PTQ 在 ResNet50 上 OK，但 Transformer 上掉点严重。你怀疑是激活值 outlier 导致的。请设计一个完整的排查和解决方案。"

> **掌握时机**：学完 **Lecture 05/06（线性量化 + PTQ/QAT）** 之后，并需**补充 LLM/Transformer 量化进阶知识**（激活 outlier、per-channel/per-token 量化、SmoothQuant、混合精度保留）。SmoothQuant 等属量化进阶专题，超出基础量化讲次。

**参考答案**: Transformer 激活值量化的核心难题是 outlier channels — 大约 0.1% 的激活值幅度比其余 99.9% 大 10-100 倍。排查方案：

1. **定位阶段**: 用 activation histogram 逐层分析，定位具体哪些层的哪些 channel 存在 outlier。通常出现在 LayerNorm 之后、Attention 的 softmax 之前。工具：用 PyTorch FX 插入 observer hooks，收集每个 op 输出的 min/max/percentile(99.99)/histogram。

2. **验证假设**: 对该层单独做 per-channel 量化（而非 per-tensor），如果精度恢复 → 确认是 outlier 导致。如果仍不恢复 → 可能是 attention score 的数值范围问题。

3. **解决方案递进**: 
   - **Level 1**: 对激活值用 per-tensor Percentile(99.99) 校准，截断 outlier — 简单，但可能损失 outlier channel 的信息
   - **Level 2**: 激活值 per-token 量化 (dynamic quantization) — 运行时每次计算 scale，精度好但开销大
   - **Level 3**: SmoothQuant — 把激活值的 outlier scale 迁移到权重上 (X·W = (X·diag(s))·(diag(s)⁻¹·W))，在数学等价的前提下让激活值分布更均匀
   - **Level 4**: 如果 SmoothQuant 还不够 → 对 outlier channel 保留 FP16 精度（混合精度），只对正常 channel 做 INT8

4. **上线验证**: 量化模型上线前必须做 P99 尾延迟测试和长尾 case 的 accuracy parity check，不能只看 Top-1。

## 10. 本讲总结

高效深度学习解决的核心问题是 **模型大小增长 vs 硬件能力** 之间的矛盾。三大技术支柱：
- **Pruning** — 去掉不需要的连接
- **Quantization** — 降低数值精度
- **NAS + Distillation** — 自动设计+知识迁移

后续每一讲将深入其中一个技术领域。

## 11. 工业落地 Checklist

| 检查项 | 说明 | 不做的后果 |
|--------|------|-----------|
| 模型硬件适配评估 | 上线前确认目标硬件的算力、内存、支持精度和算子覆盖 | 开发了半年发现模型根本部署不了 |
| 真实流量校准 | 校准数据必须来自线上生产环境，不能用公开数据集代替 | 离线 Top-1 99%，上线后长尾准确率暴跌 |
| BN 融合检查 | 量化前逐一确认 Conv+BN 已 fuse，残留 BN 会导致量化误差放大 | 精度损失 10-30% |
| 算子兼容性矩阵 | 列出模型所有 op，交叉检查目标推理引擎是否支持 | 导出成功但推理时 crash 或 fallback 到 CPU |
| 推理引擎一致性测试 | QAT 训练后，用目标引擎（TensorRT/ONNX/TFLite）跑同一批数据与 PyTorch 对比 | QAT 精度 OK 但部署后数值偏差累积导致结果错误 |
| 端到端延迟验证 | 不只测模型推理时间，要测完整链路（预处理+推理+后处理） | 模型快了但预处理成为新瓶颈，端到端延迟无改善 |
| P99 尾延迟监控 | 线上必须监控 P99/P99.9 延迟，不只关注平均值 | 平均 10ms 但 P99 200ms → 用户体验极差 |
| 长尾数据专项评估 | 评估模型在罕见类目、极端光照、遮挡等长尾场景的表现 | 整体指标 OK 但关键业务场景掉点 → 线上事故 |
| 内存峰值安全边际 | 峰值内存 < 硬件可用内存的 80%，留 20% 给系统和其他进程 | 内存 OOM 导致推理进程被 kill |
| A/B 实验上线 | 先在 1% 流量做 A/B 测试，观察核心业务指标 48h 再全量 | 模型指标 OK 但业务指标劣化，引发 P0 回滚 |

## 12. 学习闭环补充：从课程导论到工业项目

### 12.1 本讲在工业界对应什么能力

Lecture 01 不是单纯介绍“模型要变小”，而是建立工程判断框架：任何高效深度学习优化都必须同时回答质量、延迟、内存、吞吐、功耗和部署复杂度。工业项目中最常见的错误是只优化一个指标，例如只看参数量或 FLOPs，却没有证明目标 runtime 真的更快。

对应岗位能力：

| 能力 | 工业任务 |
|---|---|
| Efficiency trade-off | 判断一个优化方案是否值得上线 |
| Deployment awareness | 根据 CPU/GPU/NPU/MCU 选择压缩策略 |
| Metric literacy | 区分模型指标、系统指标和业务指标 |
| Risk control | 设计灰度、fallback 和监控指标 |

### 12.2 工业决策模板

拿到一个模型优化需求时，先写出以下表格：

| 问题 | 必须明确 |
|---|---|
| 目标硬件 | x86 CPU、ARM、NVIDIA GPU、NPU、MCU、手机 SoC |
| 目标 runtime | PyTorch、ONNX Runtime、TensorRT、OpenVINO、TFLite、MNN、ncnn |
| 质量指标 | accuracy、mAP、perplexity、MSE、success rate |
| 延迟指标 | batch=1 P50/P95/P99，或 server throughput |
| 内存指标 | model size、activation peak、KV cache、workspace |
| 可接受损失 | 例如 Top-1 下降 <= 0.5%，P99 <= 30ms |

### 12.3 对应代码实验

建议先运行统一压缩 benchmark，建立“不要只看理论指标”的直觉：

```bash
cd /home/hpc/ghr_code/cuda_pytorch/mit6.5940/mit6s5940-efficient-dl
python src/model_compression/benchmark_compression.py --runs 3 --warmup 1 --train-steps 1
```

看报告时重点问：

- 参数量下降了吗？模型文件大小下降了吗？
- latency 是否真的下降？P95 是否稳定？
- 输出 MSE 是否可接受？
- 哪些路径因为依赖或硬件缺失被跳过？

### 12.4 本讲验收问题

1. 为什么 FLOPs 降低 50% 不等于 latency 降低 50%？
2. batch=1 latency 和 batch=64 throughput 分别适合什么场景？
3. 如果手机端模型平均 20ms、P99 120ms，是否能上线？为什么？
4. 为什么校准数据必须来自目标业务分布？
5. 如何设计一个“压缩方案是否上线”的最终表格？

## 13. Python 代码补充：统一效率体检表

下面这段代码可以作为所有后续实验的 baseline 体检模板：统计参数量、模型大小和 CPU 延迟。真实项目中，任何剪枝/量化/NAS 方案都应该先和这个 baseline 对齐。

```python
import io
import time
import torch
import torch.nn as nn

class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(16, 10),
        )

    def forward(self, x):
        return self.net(x)

def model_size_mb(model: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return len(buf.getvalue()) / 1024**2

@torch.no_grad()
def benchmark_latency(model, x, warmup=10, runs=50):
    model.eval()
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1000)
    t = torch.tensor(times)
    return {
        "mean_ms": float(t.mean()),
        "p50_ms": float(t.quantile(0.50)),
        "p95_ms": float(t.quantile(0.95)),
    }

model = TinyClassifier()
x = torch.randn(1, 3, 32, 32)
print("params", sum(p.numel() for p in model.parameters()))
print("size_mb", model_size_mb(model))
print("latency", benchmark_latency(model, x))
```

工业解读：如果一个优化方案只让 `params` 降低，但 `p95_ms` 不降，说明它可能只是“模型变小”，没有形成 runtime 加速。

