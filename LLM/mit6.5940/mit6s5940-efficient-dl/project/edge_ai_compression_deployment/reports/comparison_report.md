# 模型压缩流水线对比报告 - MIT 6.5940

**生成时间**: 2026-06-22 11:36:40

> 本报告由基准测试模块自动生成，汇总了模型压缩流水线各阶段的性能数据。

## 流水线架构

```mermaid
graph LR
    baseline[baseline]
    baseline --> pruned
    pruned[pruned]
    pruned --> quantized
    quantized[quantized]
    quantized --> distilled
    distilled[distilled]
    distilled --> final
    final[final]
```

## 性能对比总览

| 阶段 | 模型 | 精度 (%) | 参数量 (M) | FLOPs (M) | 模型大小 (MB) | CPU 延迟 (ms) | 内存 (MB) |
|------|------|----------|-----------|-----------|-------------|-------------|----------|
| **baseline** | tinycnn | 0.09 | 0.0 | 0.86 | 0.02 | 0.184 | N/A |
| **pruned** | tinycnn-pruned | 0.10 | 0.0 | 0.86 | 0.02 | 0.195 | N/A |
| **quantized** | tinycnn-quantized-8bit | 0.09 | 0.0 | 0.86 | 0.02 | 0.187 | N/A |
| **distilled** | tinycnn-distilled | 0.03 | 0.0 | 0.86 | 0.02 | 0.18 | N/A |
| **final** | tinycnn-final | 0.03 | 0.0 | 0.86 | 0.02 | 0.211 | N/A |

## 各阶段详细数据

### baseline

- **模型**: tinycnn
- **精度**: 0.09%
- **参数量**: 0.0 M
- **FLOPs**: 0.86 M
- **模型大小**: 0.02 MB
- **CPU 延迟**: 0.184 ms

### pruned

- **模型**: tinycnn-pruned
- **精度**: 0.10%
- **参数量**: 0.0 M
- **FLOPs**: 0.86 M
- **模型大小**: 0.02 MB
- **CPU 延迟**: 0.195 ms
- **额外信息**: {"sparsity": 0.5}

### quantized

- **模型**: tinycnn-quantized-8bit
- **精度**: 0.09%
- **参数量**: 0.0 M
- **FLOPs**: 0.86 M
- **模型大小**: 0.02 MB
- **CPU 延迟**: 0.187 ms
- **额外信息**: {"model": "TinyCNN(\n  (conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)\n  (bn1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (dwconv1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)\n  (pwconv1): Conv2d(16, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (bn2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (dwconv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=32, bias=False)\n  (pwconv2): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)\n  (bn3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))\n  (classifier): Conv2d(64, 10, kernel_size=(1, 1), stride=(1, 1))\n)", "bits": 8, "scheme": "per_channel", "accuracy": 0.0859375}

### distilled

- **模型**: tinycnn-distilled
- **精度**: 0.03%
- **参数量**: 0.0 M
- **FLOPs**: 0.86 M
- **模型大小**: 0.02 MB
- **CPU 延迟**: 0.18 ms

### final

- **模型**: tinycnn-final
- **精度**: 0.03%
- **参数量**: 0.0 M
- **FLOPs**: 0.86 M
- **模型大小**: 0.02 MB
- **CPU 延迟**: 0.211 ms
- **额外信息**: {"export": {"onnx_export": {"success": false, "error": "ONNX 库未安装"}, "size_report": {"pytorch_pt": {"path": "./output/model.pt", "size_bytes": 24233, "size_mb": 0.02}, "torchscript": {"path": "./output/model_scripted.pt", "size_bytes": 50118, "size_mb": 0.05}, "parameters": {"total": 4298, "fp32_memory_mb": 0.02, "int8_memory_mb": 0.0, "int4_memory_mb": 0.0}}, "tensorrt_simulation": {"simulated": true, "note": "本报告为 TensorRT 工作流的概念验证模拟，无实际 GPU 环境。", "steps": ["1. 解析 ONNX 模型图", "2. 识别可融合的层（Conv+BN+ReLU → CBR 融合）", "3. 消除不必要的 reshape/transpose 操作", "4. 常量折叠（预计算常量表达式）", "5. 选择最优 kernel 实现（精度校准中）", "6. INT8 校准表生成（需要校准数据集）"], "original_size_mb": 0.0, "estimated_optimizations": {"FP32": {"size_mb": 0.0, "latency_improvement": "1.5x"}, "FP16": {"size_mb": 0.0, "latency_improvement": "2.0x"}, "INT8": {"size_mb": 0.0, "latency_improvement": "3.0x"}}}}}

## 压缩效果分析

### 相对基线的变化

#### **pruned**

- 精度变化: +0.02%
- 参数量: 0.0% (压缩 100.0%)
- 模型大小: 100.0% (压缩 0.0%)
- 推理加速: 0.94×

#### **quantized**

- 精度变化: +0.00%
- 参数量: 0.0% (压缩 100.0%)
- 模型大小: 100.0% (压缩 0.0%)
- 推理加速: 0.98×

#### **distilled**

- 精度变化: -0.05%
- 参数量: 0.0% (压缩 100.0%)
- 模型大小: 100.0% (压缩 0.0%)
- 推理加速: 1.02×

#### **final**

- 精度变化: -0.05%
- 参数量: 0.0% (压缩 100.0%)
- 模型大小: 100.0% (压缩 0.0%)
- 推理加速: 0.87×

## 技术方法说明

| 技术 | 方法 | 参考论文 |
|------|------|----------|
| 剪枝 | 幅度剪枝/通道剪枝/渐进剪枝 | Deep Compression (Han et al., ICLR 2016) |
| 量化 | PTQ/QAT, INT8/INT4/INT2, per-tensor/per-channel | AWQ (Lin et al., MLSys 2024), SmoothQuant (Xiao et al., ICML 2023) |
| 蒸馏 | KD 蒸馏 + 特征蒸馏 | Hinton et al., 2015 |
| 导出 | ONNX + ONNX Runtime | - |
| 基准测试 | 参数量/FLOPs/延迟/内存/吞吐量 | MCUNet (Lin et al., NeurIPS 2020) |

## 运行环境

- **操作系统**: Linux 6.17.0-35-generic
- **Python 版本**: 3.13.12
- **PyTorch 版本**: 2.7.0+cu128
- **设备**: CPU (无 GPU)

---

*报告由 ReportGenerator 自动生成于 2026-06-22 11:36:40*
