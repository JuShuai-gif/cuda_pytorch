# 模型压缩实验报告 (Model Compression Benchmark Report)

**生成时间**: 2026-06-12 11:03:21
**运行设备**: cpu
**PyTorch 版本**: 2.7.0+cu128

> 本报告由 `benchmark_compression.py` 自动生成，所有数据均为脚本真实测量。

## 运行环境

- **OS**: Linux 6.17.0-35-generic
- **Python**: 3.13.12
- **PyTorch**: 2.7.0+cu128
- **CPU Count**: 32
- **CUDA Available**: Yes
- **CUDA Version**: 12.8
- **GPU**: NVIDIA GeForce RTX 4090 Laptop GPU
- **onnxruntime**: not installed
- **onnx**: not installed
- **psutil**: 7.2.2
- **fvcore**: installed
- **thop**: installed

## TensorRT 可用性

> TensorRT is **NOT** available: tensorrt Python package not installed; trtexec CLI not found in PATH
> TensorRT benchmarks are skipped. To enable, install TensorRT SDK and Python bindings.

## SmallCNN

| 方法 | 参数量 (M) | 模型大小 (MB) | 压缩率 | 延迟 (ms) | P95延迟 (ms) | 吞吐 (samples/s) | 内存 (MB) | GPU显存 (MB) | MSE vs Baseline | 端侧可部署 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| SmallCNN FP32 Baseline | 0.621 | 2.3777 | 1.00x | 1.1949 | 1.2337 | 5683.32 | 0.0000 | 0.0000 | 0.000000 | No |
| SmallCNN Unstructured Prune | 0.621 (eff 0.311) | 2.3777 | 1.00x | 1.2005 | 1.2343 | 5656.54 | 0.0000 | 0.0000 | 0.000292 | Yes |
| SmallCNN Channel Prune | 0.621 | 2.3777 | 1.00x | 1.1185 | 1.1408 | 5926.61 | 0.0000 | 0.0000 | 0.002019 | Yes |
| SmallCNN PTQ INT8 | 0.621 | 2.3777 | 1.00x | 1.1701 | 1.1839 | 5778.07 | 0.0000 | 0.0000 | 0.000000 | Yes |

### SmallCNN 压缩分析

**SmallCNN Unstructured Prune**:
  - 模型大小: 2.3777 MB (基线 2.3777 MB, 压缩率 1.00x)
  - 延迟: 1.2005 ms (基线 1.1949 ms, 减速 1.00x)
  - MSE (vs baseline): 0.000292
  - Cosine Similarity: 0.9808

**SmallCNN Channel Prune**:
  - 模型大小: 2.3777 MB (基线 2.3777 MB, 压缩率 1.00x)
  - 延迟: 1.1185 ms (基线 1.1949 ms, 加速 1.07x)
  - MSE (vs baseline): 0.002019
  - Cosine Similarity: 0.9144

**SmallCNN PTQ INT8**:
  - 模型大小: 2.3777 MB (基线 2.3777 MB, 压缩率 1.00x)
  - 延迟: 1.1701 ms (基线 1.1949 ms, 加速 1.02x)
  - MSE (vs baseline): 0.0
  - Cosine Similarity: 1.0

## Transformer

| 方法 | 参数量 (M) | 模型大小 (MB) | 压缩率 | 延迟 (ms) | P95延迟 (ms) | 吞吐 (samples/s) | 内存 (MB) | GPU显存 (MB) | MSE vs Baseline | 端侧可部署 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| Transformer FP32 Baseline | 0.198 | 0.7619 | 1.00x | 1.1616 | 1.2260 | 4821.42 | 0.0000 | 0.0000 | 0.000000 | No |
| Transformer PTQ INT8 | 0.198 | 0.7619 | 1.00x | 1.1592 | 1.1871 | 4750.19 | 0.0000 | 0.0000 | 0.000001 | No |
| Transformer Dynamic Quant | 0.198 | 0.7619 | 1.00x | 1.3970 | 1.4290 | 4225.60 | 0.0000 | 0.0000 | 0.000034 | No |

### Transformer 压缩分析

**Transformer PTQ INT8**:
  - 模型大小: 0.7619 MB (基线 0.7619 MB, 压缩率 1.00x)
  - 延迟: 1.1592 ms (基线 1.1616 ms, 加速 1.00x)
  - MSE (vs baseline): 1e-06
  - Cosine Similarity: 1.0

**Transformer Dynamic Quant**:
  - 模型大小: 0.7619 MB (基线 0.7619 MB, 压缩率 1.00x)
  - 延迟: 1.3970 ms (基线 1.1616 ms, 减速 0.83x)
  - MSE (vs baseline): 3.4e-05
  - Cosine Similarity: 1.0

## VLA ActionHead

| 方法 | 参数量 (M) | 模型大小 (MB) | 压缩率 | 延迟 (ms) | P95延迟 (ms) | 吞吐 (samples/s) | 内存 (MB) | GPU显存 (MB) | MSE vs Baseline | 端侧可部署 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| VLA ActionHead FP32 Baseline | 1.020 | 3.8929 | 1.00x | 0.2320 | 0.2375 | 29518.06 | 0.0000 | 0.0000 | 0.000000 | Yes |
| VLA ActionHead Pruned | 1.020 (eff 0.511) | 3.8929 | 1.00x | 0.1153 | 0.1274 | 58200.00 | 0.0000 | 0.0000 | 0.000560 | Yes |
| VLA ActionHead PTQ INT8 | 1.020 | 3.8929 | 1.00x | 0.1342 | 0.1684 | 54972.56 | 0.0000 | 0.0000 | 0.000000 | Yes |

### VLA ActionHead 压缩分析

**VLA ActionHead Pruned**:
  - 模型大小: 3.8929 MB (基线 3.8929 MB, 压缩率 1.00x)
  - 延迟: 0.1153 ms (基线 0.2320 ms, 加速 2.01x)
  - MSE (vs baseline): 0.00056
  - Cosine Similarity: 0.8743
  - 注: Measured as output MSE vs FP32 baseline. In production, use rollout success rate and trajectory error.

**VLA ActionHead PTQ INT8**:
  - 模型大小: 3.8929 MB (基线 3.8929 MB, 压缩率 1.00x)
  - 延迟: 0.1342 ms (基线 0.2320 ms, 加速 1.73x)
  - MSE (vs baseline): 0.0
  - Cosine Similarity: 1.0
  - 注: INT8 quantization typically introduces small action drift. For safety-critical VLA, always validate with rollout MSE.

## SimpleMLP

| 方法 | 参数量 (M) | 模型大小 (MB) | 压缩率 | 延迟 (ms) | P95延迟 (ms) | 吞吐 (samples/s) | 内存 (MB) | GPU显存 (MB) | MSE vs Baseline | 端侧可部署 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| MLP FP32 Baseline | 0.235 | 0.9000 | 1.00x | 0.0597 | 0.0704 | 85940.96 | 0.0000 | 0.0000 | 0.000000 | No |
| MLP Unstructured Prune | 0.235 (eff 0.118) | 0.9000 | 1.00x | 0.0597 | 0.0687 | 100264.42 | 0.0000 | 0.0000 | 0.001295 | No |
| MLP PTQ INT8 | 0.235 | 0.9000 | 1.00x | 0.0542 | 0.0605 | 84582.73 | 0.0000 | 0.0000 | 0.000000 | No |

### SimpleMLP 压缩分析

**MLP Unstructured Prune**:
  - 模型大小: 0.9000 MB (基线 0.9000 MB, 压缩率 1.00x)
  - 延迟: 0.0597 ms (基线 0.0597 ms, 加速 1.00x)
  - MSE (vs baseline): 0.001295
  - Cosine Similarity: 0.9227

**MLP PTQ INT8**:
  - 模型大小: 0.9000 MB (基线 0.9000 MB, 压缩率 1.00x)
  - 延迟: 0.0542 ms (基线 0.0597 ms, 加速 1.10x)
  - MSE (vs baseline): 0.0
  - Cosine Similarity: 1.0

## 工业部署建议

1. **CNN/ViT 端侧部署**: 优先尝试结构化通道剪枝 + INT8 PTQ。结构化稀疏对硬件友好，PTQ 部署成本低。
2. **Transformer/LLM CPU 推理**: 优先尝试动态量化 (torch.quantize_dynamic) 或 weight-only INT4 量化。GPU 推理优先使用 TensorRT-LLM 或 vLLM。
3. **VLA/机器人 Action Head**: MLP action head 中间层可积极量化 (INT8)，最后输出层谨慎处理。验收指标必须包含 action MSE、rollout success rate 和 P99 latency。
4. **非结构化剪枝**: 虽然压缩率高，但需要专用 sparse kernel (如 cuSPARSE、MKL Sparse BLAS) 才能在中低稀疏度 (<=90%) 上实现延迟收益。无 sparse kernel 环境下可能不降反升。
5. **精度验证**: 本报告使用 MSE 作为代理 metric。真实项目必须用 task metric（准确率、perplexity、mAP、success rate）验证。
6. **TensorRT 部署**: engine 必须用目标硬件真实构建和 benchmark。ONNX → TensorRT engine 过程中 INT8 校准需要真实校准数据。

## 免责声明

本报告中的量化方法为简化实现（手动 scale/round/clamp），与 PyTorch 原生量化后端 (fbgemm/qnnpack) 略有不同。生产部署请使用 torch.ao.quantization 或 TensorRT 原生量化流程。

---
*报告由 benchmark_compression.py 自动生成于 2026-06-12 11:03:21*
