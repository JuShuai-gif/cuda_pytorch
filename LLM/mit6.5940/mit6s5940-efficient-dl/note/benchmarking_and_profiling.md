# Benchmarking and Profiling 工业规范

高效深度学习项目不能只报 FLOPs。FLOPs 是解释变量，不是上线指标。

## 1. 延迟测量规范

| 项 | 要求 |
|---|---|
| warmup | 至少 10 次，quick 模式可 1 次 |
| repeat | 至少 50-100 次，quick 模式可 3 次 |
| 统计 | mean、median/P50、P90、P95、P99 |
| CUDA | 每次计时前后 `torch.cuda.synchronize()` |
| CPU | 固定线程数，记录 `torch.get_num_threads()` |
| 数据 | 不把 data loading 算入 model latency，除非测端到端 pipeline |

## 2. 统一指标

- 模型大小：state_dict bytes、ONNX bytes、engine bytes。
- 参数量：trainable/all parameters。
- 计算量：MACs/FLOPs，但必须标注估算工具。
- 内存：CPU RSS、CUDA allocated/reserved、activation/KV cache。
- 吞吐：images/s、QPS、tokens/s。
- 质量：accuracy、MSE、cosine、perplexity、success rate。

## 3. 报告模板

```text
Experiment: model + compression + runtime + hardware
Baseline: metric table
Compressed: metric table
Delta: size/speed/accuracy/memory
Skipped: dependency or hardware reason
Decision: accept / reject / needs more calibration
```

## 4. 工具映射

| 工具 | 适用 |
|---|---|
| torch.profiler | PyTorch eager hotspot |
| Nsight Systems/Compute | CUDA kernel timeline and occupancy |
| onnxruntime benchmark | CPU/GPU provider latency |
| trtexec | TensorRT engine latency and memory |
| perf / VTune | CPU cache/memory bottleneck |
| TFLite benchmark_model | mobile/TinyML runtime |
