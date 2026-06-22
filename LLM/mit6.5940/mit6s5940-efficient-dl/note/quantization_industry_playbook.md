# 量化工业实战 Playbook

本文件补充 Lecture 05/06：量化不是 cast，而是 calibration、scale/zp、graph rewrite 和 kernel/runtime 支持的组合。

## 1. 方法选择

| 场景 | 推荐方法 | 说明 |
|---|---|---|
| CNN/ViT 端侧推理 | W8A8 PTQ 或 QAT | TensorRT/ONNX Runtime/OpenVINO 支持成熟 |
| CPU Transformer | dynamic quantization | Linear-heavy 模型收益明显 |
| LLM GPU 推理 | W4A16 AWQ/GPTQ | 主要降低权重带宽 |
| LLM server W8A8 | SmoothQuant / TensorRT-LLM | 需要处理 activation outlier |
| MCU | int8 per-channel weight + int8 activation | TFLite Micro/CMSIS-NN 常用 |

## 2. 必须测的指标

- Layerwise SQNR / cosine similarity。
- Task metric：accuracy、mAP、perplexity、success rate 或 action MSE。
- Calibration set 覆盖率：是否覆盖长尾输入、异常光照、极端 token。
- Runtime：P50/P95/P99 latency、throughput、peak memory。
- Export：Q/DQ graph、calibration cache、opset、runtime provider。

## 3. 失败模式

- 最后一层分类头、回归头、robot action head 直接 INT8 导致输出漂移。
- LayerNorm/Softmax/GELU 被错误量化导致 Transformer 崩溃。
- calibration data 太少，min/max 被 outlier 主导。
- per-tensor quant 用在 outlier 很强的 channel，精度显著下降。
- ONNX Runtime/TensorRT 支持的 quantized op 不一致。

## 4. 代码补齐目标

- `src/model_compression/quantization_observers.py`：MinMax、Percentile、MSE/KL observer。
- `src/model_compression/quantization_report.py`：输出 SQNR/cosine/error histogram。
- `labs/lab-02/calibration_checklist.md`：校准集和 QAT 验收清单。
