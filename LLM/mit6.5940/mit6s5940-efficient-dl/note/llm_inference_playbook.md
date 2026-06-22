# LLM 推理优化 Playbook

LLM 推理必须分开看 prefill 和 decode。decode 常常是 memory-bound。

## 指标

| 阶段 | 指标 |
|---|---|
| Prefill | prompt tokens/s、time-to-first-token |
| Decode | generated tokens/s、per-token latency |
| Serving | throughput、batching efficiency、P99 latency |
| Memory | weights、KV cache、activation workspace |

## 技术地图

- Weight-only quantization：AWQ、GPTQ、GGUF，降低权重带宽。
- Activation smoothing：SmoothQuant，处理 W8A8 outlier。
- KV cache：PagedAttention、KV quant、sliding window。
- Attention kernel：FlashAttention、GQA、MQA。
- Serving：continuous batching、speculative decoding、prefix cache。

## 工业坑

- 平均 tokens/s 掩盖长 prompt 的 TTFT 问题。
- INT4 权重快不快取决于 runtime 是否有高效 dequant+GEMM fusion。
- batch size 增大吞吐提升，但 P99 latency 可能恶化。
- KV cache 往往比权重量化更早触发 OOM。
