# Additional Documentation

## Architecture Diagrams

```mermaid
graph TD
    subgraph "Learning Path"
        CH01["Ch01: Basics"] --> CH02["Ch02: GPU"]
        CH02 --> CH03["Ch03: Profiling"]
        CH03 --> CH04["Ch04: FlashAttn V1"]
        CH04 --> CH05["Ch05: FlashAttn V2"]
        CH05 --> CH06["Ch06: FlashAttn V3"]
        CH03 --> CH07["Ch07: KV Cache"]
        CH07 --> CH08["Ch08: PagedAttn"]
        CH07 --> CH09["Ch09: MQA/GQA"]
        CH04 --> CH10["Ch10: Sliding Window"]
        CH04 --> CH11["Ch11: Sparse Attn"]
        CH04 --> CH12["Ch12: Linear Attn"]
        CH07 --> CH13["Ch13: Quantized"]
        CH04 --> CH14["Ch14: TensorRT-LLM"]
        CH15["Ch15: xFormers"]
        CH16["Ch16: vLLM"]
        CH14 --> FP["Final Project"]
        CH08 --> FP
        CH09 --> FP
    end
```

## GPU Architecture Reference

| GPU | Arch | Compute Capability | TFLOPS (FP16) | HBM |
|-----|------|-------------------|---------------|-----|
| A100 | Ampere | 8.0 | 312 | 80GB @ 2TB/s |
| H100 | Hopper | 9.0 | 990 | 80GB @ 3.35TB/s |
| RTX 4090 | Ada | 8.9 | 330 | 24GB @ 1TB/s |
| RTX 3090 | Ampere | 8.6 | 142 | 24GB @ 936GB/s |

## Key References

1. **FlashAttention**: https://arxiv.org/abs/2205.14135 (V1), https://arxiv.org/abs/2307.08691 (V2)
2. **vLLM**: https://arxiv.org/abs/2309.06180
3. **GQA**: https://arxiv.org/abs/2305.13245
4. **MQA**: https://arxiv.org/abs/1911.02150
5. **xFormers**: https://github.com/facebookresearch/xformers
6. **TensorRT-LLM**: https://github.com/NVIDIA/TensorRT-LLM
7. **Performer**: https://arxiv.org/abs/2009.14794

## Performance Checklist

For each chapter implementation, verify:
- [ ] Correctness vs PyTorch reference (max error < 1e-3)
- [ ] Latency reported in milliseconds
- [ ] GFLOPS / TFLOPS computed
- [ ] Memory bandwidth utilization estimated
- [ ] Profiling data captured (Nsight or torch.profiler)
- [ ] Roofline analysis done (AI computed)
