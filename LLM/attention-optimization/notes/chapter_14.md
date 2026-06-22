# Chapter 14: TensorRT-LLM Attention

## 1. TensorRT-LLM 简介

TensorRT-LLM 是 NVIDIA 的 LLM 推理优化框架，提供：

- **Fused Attention Kernels**: 将多个操作融合为一个 GPU kernel
- **Plugin 机制**: 自定义 kernel 的模块化集成
- **Graph Optimization**: 计算图级别的优化
- **Multi-GPU**: 张量并行 + 流水线并行

## 2. Fused Attention

### 2.1 为什么需要 Kernel Fusion

标准 PyTorch 实现中，一个 Attention 操作会被分解为多个 CUDA kernel：

```
Q @ K^T     → cublasGemmEx    (kernel 1)
Scale       → element_wise     (kernel 2)
Softmax     → softmax_kernel   (kernel 3)
P @ V       → cublasGemmEx    (kernel 4)
```

每次 kernel launch 有开销，且中间结果要写回 HBM。

### 2.2 TensorRT-LLM 的 Fusion

将整个 Attention 融合为一个 kernel：

```mermaid
graph LR
    subgraph "Before Fusion"
        K1["Gemm"] --> HBM1["Write to HBM"]
        HBM1 --> K2["Scale"] --> HBM2["Write to HBM"]
        HBM2 --> K3["Softmax"] --> HBM3["Write to HBM"]
        HBM3 --> K4["Gemm"] --> HBM4["Write to HBM"]
    end

    subgraph "After Fusion"
        FK["Fused Attention Kernel<br/>(All ops in one kernel)"]
    end

    style HBM1 fill:#FF6B6B
    style HBM2 fill:#FF6B6B
    style HBM3 fill:#FF6B6B
    style FK fill:#90EE90
```

## 3. Plugin 机制

### 3.1 架构

```
TensorRT Graph
├── Plugin: GPTAttention
│   ├── QKV Projection (fused GEMM)
│   ├── FlashAttention (custom CUDA kernel)
│   ├── KV Cache append
│   └── Output projection
├── Plugin: LayerNorm
└── Plugin: GELU Activation
```

### 3.2 Plugin 接口

```cpp
class GPTAttentionPlugin : public IPluginV2DynamicExt {
    // Required interfaces:
    int getNbOutputs() const override;
    DimsExprs getOutputDimensions(...) override;
    int enqueue(const PluginTensorDesc* inputs,
                const PluginTensorDesc* outputs,
                const void* const* buffers,
                void* const* output_buffers,
                cudaStream_t stream) override;
    // ...
};
```

## 4. Mini TensorRT-LLM 实现

`mini_trt_attention/` 目录包含：
1. Fused Attention kernel
2. Plugin 接口简化实现
3. Llama-style attention 集成

## 5. 关键优化

| 优化 | 效果 |
|------|------|
| QKV Fusion | 1 GEMM 代替 3 个 GEMM |
| FP8 GEMM | 2× 带宽减少 |
| FlashAttention | 消除 O(N²) 中间矩阵 |
| KV Cache in INT8 | 2× KV Cache 减少 |
| Multi-stream | 重叠计算和通信 |

---

## TensorRT-LLM 工业增强

补充 plugin contract、fused attention 数据流和 mini plugin demo。

### 1. 工业视角

优化 attention 不能只看 Big-O。生产环境必须同时记录：`shape`、`dtype`、`batch`、`heads`、`seq_len`、`head_dim`、`causal`、warmup/iters、GPU 型号和 profiler 版本。对于推理系统，还要区分 **prefill** 与 **decode**，因为两者的瓶颈完全不同。

### 2. 复杂度与显存公式

标准 attention 的主要计算量：

$$\mathrm{FLOPs} \approx 2N^2d_k + 2N^2d_v + O(N^2)$$

显式保存 `S` 和 `P` 的中间显存：

$$\mathrm{Bytes}_{S+P} = 2N^2 \times \mathrm{sizeof(dtype)}$$

Roofline 判断：

$$\mathrm{AI}=\frac{\mathrm{FLOPs}}{\mathrm{Bytes\ moved}},\quad
\mathrm{Perf} \le \min(\mathrm{PeakFLOPS},\mathrm{PeakBW}\times\mathrm{AI})$$

### 3. 工程检查清单

| 检查项 | 要求 |
|---|---|
| Correctness | 小 shape 与 CPU/PyTorch reference 对齐。 |
| Benchmark | 固定 warmup、iters、shape、dtype，输出 latency 和吞吐。 |
| Memory | 说明 HBM 读写、中间 tensor、KV cache 或 block table 成本。 |
| Profiling | 至少记录 occupancy、DRAM throughput、SM throughput、stall reason。 |
| Reproducibility | README 中给出构建、运行、profiling 命令。 |

### 4. 本章实现入口

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build . --target mini_trt_attention -j
./chapters/mini_trt_attention
```

### 5. 示意图

```mermaid
flowchart LR
    Math[Formula / Complexity] --> Impl[Reference Implementation]
    Impl --> Test[Correctness Check]
    Test --> Bench[Benchmark]
    Bench --> Profile[Nsight / torch.profiler]
    Profile --> Optimize[Next Optimization]
```
