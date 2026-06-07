# Lecture 13 - LLM Deployment Simulation

> MIT 6.5940: Efficient Deep Learning Computing  
> Topic: LLM Deployment — 大语言模型部署优化

## Overview

Deploying large language models requires aggressive compression to fit within memory budgets. This simulation demonstrates weight-only quantization (AWQ-style), KV cache quantization, and the FlashAttention concept.

## Key Concepts

- **Weight-Only Quantization (AWQ)**: Quantize model weights to 4-bit per group (~128 weights) while keeping activations in higher precision. Achieves ~6-8× memory reduction.
- **KV Cache Quantization**: Cache key-value pairs in INT8 instead of FP16 to halve memory usage during generation.
- **FlashAttention**: Tiling-based exact attention computation that avoids materializing the full N×N attention matrix in HBM. Uses online softmax rescaling for numerical stability.

## Implementation

| Component          | Description                                              |
| ------------------ | -------------------------------------------------------- |
| SmallGPT           | 3-layer transformer model (~2.7M params) for quantization |
| Group-Wise INT4    | Group size=128, scale/zero per group, nibble-packed int8  |
| Perplexity Measure | Before/after quantization on synthetic text              |
| KV Cache Quant     | FP16 vs INT8 cache at lengths [256, 512, 1024, 2048]     |
| FlashAttention Docs| Extensive comments explaining tiling + online softmax    |

## Usage

```bash
cd src/lecture-13
python main.py
```

## Expected Output

```
============================================================
Weight-Only Quantization Results
============================================================
Format    Model Size (MB)    Reduction    Perplexity
------------------------------------------------------------
FP32          10.24            1.00×        1.000
FP16           5.12            2.00×        1.000
INT4           1.60            6.40×        1.234
============================================================
KV Cache Size Comparison:
SeqLen    FP16 KV (MB)    INT8 KV (MB)    Reduction
------------------------------------------------------------
256           0.77            0.38          50.0%
512           1.54            0.77          50.0%
1024          3.07            1.54          50.0%
2048          6.14            3.07          50.0%
============================================================
```

## References

- Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression" (2024)
- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention" (NeurIPS 2022)
- MIT 6.5940 Lecture 13 Slides
