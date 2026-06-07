# Lecture 12 - Transformer Efficiency Analysis

> MIT 6.5940: Efficient Deep Learning Computing  
> Topic: Transformer & LLM — 注意力机制效率分析

## Overview

Understanding the computational and memory costs of Transformer attention is essential for efficient LLM deployment. This analysis quantifies FLOPs, KV-cache memory, and compares attention variants.

## Key Concepts

- **O(n²) Attention**: Standard self-attention scales quadratically with sequence length — the dominant bottleneck for long contexts
- **KV Cache**: During autoregressive generation, key-value pairs are cached to avoid recomputation; size grows linearly with sequence length
- **MHA**: Multi-Head Attention — each head has independent K, Q, V projections
- **MQA**: Multi-Query Attention — all heads share K, V projections (reduces KV cache by 1/n_heads)
- **GQA**: Grouped-Query Attention — groups of heads share K, V projections (tradeoff between MHA and MQA)

## Implementation

| Component          | Description                                              |
| ------------------ | -------------------------------------------------------- |
| GPT-style Model    | Configurable n_layers, d_model, n_heads, head_dim, vocab |
| MHA / MQA / GQA    | Three attention variants with parameter counting         |
| FLOPs Calculator   | Analytical FLOP counts for attention + FFN per seq_len   |
| KV Cache Analyzer  | Memory in MB for FP16 KV cache at different seq lengths  |
| Comparison Tables  | MHA vs MQA vs GQA across 4 model scales                  |

## Usage

```bash
cd src/lecture-12
python main.py
```

## Expected Output

```
============================================================
Attention FLOPs vs Sequence Length
============================================================
SeqLen    Attn FLOPs (M)    Total FLOPs (M)    KV Cache (MB)
------------------------------------------------------------
64            0.52              3.41              0.38
128           2.10             11.68              0.77
256           8.39             42.67              1.54
512          33.55            161.63              3.07
1024        134.22            626.69              6.14
2048        536.87           2465.15             12.29
============================================================
Quadratic growth: FLOPs(2n) / FLOPs(n) = 4.0×
============================================================
```

## References

- Vaswani et al., "Attention Is All You Need" (NeurIPS 2017)
- Shazeer, "Fast Transformer Decoding: One Write-Head Is All You Need" (2019)
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models" (2023)
- MIT 6.5940 Lecture 12 Slides
