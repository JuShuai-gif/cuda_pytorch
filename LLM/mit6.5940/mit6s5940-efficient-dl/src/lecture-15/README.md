# Lecture 15 - Long-Context Attention Optimizations

> MIT 6.5940: Efficient Deep Learning Computing  
> Topic: Long-Context LLM — 长上下文注意力优化

## Overview

Standard self-attention scales O(n²) with sequence length, making it infeasible for long contexts (e.g., 128K tokens). This demo explores optimization strategies: sliding window, streaming, RoPE extensions, and KV cache eviction.

## Key Concepts

- **Sliding Window Attention**: Each query attends only to the last w tokens, reducing complexity to O(n·w)
- **Streaming Attention**: Process sequence in chunks with a rolling KV cache window
- **NTK-Aware RoPE Scaling**: Extend RoPE to longer contexts by scaling rotary frequencies (base frequency adjustment)
- **KV Cache Eviction**: Retain only important tokens — keep first k (sink tokens) + last m (recent tokens)

## Implementation

| Technique           | Description                                              |
| ------------------- | -------------------------------------------------------- |
| Full Attention      | Standard O(n²) scaled dot-product, baseline              |
| Sliding Window      | O(n·w) local attention with configurable window size     |
| Streaming Attention | Chunked processing with rolling window KV cache          |
| RoPE + NTK Scaling  | Rotary Position Embedding with frequency rescaling       |
| KV Cache Eviction   | Keep first 4 + last 4 tokens, evict middle               |
| Memory Comparison   | Attention score matrix size across seq lengths [256..4096] |

## Usage

```bash
cd src/lecture-15
python main.py
```

## Expected Output

```
============================================================
Attention Memory Footprint Comparison
============================================================
SeqLen    Full O(n²)    Window(w=128)    Streaming    Reduction
----------------------------------------------------------------
256        65,536            32,768         32,768       2.0×
512       262,144            65,536         65,536       4.0×
1024    1,048,576           131,072        131,072       8.0×
2048    4,194,304           262,144        262,144      16.0×
4096   16,777,216           524,288        524,288      32.0×
============================================================
```

## References

- Beltagy et al., "Longformer: The Long-Document Transformer" (2020)
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
- bloc97, "NTK-Aware Scaled RoPE" (2023)
- Xiao et al., "Efficient Streaming Language Models with Attention Sinks" (2024)
- MIT 6.5940 Lecture 15 Slides
