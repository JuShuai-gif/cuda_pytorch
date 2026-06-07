# Lecture 16 - Vision Transformer Efficiency

> MIT 6.5940: Efficient Deep Learning Computing  
> Topic: ViT — 视觉 Transformer 效率分析

## Overview

Vision Transformers (ViT) apply the Transformer architecture to images by treating image patches as tokens. This analysis compares ViT with CNNs across different patch/image sizes and visualizes attention patterns.

## Key Concepts

- **Patch Embedding**: Split image into fixed-size patches (e.g., 16×16) and linearly project each to d_model
- **Positional Encoding**: Learnable vectors added to patch embeddings to encode spatial information
- **[CLS] Token**: Special classification token whose final representation is used for prediction
- **Patch Size Tradeoff**: Smaller patches → more tokens → O(n²) attention cost grows rapidly

## Implementation

| Component            | Description                                              |
| -------------------- | -------------------------------------------------------- |
| PatchEmbedding       | Image → patches → linear projection                      |
| ViT                  | 4-layer transformer with pre-LN and [CLS] token          |
| ResNet-style CNN     | Comparable CNN with matched param budget                 |
| FLOPs/Params Counter | Analytical count for patch_size ∈ [4,8,16], img ∈ [32,64,96] |
| Attention Visualizer | Extract attention map → save heatmap plot                |
| ViT vs CNN Compare   | Params, FLOPs, and characteristics table                 |

## Usage

```bash
cd src/lecture-16
python main.py
```

The script saves an attention map visualization to `/tmp/vit_attention_map.png`.

## Expected Output

```
============================================================
ViT Parameter & FLOP Analysis
============================================================
Patch    Image    Tokens    Params (K)    FLOPs (M)
------------------------------------------------------------
4×4      32×32    64        312.5         18.2
8×8      32×32    16        312.5          5.1
16×16    32×32     4        312.5          2.8
4×4      64×64    256       312.5        124.0
8×8      64×64    64        312.5         33.5
16×16    64×64    16        312.5         11.2
============================================================
ViT vs CNN Comparison (both ~300K params):
Model    Params (K)    FLOPs (M)    Scaling
------------------------------------------------------------
ViT          312.5          5.1      O(P²) attention
CNN          305.2          8.3      O(P) conv
============================================================
```

## References

- Dosovitskiy et al., "An Image is Worth 16x16 Words" (ICLR 2021)
- MIT 6.5940 Lecture 16 Slides
