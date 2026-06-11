# Chapter 15: xFormers 源码解析

## 1. xFormers 简介

xFormers 是 Meta 开源的 Transformer 优化库，核心组件是 **Memory-Efficient Attention**。

### 1.1 核心 API

```python
import xformers.ops as xops

# Memory-efficient attention (automatically selects best backend)
out = xops.memory_efficient_attention(Q, K, V)
```

## 2. 架构设计

### 2.1 Dispatch 机制

```mermaid
flowchart TD
    Input["Q, K, V tensors"] --> Check{Check properties:<br/>dtype, device, shape}
    Check --> Dispatch["Dispatch to backend"]
    Dispatch --> CUDA_GEN["Generic CUDA<br/>(cutlass-based)"]
    Dispatch --> CUDA_FLASH["FlashAttention<br/>(Dao-AILab)"]
    Dispatch --> CUDA_TRT["TensorRT<br/>(if available)"]
    Dispatch --> CPU["CPU backend"]
```

### 2.2 关键文件结构

```
xformers/
├── components/
│   └── attention/
│       ├── csrc/
│       │   └── attention/
│       │       ├── attention_forward_generic.cu    # Generic implementation
│       │       └── flash_attention/                # FlashAttention wrapper
│       ├── _sdp_backend.py                        # SDPA backend dispatch
│       └── attention_patterns.py                  # Attention computation patterns
```

### 2.3 核心优化

| 技术 | 位置 | 效果 |
|------|------|------|
| Tiled Matmul + SMEM | Generic CUDA | 减少 HBM 访问 |
| Online Softmax | Generic CUDA | 消除中间矩阵 |
| FlashAttention V2 | Flash backend | 最高性能 |
| Cutlass GEMM | Generic CUDA | 通用性 |
| FP16/BF16 支持 | 所有 backend | 2× 带宽节省 |

## 3. Memory-Efficient Attention (Generic)

### 3.1 算法

与 FlashAttention V1 类似，但使用 Cutlass 实现 tiled matmul：

```
For each tile of Q:
    For each tile of K, V:
        S = Q_tile @ K_tile^T          (Cutlass GEMM)
        P = online_softmax(S)          (Custom kernel)
        O += P @ V_tile                 (Cutlass GEMM)
```

### 3.2 与 FlashAttention 的对比

| | xFormers Generic | FlashAttention |
|---|-----------------|----------------|
| 基础 | Cutlass GEMM | 手写 CUDA |
| 通用性 | 任意 dtype/shape | 需要特定对齐 |
| 性能 | 较好 | 最优 |
| 维护性 | 高（基于 Cutlass） | 中（手写 kernel） |

## 4. 调用示例

### 4.1 基础用法

```python
import torch
import xformers.ops as xops

B, M, H, K = 1, 4096, 32, 128
Q = torch.randn(B, M, H, K, device="cuda", dtype=torch.float16)
K = torch.randn(B, M, H, K, device="cuda", dtype=torch.float16)
V = torch.randn(B, M, H, K, device="cuda", dtype=torch.float16)

# Bi-directional attention (no mask)
out = xops.memory_efficient_attention(Q, K, V)

# Causal attention
attn_bias = xops.LowerTriangularMask()
out_causal = xops.memory_efficient_attention(Q, K, V, attn_bias=attn_bias)
```

### 4.2 性能

预期比标准 PyTorch 快 2-5×，显存使用减少 10-20×。

## 5. 源码阅读指南

推荐阅读顺序：
1. `_sdp_backend.py` - 了解 dispatch 逻辑
2. `attention_forward_generic.cu` - 理解通用实现
3. `flash_attention/` - 理解 FlashAttention 集成

源码核心注释在 `src/chapter_15/source_reading_notes.md`。
