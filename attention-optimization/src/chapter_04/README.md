# Chapter 04: FlashAttention V1

## Implementation

Step-by-step CUDA implementation of FlashAttention V1:

1. `step1_tiled_matmul.cu` - Tiled matrix multiply with shared memory
2. `step2_online_softmax.cu` - Online softmax algorithm
3. `step3_flash_attention.cu` - Full FlashAttention V1 kernel
4. `benchmark.py` - Compare vs naive and PyTorch SDPA

## Key Concepts

- Online softmax rescaling
- Tiling strategy (Q blocks x K/V blocks)
- Shared memory tiling
- Kernel fusion (avoid writing S, P to HBM)

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80
make chapter_04_flash_attention
```

## Architecture

```
Q [N, d] ──┐
            ├──> Tile Q_i [Br, d] ──┐
K [N, d] ──┘                        ├──> S_ij [Br, Bc] (on SMEM)
                                     │      ↓
V [N, d] ───────────────────────────> P_ij @ V_j [Br, d] (on SMEM)
                                            ↓
                                         O_i += rescale * new
```
