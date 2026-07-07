# FlashAttention / SageAttention / FlashInfer

> 知识点扩展：三个 attention 加速库的原理与在 FastVideo 的使用。

## 1. FlashAttention

**原理**：IO-aware exact attention。传统 attention 把 `QKᵀ`（L×L）写回显存再 softmax，IO 瓶颈。FlashAttention 分块（tile）计算，用 online softmax 在 SRAM 内完成，不物化全矩阵。

版本：
- **FA2**：Ampere/Hopper 通用。
- **FA3**：Hopper 专用（TMA + wgmma），更快。
- **FA4**：CuTe DSL，sm90+，支持 FP4。

FastVideo（`attention/backends/flash_attn.py`）：
```python
# FASTVIDEO_FA4=1 → FA4；默认 FA3 > FA2
# 变长/padding → flash_attn_varlen / flash_attn_no_pad
# FP4（Blackwell）→ _forward_nvfp4（需 flashinfer nvfp4_quantize）
```

## 2. SageAttention

**原理**：量化 attention。把 Q/K 量化成 INT8（v1）或 FP8/FP4（v3），用低精度 tensor core 加速 `QKᵀ`，同时用 smoothing 技巧保持精度。

FastVideo：
- `sage_attn.py`（v1，INT8）：`sageattn(q, k, v, tensor_layout="NHD")`。
- `sage_attn3.py`（v3，Blackwell）：`sageattn3_blackwell`，需转置到 BHSD。

## 3. FlashInfer

**原理**：LLM 推理加速库，提供高效 attention kernel、量化 kernel。FastVideo 用它的 `nvfp4_quantize` 做 FP4 量化（Blackwell FP4 attention 路径）。依赖：`flashinfer-python`（pyproject.toml）。

## 4. 三者对比

| 库 | 核心 | 精度 | 架构 | FastVideo 用途 |
|----|------|------|------|---------------|
| FlashAttention | tiling + online softmax | exact / FP4 | Ampere+ | 默认 attention |
| SageAttention | Q/K 量化 | INT8/FP8/FP4 | 通用/Blackwell | 加速 attention |
| FlashInfer | LLM kernel 库 | FP4 量化 | Blackwell | FP4 quantize |

## 5. 在去噪循环的位置

这些都是 dense attention 后端，在 DiT 每个 block 的 self/cross attention 调用，通过 `selector.py` 选择：
```bash
FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
FASTVIDEO_ATTENTION_BACKEND=SAGE_ATTN
```

## 6. FP4 量化路径（Blackwell）

```python
# flash_attn.py:_nvfp4_quantize_for_fa4
# Q/K: BF16 → float4_e2m1fn + scale factor（用 flashinfer）
# V: BF16
# 调 flash_attn_cute._flash_attn_fwd(mSFQ, mSFK)
```
仅 sm_100a/sm_103a（Blackwell）。

## 7. torch.compile 兼容

FA 默认不可追踪。FastVideo 用 `torch.library.custom_op("fastvideo::_flash_attn_default_forward")` 包装（`flash_attn.py:65`）。

## 8. 安装

```bash
uv pip install flash-attn==2.8.1 --no-cache-dir --no-build-isolation   # FA2
# FA3/FA4/Sage/FlashInfer 各有独立安装
```

## 9. 回扣源码
| 库 | 源码 |
|----|------|
| FlashAttention | `attention/backends/flash_attn.py`, `utils/flash_attn_cute.py` |
| SageAttention | `attention/backends/sage_attn.py`, `sage_attn3.py` |
| FlashInfer | `flash_attn.py:_nvfp4_quantize_for_fa4` |

## 10. 延伸
- attention 加速：[`05_attention_acceleration.md`](05_attention_acceleration.md)
- kernel：[`11_cuda_kernel_and_pytorch_extension.md`](11_cuda_kernel_and_pytorch_extension.md)
