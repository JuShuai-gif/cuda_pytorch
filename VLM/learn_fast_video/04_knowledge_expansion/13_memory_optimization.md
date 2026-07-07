# 显存优化

> 知识点扩展：FastVideo 的显存优化手段汇总，回扣源码开关。

## 1. 显存都花在哪

视频扩散显存大户：
1. **模型权重**：14B 模型 bf16 约 28GB。
2. **激活**：长序列 DiT 的中间激活（数万 token）。
3. **VAE decode**：解码时的峰值。
4. **KV cache / attention 中间矩阵**。

### 1.1 训练 vs 推理的显存构成

**推理**（无梯度）：
```
权重 + 激活（当前步）+ VAE decode 峰值
```

**训练**（有梯度+优化器）：
```
权重 + 梯度 + 优化器状态(AdamW×2) + 全部激活(backward 用) + VAE
```
训练显存 ≈ 推理的 4-5 倍。这就是为什么训练必开 FSDP + AC，推理可只开 offload。

### 1.2 一个估算例子（14B 模型 bf16）

| 项 | 训练 | 推理 |
|----|------|------|
| 权重 | 28 GB | 28 GB |
| 梯度 | 28 GB | 0 |
| 优化器(AdamW fp32) | 112 GB | 0 |
| 激活 | 巨大（AC 前） | 小（仅当前步） |
可见训练光"权重+梯度+优化器"就 168 GB，单卡 80GB H100 放不下，必须 FSDP 分片 + AC 砍激活。

## 2. 优化手段总览

| 手段 | 针对 | 源码开关 |
|------|------|---------|
| FSDP 参数分片 | 权重 | `hsdp_shard_dim` |
| 序列并行 | 激活 | `sp_size` |
| Activation Checkpointing | 激活 | `enable_gradient_checkpointing_type` |
| CPU offload | 权重 | `dit_cpu_offload` 等 |
| Layerwise offload | 权重 | `dit_layerwise_offload` |
| VAE tiling | VAE 峰值 | `vae_config.use_tiling` |
| 稀疏 attention | attention 矩阵 | `VSA_sparsity` |
| 量化 | 权重+计算 | fp8/fp4/int8 |
| Mixed precision | 全部 | `param_dtype=bf16` |

## 3. FSDP 参数分片

每卡只存 1/N 参数，用时 all-gather。见 [`08_fsdp_and_distributed_training.md`](08_fsdp_and_distributed_training.md)。

## 4. 序列并行

激活按序列切分到多卡，单卡激活降为 1/sp_size。见 [`07_sequence_parallelism.md`](07_sequence_parallelism.md)。

## 5. Activation Checkpointing

```
train/models/wan/wan.py:apply_activation_checkpointing
```
不存中间激活，backward 重算。用算力换显存。视频 DiT 激活巨大，训练必需。

## 6. CPU Offload（推理）

```python
# fastvideo_args.py
dit_cpu_offload=True          # DiT 不用时放 CPU
text_encoder_cpu_offload=True # 编码完就 offload
vae_cpu_offload=True
dit_layerwise_offload=True    # 逐层 offload（极致，但慢）
pin_cpu_memory=True           # pin 内存加速 CPU↔GPU
```
适合单卡小显存推理。代价是 CPU↔GPU 传输延迟。

## 7. VAE Tiling

```
models/vaes/common.py:ParallelTiledVAE
```
VAE decode 是显存峰值点。tiling 分块解码（时间/空间/并行），块间 blend。配置 `use_tiling`, `tile_sample_min_*`。

## 8. 稀疏 Attention

VSA 只算 top-k block，避免 O(L²) 的 attention 矩阵。`VSA_sparsity=0.9` 省 90% attention 计算。见 [`06_sparse_attention.md`](06_sparse_attention.md)。

## 9. 量化

- FP8（`layers/fp8linear.py`）、FP4/NVFP4（`layers/fp4linear.py`）、INT8（TurboDiffusion GEMM）。
- 权重和激活用低精度，省显存 + 加速。FastWan-QAD 用 FP8 量化蒸馏。

### 9.1 精度格式对比

| 格式 | 位数 | 相对 bf16 显存 | 硬件 | 场景 |
|------|------|---------------|------|------|
| fp32 | 32 | 2× | 全部 | 梯度 reduce、数值敏感处 |
| bf16 | 16 | 1× | 全部 | 训练/推理默认 |
| fp16 | 16 | 1× | 全部 | 部分推理 |
| fp8 (e4m3/e5m2) | 8 | 0.5× | Hopper+ | 量化推理/蒸馏 |
| int8 | 8 | 0.5× | 全部 | GEMM 量化 |
| nvfp4 | 4 | 0.25× | Blackwell | 极致量化 |

### 9.2 量化的两种方式

- **PTQ（训练后量化）**：训练完直接量化权重，简单但可能掉点。
- **QAT（量化感知训练）**：训练时模拟量化误差，质量更好。FastVideo 的 `attn_qat_train`/`attn_qat_infer` 和 FastWan-QAD 属于此类。

### 9.3 量化粒度

- **per-tensor**：整个张量一个 scale，粗。
- **per-channel / per-block**：每通道/每块一个 scale，细，精度好。TurboDiffusion 的 `quant.cu` 用 block_size=128 的 per-block scale。
量化本质：`x_int = round(x / scale)`，`scale = amax / max_int`。scale 越细，量化误差越小。

### 9.4 量化省显存 + 加速的双重收益

- **显存**：权重从 16 位降到 8/4 位。
- **速度**：低精度 tensor core 吞吐更高（int8 约 2× fp16，fp4 更多）。
但要配合 QAT 或 smoothing 保精度，否则掉点。

## 10. Mixed Precision

bf16 参数 + fp32 梯度 reduce（`MixedPrecisionPolicy`）。权重 bf16 省一半显存。

## 11. 预分配 + pin_memory（推理输出）

`_generate_single_video` 预分配 pin_memory CPU buffer 接收输出，避免临时大分配。

## 12. 组合策略示例

- **单卡小显存推理**：CPU offload 全开 + VAE tiling + bf16。
- **多卡大模型推理**：SP + FSDP shard + 稀疏 attention。
- **训练**：FSDP + AC + SP + bf16。

## 13. 回扣源码
| 手段 | 源码 |
|------|------|
| offload | `fastvideo_args.py` + worker 加载逻辑 |
| tiling | `models/vaes/common.py` |
| AC | `train/models/wan/wan.py` |
| 量化 | `layers/quantization/`, `fp8linear.py` |

## 14. 延伸
- 性能分析：[`../06_practical_guides/06_how_to_profile_performance.md`](../06_practical_guides/06_how_to_profile_performance.md)
