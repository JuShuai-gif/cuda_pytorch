# FSDP 与分布式训练

> 知识点扩展：FSDP/FSDP2、activation checkpointing、mixed precision、offload，回扣 FastVideo。

## 1. FSDP 是什么

FSDP（Fully Sharded Data Parallel）把模型参数、梯度、optimizer 状态**分片**到多个 GPU：
- 每卡只存 1/N 的参数。
- forward/backward 时按需 all-gather 完整参数，用完释放。
- 梯度用 reduce-scatter 分片。

相比 DDP（每卡全量参数），FSDP 能训练远大于单卡显存的模型。

### 1.1 三种并行的显存账本

| 并行 | 参数 | 梯度 | optimizer 状态 | 激活 | 通信 |
|------|------|------|---------------|------|------|
| **DDP** | 全量×N | 全量×N | 全量×N | 全量×N | 梯度 all-reduce |
| **FSDP (ZeRO-3)** | 1/N | 1/N | 1/N | 全量 | 参数 all-gather + 梯度 reduce-scatter |
| **+ SP** | 1/N | 1/N | 1/N | 1/sp | 额外 all-to-all |
| **+ AC** | 1/N | 1/N | 1/N | ~√L | 重算开销 |

FSDP 本质是 ZeRO-3（参数/梯度/优化器全分片）。AdamW 的 optimizer 状态（一阶+二阶矩）是参数的 2 倍，分片它省很多。

### 1.2 FSDP 的一次 forward/backward

```
forward:  all-gather layer_i 参数 → 计算 → 释放参数
backward: all-gather layer_i 参数 → 算梯度 → reduce-scatter 梯度 → 释放
```
每层用完立即释放完整参数，所以峰值只多存"当前层"的完整参数，而非整个模型。这是"用通信换显存"。

## 2. FSDP2（PyTorch 新 API）

FastVideo 用 FSDP2（`fully_shard` + DTensor），比 FSDP1 更灵活：
```
源码：models/loader/fsdp_load.py:shard_model (L219)
```
```python
for name, module in reversed(list(model.named_modules())):
    if any(cond(name, module) for cond in fsdp_shard_conditions):
        fully_shard(module, mesh=device_mesh, mp_policy=mp_policy)
fully_shard(model, ...)   # 根模块兜底
```

### 2.1 FSDP1 vs FSDP2

| | FSDP1（FullyShardedDataParallel） | FSDP2（fully_shard） |
|--|-----------------------------------|---------------------|
| 参数表示 | FlatParameter（打平） | DTensor（每参数分片） |
| API | 包装整个 module | 函数式 `fully_shard(module)` |
| 灵活性 | 较死 | 可按模块粒度、易组合 SP/TP |
| state_dict | 需特殊处理 | DTensor 原生支持 DCP |

FastVideo 选 FSDP2 因为要和 SP、LoRA 的 DTensor 组合（`_replicate_lora_parameters` 用 DTensor Replicate）。

### 2.2 `_fsdp_shard_conditions`

每个 DiT 类声明哪些子模块该被独立分片（通常是每个 transformer block）。`shard_model` 反向遍历（从叶子到根）逐个 `fully_shard`。为什么反向：确保子模块先分片，父模块后分片，嵌套正确。

## 3. HSDP（Hybrid Sharded Data Parallel）

FastVideo 用 2D DeviceMesh 组合数据并行和模型分片：
```python
device_mesh = init_device_mesh("cuda",
    mesh_shape=(hsdp_replicate_dim, hsdp_shard_dim),
    mesh_dim_names=("replicate", "shard"))
```
- `replicate` 维：数据并行（多份数据）。
- `shard` 维：参数分片。
- `world_size = replicate_dim × shard_dim`。

策略选择：
- 小模型（Wan 1.3B）：`replicate=8, shard=1`（全复制，无分片通信，速度快）。
- 大模型：`shard>1`（分片省显存）。

## 4. meta 设备加载

```
源码：fsdp_load.py:maybe_load_fsdp_model (L100)
```
```python
with torch.device("meta"):
    model = model_cls(**init_params)      # 只建结构，不分配内存
shard_model(model, mesh=device_mesh)      # 分片
load_model_from_full_model_state_dict(...) # distribute_tensor 分发权重
```
避免加载 14B 模型时先在单卡塞满。

## 5. Mixed Precision

```python
mp_policy = MixedPrecisionPolicy(
    param_dtype=bf16,      # 参数存储/计算精度
    reduce_dtype=fp32,     # 梯度 reduce 精度（保数值稳定）
    cast_forward_inputs=False)
```
参数用 bf16 省显存/加速，梯度 reduce 用 fp32 保稳定。这是训练标配。

## 6. Activation Checkpointing

```
源码：train/models/wan/wan.py:apply_activation_checkpointing
```
不保存中间激活，backward 时重算。用时间换显存——视频 DiT 激活巨大，AC 是训练必需。配置 `enable_gradient_checkpointing_type`（full/selective）。

## 7. CPU Offload

推理时把不用的模块放 CPU，用时再搬 GPU：
- `dit_cpu_offload`, `vae_cpu_offload`, `text_encoder_cpu_offload`。
- `dit_layerwise_offload`：逐层 offload（极致省显存）。
- `pin_cpu_memory`：pin 内存加速传输。

## 8. 精度选择（bf16/fp16/fp8/int8）

| 精度 | 用途 |
|------|------|
| bf16 | 训练/推理默认 |
| fp16 | 部分场景 |
| fp8 | 量化推理（`layers/fp8linear.py`） |
| int8 | TurboDiffusion GEMM |
| fp4/nvfp4 | Blackwell 极致量化 |

## 9. 通信开销

- FSDP：每层 all-gather 参数 + reduce-scatter 梯度。
- 全分片通信多，全复制通信少但显存多。需权衡。

## 10. Checkpoint（DCP）

FSDP 分片模型用 `torch.distributed.checkpoint`（DCP）保存分片状态：
```
checkpoint-1000/dcp/   # 分片数据
```
见 [`../02_source_by_directory/08_training.md`](../02_source_by_directory/08_training.md)。

## 11. 回扣源码
| 概念 | 源码 |
|------|------|
| FSDP2 加载 | `fsdp_load.py:maybe_load_fsdp_model` |
| shard | `fsdp_load.py:shard_model` |
| HSDP mesh | `fsdp_load.py` init_device_mesh |
| AC | `train/models/wan/wan.py` |
| offload | `fastvideo_args.py` 各 offload 字段 |

## 12. 延伸
- 序列并行：[`07_sequence_parallelism.md`](07_sequence_parallelism.md)
- 显存优化：[`13_memory_optimization.md`](13_memory_optimization.md)
