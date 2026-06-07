# Lecture 02: 资源计算 (Resource Accounting)

## 本讲核心问题

1. 一个 70B 参数的模型，用 fp16 训练，到底需要多少显存？
2. 为什么 bf16 比 fp16 更适合训练？"动态范围"到底是什么？
3. 一次训练 step 的 FLOPs 如何精确计算？
4. Mixed precision training 中，为什么只有 optimizer states 和 master weights 用 fp32？
5. 给定 8 张 H100 (80GB)，能训练一个 70B 模型吗？如果能，batch size 能开到多大？

## 通俗解释

### 混合精度 ≈ 算账用计算器，数钱用算盘

训练大模型需要存很多东西：模型参数（weights）、梯度（gradients）、优化器状态（optimizer states）、中间激活（activations）。这就像开一家银行：

- **模型参数（fp16）**：这是你的账本，用半精度存着——够了，因为每次 forward/backward 只是读写，不需要极高精度
- **优化器状态（fp32）**：这是你的计算器，更新参数时需要精确计算。Adam 需要存一阶动量 m 和二阶动量 v，每个都是 fp32
- **Master weights（fp32）**：这是你的金库，最终版的参数存在这里——宁可多占空间，不能丢精度
- **Activations（fp16）**：forward 过程中的中间结果，占显存大户

所以混合精度的核心思想是：**数据存储用低精度省空间，关键计算用高精度保质量**。这就好比你平时用算盘（fp16）记账足够，但月底做报表时一定要用计算器（fp32）。

### bf16 比 fp16 更稳定——为什么？

fp16 的表示范围是 `[-65504, 65504]`，超出就 overflow 变成 NaN。bf16 的表示范围是 `[-3.39e38, 3.39e38]`，和 fp32 一样大。差异在于：

```
fp16: 1 bit sign | 5 bits exponent | 10 bits mantissa  -> 范围小，精度高
bf16: 1 bit sign | 8 bits exponent | 7 bits mantissa   -> 范围大，精度低（但够用）
```

训练大模型时，梯度值可能很大（比如几千），fp16 就直接炸了。bf16 的范围和 fp32 一样大，所以**不需要 loss scaling** 就能稳定训练。

打个比方：fp16 是把尺子——量 1 到 100 毫米的东西非常精确，但超过 100 毫米就量不了。bf16 是一把卷尺——虽然刻度粗一些（7 位精度 vs 10 位），但是量 1 毫米到 1 公里都没问题。

## 数学公式 + 工程意义

### 数值格式的显存占用

| 格式 | bits/param | 70B 模型显存 | 动态范围 | 用途 |
|------|-----------|-------------|---------|------|
| fp32 | 32 | 280 GB | 10^38 | optimizer states, master weights |
| fp16 | 16 | 140 GB | 65504 | forward/backward activations |
| bf16 | 16 | 140 GB | 10^38 | 训练 activations（首选） |
| fp8  | 8  | 70 GB  | 240 | H100 推理，部分训练 |
| fp4  | 4  | 35 GB  | 8 | BitDelta 等极低精度推理 |

### 训练 FLOPs 精确计算

一个 Transformer 层的 forward FLOPs：

```
attention_flops = 4 * B * S * d_model^2   (Q, K, V proj + output proj)
                 + 2 * B * S^2 * d_model  (attention scores + weighted sum)
mlp_flops       = 2 * B * S * d_model * d_ff * 2  (up + gate + down, 2x for matmul)

# Simplified: For standard Transformer with d_ff = 4*d_model
mlp_flops       ≈ 16 * B * S * d_model^2
layer_flops     ≈ 20 * B * S * d_model^2
```

总的 forward FLOPs = N_layers * 20 * B * S * d_model^2。Backward 约是 forward 的 2 倍。所以一个完整的训练 step：

```
FLOPs_per_step ≈ 3 * forward_flops ≈ 60 * N_layers * B * S * d_model^2
```

更常用的近似公式（Kaplan et al. 2020）：

```
FLOPs_per_step ≈ 6 * B * S * N_params
```

为什么是 6？因为 forward 是 2 倍 N_params（每个参数做一次乘法和一次加法），backward 是 4 倍（计算梯度和对输入的梯度）。

对 Llama 2-70B 模型，S=4096, B=1：

```
FLOPs_per_step = 6 * 1 * 4096 * 70e9 = 1.72e15 FLOPs = 1.72 PFLOPs
```

### 训练总显存公式

```
total_memory = model_params * bytes_per_param          # fp16: 2
             + gradients * bytes_per_grad               # fp16: 2
             + optimizer_states * bytes_per_state       # fp32: 4*2=8 (m+v)
             + activations * bytes_per_activation       # depends on B, S
```

对于 Llama 2-70B (fp16 + Adam)：

```
model + grads  = 70B * 4 bytes = 280 GB
optimizer      = 70B * 8 bytes = 560 GB   (m in fp32 + v in fp32)
activations    = variable (batch * seq * layers * d_model * 2 bytes)
---------------------------------------------------------------
total (before parallelism) ≈ 840+ GB
```

这就是为什么 70B 模型不能在单卡上训练——一张 H100 只有 80GB。即使用 8 张卡做 tensor parallelism，也需要 gradient accumulation + activation checkpointing 来进一步压缩显存。

### Gradient Accumulation

如果 global batch size = 128，但每卡只能放 micro batch = 4：

```
steps_per_update = 128 / (4 * 8_cards) = 4
```

前 3 个 micro-batch 只累积梯度不做 optimizer step，第 4 个 micro-batch 完成后统一更新。total FLOPs 不变，但显存峰值只取决于 micro batch size。

### Activation Checkpointing (Gradient Checkpointing)

Forward 时不存所有中间激活，只存 checkpoint 的激活。Backward 时从 checkpoint 重新 forward 一次恢复中间激活。时间复杂度增加 ~33%（多跑一次 forward），但显存从 O(N_layers) 降到 O(sqrt(N_layers))。

### Roofline Model

```
Arithmetic Intensity (AI) = FLOPs / bytes_accessed
```

- AI < "ridge point" → memory bound（瓶颈是带宽）
- AI > "ridge point" → compute bound（瓶颈是算力）

对于 H100 (SXM)：
- Peak BF16: 989 TFLOPS
- HBM bandwidth: 3.35 TB/s
- Ridge point = 989e12 / 3.35e12 = 295 FLOPs/byte

Elementwise 操作（ReLU, LayerNorm）通常 AI 很低（<10），是 memory bound。Matmul 操作 AI 很高（batch * seq / d_model 量级），通常是 compute bound。

## 工业界真实实现

### AMP (Automatic Mixed Precision) — PyTorch 实现

```python
import torch

scaler = torch.cuda.amp.GradScaler()  # Only needed for fp16, not bf16

with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()   # Scale loss to prevent fp16 underflow
scaler.step(optimizer)          # Unscale before step
scaler.update()                 # Adjust scale factor
```

对于 bf16 训练，`torch.cuda.amp.autocast(dtype=torch.bfloat16)` 可以直接用，不需要 `GradScaler`——因为 bf16 的动态范围和 fp32 一样大。

### DeepSpeed ZeRO 的显存优化

DeepSpeed 的显存分为三阶段优化：

- **ZeRO-1**：优化器状态分片（每张卡存 1/N 的 optimizer states）
- **ZeRO-2**：+ 梯度分片（每个 rank 只存 1/N 的梯度）
- **ZeRO-3**：+ 参数分片（forward/backward 时从其他 rank 收集需要的参数）

对于 70B 模型：

| 策略 | 显存/GPU (8 GPUs) | 是否可行 |
|------|---------------------|---------|
| 无优化 | 840+ GB / 8 = 105 GB | No (>80GB) |
| ZeRO-1 | 210 GB / 8 = 26 GB | Yes |
| ZeRO-3 | ~10 GB | Yes, 还能开大 batch |

### Llama 训练配置 (Meta)

Meta 训练 Llama 2-70B 的实际配置：

- 2048 x A100-80GB (写了但实际用了更多)
- Global batch size: 4M tokens（~1024 sequences of 4096 tokens）
- AdamW, beta1=0.9, beta2=0.95
- Cosine LR schedule, warmup 2000 steps
- bf16 + fp32 master weights
- Gradient clipping: 1.0

### DeepSeek-V3 的训练配置

- GPU: 2048 x H800 (80GB)
- 训练精度: fp8 (forward) + bf16 (gradients) + fp32 (optimizer)
- 使用 block-wise fp8 quantization 来减少通信量
- MoE 架构：每个 token 只激活 37B / 671B 总参数

## CUDA/GPU 视角

### fp16 的 loss scaling 为什么需要

fp16 可表示的最小正值是 2^(-24) ≈ 5.96e-8（denormal 之前是 6.1e-5）。训练后期梯度可能小到 1e-7，在 fp16 下直接变成 0（underflow）。解决办法是用 `GradScaler` 把 loss 乘以一个大数（比如 2^16），让梯度放大到 fp16 可表示范围内，optimizer step 之前再除回来。

bf16 可表示的最小正值是 2^(-126) ≈ 1.17e-38，比 fp16 小 30 个数量级，所以几乎不会 underflow。这就是 bf16 在训练中更受欢迎的底层原因。

### HBM vs SRAM

H100 有 80GB HBM，带宽 3.35 TB/s。但每次 matmul 之前，需要把数据从 HBM 搬到 SRAM（L1/shared memory）。SRAM 只有 256KB/SM，但带宽是 19.5 TB/s（H100 L1）。

**Fused kernel 的目的就是把多次 HBM 读写合并成一次**：把多个操作（比如 matmul + bias + activation）在 SRAM 内部完成，只把最终结果写回 HBM。

### Compute bound vs Memory bound

判断标准：对比 peak FLOPs 和 peak bandwidth：

```
compute_time = FLOPs / peak_TFLOPS
memory_time  = bytes / bandwidth_TBps
```

实际耗时 = max(compute_time, memory_time)。优化方向取决于哪个是瓶颈：
- Memory bound → fused kernel, quantization (fp8/int8 减少字节)
- Compute bound → tensor core 优化, 提高 occupancy

## 本讲与整个 LLM 系统的关系

```
Tokenizer -> Embedding -> Attention -> MLP -> Loss -> Optimizer -> Distributed -> Inference
                                                              ^
                                                        本讲（核心）|
```

Resource accounting 是整个 LLM 系统设计的**数学基础**。没有精确的显存/FLOPs 计算，分布式训练策略、batch size 选择、精度选择都是拍脑袋。这不仅是训练阶段的事——推理时 KV cache 的显存规划同样依赖这些计算。

## 面试问题

**Q1: 8 张 H100 (80GB)，用 fp16 + Adam，能训练一个 70B 模型吗？如果能，micro batch size 能开到多大？**

A: 总显存 = 8 x 80 = 640 GB。纯 fp16 (no optimizer)：70B x 2 = 140 GB。加上 optimizer states (fp32)：70B x 8 = 560 GB。需要 (140 + 560 + grad + act) / 8 = 87.5+ GB/GPU。必须用 ZeRO 或 FSDP。用 ZeRO-2（optimizer + grad sharded）：total = 140 + (8+2)/8 * 70B ≈ 140 + 87.5 = 227.5 GB，per GPU = 28.4 GB。剩余 51.6 GB 给 activations。对于 Llama 架构 (d_model=8192, n_layers=80), activation per micro batch (B=1, S=4096) ≈ 80 * 1 * 4096 * 8192 * 2 bytes / 8 GPUs ≈ 0.67 GB（用了 activation checkpointing），因此 micro batch 可以开到 4-8。

**Q2: 为什么大模型训练都用 bf16 而不是 fp16？**

A: bf16 的 exponent 位数（8 bits）和 fp32 一样，动态范围相同。训练时梯度可能很小（1e-7）也可能很大（1e3），fp16 容易 overflow/underflow 需要 loss scaling，而 bf16 天然不会。此外 bf16 不需要 GradScaler，代码更简洁，也避免了 scaling factor 调节不当导致的训练不稳定。

**Q3: 模型参数中的 "activation" 为什么占那么多显存？如何计算？**

A: 对于每层 Transformer，forward 需存储 Q/K/V 值、attention scores、MLP hidden states。以 Llama 2-70B (d_model=8192, n_layers=80, S=4096)：activation per layer = 2 * B * S * d_model (pre-norm) + 4 * B * S * d_model (QKVO) + 2 * B * H * S * S (attention scores) + 2 * B * S * d_ff (MLP)。总计约 80 * B * 4096 * 8192 * 10 = 26.8 GB * B ——不用 checkpointing 时这就是激活显存。Gradient checkpointing 可以将 activation 从 O(n_layers) 降到 O(sqrt(n_layers)) 或 O(1)（取决于 checkpoint 策略）。

**Q4: FLOPs 利用率 (MFU) 的计算方式和行业标准是什么？**

A: MFU = measured_FLOPs / peak_FLOPs。Peak FLOPs 是硬件理论峰值（H100 BF16: 989 TFLOPS），measured FLOPs 通过公式 `6 * B * S * N_params` 或通过 profiler 获取。工业界训练大模型的 MFU 通常在 45-55%（如 DeepSeek 报告的约 54%）。低于预期通常意味着通信开销（跨 GPU all-reduce）或 kernel launch overhead。推理的 MFU 通常只有 1%-3%——因为自回归生成的每个 step 只处理一个 token，compute intensity 极低，完全是 memory bound。
