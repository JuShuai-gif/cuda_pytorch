# 背景知识：FLOPs 计算、Roofline Model 与 Memory Bandwidth

## 1. Transformer FLOPs 计算

### 1.1 为什么计算 FLOPs？

- **训练资源预估**：知道需要多少 compute 才能规划 budget
- **Scaling Law 基础**：Kaplan/Chinchilla 都基于 FLOPs
- **优化目标**：知道哪个操作最耗算力，才能针对性优化

### 1.2 Forward Pass FLOPs 推导

设模型参数：

| 参数           | 符号     | 示例 (LLaMA-7B) |
| -------------- | -------- | ---------------- |
| vocab size     | $V$      | 32,000           |
| hidden dim     | $d$      | 4,096            |
| FFN hidden     | $d_{ff}$ | 11,008           |
| num layers     | $L$      | 32               |
| num heads      | $h$      | 32               |
| head dim       | $d_h$    | 128              |
| sequence len   | $s$      | 2,048            |

**每个 Transformer Layer 的 FLOPs：**

| 操作                | 矩阵维度             | FLOPs (per token)             |
| ------------------- | -------------------- | ----------------------------- |
| QKV 投影            | $d \times 3d$        | $6d^2$                        |
| Attention scores    | $s \times d_h$       | $2s \cdot d_h \cdot h = 2sd$  |
| Attention output    | $d \times d$         | $2d^2$                        |
| Output projection   | $d \times d$         | $2d^2$                        |
| FFN gate/up         | $d \times 2d_{ff}$   | $4d \cdot d_{ff}$             |
| FFN down            | $d_{ff} \times d$    | $2d \cdot d_{ff}$             |
| **Total per layer** |                      | $\approx 12d^2 + 6d \cdot d_{ff} + 2sd$ |

**总 Forward FLOPs (忽略 embedding)：**

$$C_{\text{forward}} \approx L \cdot (12d^2 + 6d \cdot d_{ff}) \cdot s$$

LLaMA 类模型 ($d_{ff} \approx 8/3 \cdot d$)：

$$C_{\text{forward}} \approx L \cdot (12d^2 + 16d^2) \cdot s = 28 d^2 \cdot L \cdot s$$

### 1.3 Backward Pass FLOPs

Backward 需要计算 gradients，约等于 forward 的 **2 倍**（rough estimate）：

$$C_{\text{total}} \approx 3 \times C_{\text{forward}}$$

更精确的计算（Kaplan scaling law paper）：

$$C \approx 6N \cdot s$$

其中 $N$ 是模型参数量（不含 embedding），$s$ 是训练 token 数。

### 1.4 Embedding 层的 FLOPs

在实际系统中，embedding 和 lm_head 的计算量通常**可以忽略**（相对于 self-attention 和 FFN）：

$$C_{\text{embed}} \approx 2d \cdot V$$

对大 vocab（32K+）来说 embedding FLOPs 占比 < 1%。

---

## 2. Memory Accounting

### 2.1 训练时的 Memory 组成

#### (a) Model States

| 组件            | 类型       | Byte per param (FP16 mixed precision) |
| --------------- | ---------- | ------------------------------------- |
| Parameters      | fp16       | 2                                     |
| Gradients       | fp16       | 2                                     |
| Optimizer (Adam) param | fp32 | 4                                     |
| Optimizer (Adam) m     | fp32 | 4                                     |
| Optimizer (Adam) v     | fp32 | 4                                     |
| **Total**       |            | **16**                                |

> 纯 FP32 训练：4 + 4 + 4 + 4 + 4 = 20 bytes/param
> 混合精度 + Adam：2 + 2 + 4 + 4 + 4 = 16 bytes/param

#### (b) Residual States (Activations)

- 每个 transformer layer 存储：input、attention softmax、dropout mask 等
- Activation memory 随 sequence length 和 batch size 线性增长
- 大模型中 activation memory 可能远超 parameter memory

#### (c) Temporary Buffers

- 全精度梯度累加 buffer
- AllReduce 通信 buffer (DDP)

### 2.2 推理时的 Memory

推理只需要存储 **parameters** (FP16: 2 bytes/param) + **KV Cache**：

$$\text{KV Cache bytes} = 2 \times L \times s \times d \times 2 \text{ (bytes/fp16)} = 4Lsd$$

对于 LLaMA-7B ($L=32$, $d=4096$, $s=2048$): KV Cache ≈ 1 GB

### 2.3 Megatron-LM 的 Memory Formula

Megatron-LM 论文给出了分布式训练内存公式：

$$M = \frac{N \cdot B}{P} + A \cdot \frac{B}{P}$$

其中 $N$ = model params, $B$ = bytes per param, $A$ = activation memory, $P$ = parallelism degree

---

## 3. Roofline Model

### 3.1 核心概念

Roofline Model 将操作分类为两类瓶颈：

| 瓶颈类型        | 定义                                        | 优化方向             |
| --------------- | ------------------------------------------- | -------------------- |
| **Compute-bound** | 受限于 GPU FLOPS (peak compute)            | 增加 arithmetic 密度  |
| **Memory-bound**  | 受限于 GPU memory bandwidth                | 减少内存访问          |

### 3.2 Arithmetic Intensity (AI)

$$\text{AI} = \frac{\text{FLOPs}}{\text{Bytes Transferred}}$$

- **高 AI** → compute-bound（如大矩阵乘法）
- **低 AI** → memory-bound（如 element-wise ops, reduction）

### 3.3 GPU 规格参考

| GPU               | FP16 TFLOPS | HBM Bandwidth (GB/s) | Ridge Point (FLOP/byte) |
| ----------------- | ----------- | -------------------- | ----------------------- |
| A100 80GB         | 312         | 2,039                | ~153                    |
| H100              | 989         | 3,352                | ~295                    |
| RTX 4090          | 82.6        | 1,008                | ~82                     |

Ridge Point = Peak FLOPS / Peak Bandwidth。操作 AI > Ridge Point 则 compute-bound。

### 3.4 Transformer 中关键操作的 AI

| 操作                 | FLOPs            | Bytes        | AI      | 瓶颈          |
| -------------------- | ---------------- | ------------ | ------- | ------------- |
| MatMul (large)       | $O(d^2 \cdot s)$ | $O(d^2)$     | $O(s)$  | compute-bound |
| Attention scores     | $O(d \cdot s^2)$ | $O(s^2)$     | $O(d)$  | compute-bound |
| Softmax/LayerNorm    | $O(d \cdot s)$   | $O(d \cdot s)$ | $O(1)$  | memory-bound  |
| GELU/Swish (element)| $O(d \cdot s)$   | $O(d \cdot s)$ | $O(1)$  | memory-bound  |
| Dropout             | $O(d \cdot s)$   | $O(d \cdot s)$ | $O(1)$  | memory-bound  |

> **关键洞察**：Transformer 中的大部分 FLOPs 来自大矩阵乘法（compute-bound），但大部分 kernel launch 来自 element-wise ops（memory-bound）。这就是为什么 **kernel fusion** 如此重要。

---

## 4. 从 FLOPs 到训练时间

### 4.1 Model FLOPs Utilization (MFU)

MFU 衡量实际利用率：

$$\text{MFU} = \frac{\text{实际 FLOPs/s}}{\text{理论峰值 FLOPs/s}}$$

- 大模型训练 MFU 通常在 30-50%
- 最优情况下（如 Megatron-LM + A100 集群）可达 55-60%

### 4.2 训练时间估算

$$\text{训练时间} = \frac{C_{\text{total}}}{\text{GPU count} \times \text{Peak TFLOPS} \times \text{MFU}}$$

例如：LLaMA-7B 训练 1T tokens 在 2048 张 A100 上的时间约为 3-5 天。

---

## 5. 核心公式速查

| 公式                    | 表达式                                                     |
| ----------------------- | ---------------------------------------------------------- |
| Forward FLOPs (approx)  | $C_f \approx 28 \cdot d^2 \cdot L \cdot s$                 |
| Total FLOPs (Kaplan)    | $C \approx 6N s$                                           |
| Model states memory     | $M_{\text{model}} = N \times 16$ bytes (FP16 mixed)        |
| KV Cache memory         | $M_{\text{kv}} = 4Lsd$ bytes                               |
| Arithmetic Intensity    | $\text{AI} = \frac{\text{FLOPs}}{\text{Bytes}}$            |
| MFU                     | $\text{MFU} = \frac{\text{observed FLOPS}}{\text{peak FLOPS}}$ |
