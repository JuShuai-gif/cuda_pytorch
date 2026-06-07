# Lecture 08: Attention 机制详解

## 本讲核心问题

Attention 是 Transformer 的核心——没有它就没有 LLM。但标准 Attention 的 O(N²) 复杂度让它成为显存和计算的黑洞。本讲深入回答：(1) Q/K/V 到底在做什么？(2) Multi-Head Attention 为什么需要多个头？(3) 工业界如何用 GQA/MQA/MLA 降低 KV Cache？(4) FlashAttention 如何在 GPU 上对 Attention 做 tiling？

---

## 通俗解释

**基础 Attention** 就像学生在图书馆查资料时做笔记：
- **Q（Query，查询）** = 学生当前的问题："我现在要写什么？"
- **K（Key，键）** = 每本书的索引标签："这本书讲什么？"
- **V（Value，值）** = 书里的实际内容

学生看一个问题（Q），扫一眼所有书的标签（K），判断哪些书相关（计算相似度），然后从相关的书里提取内容（V）。这就是 Scaled Dot-Product Attention：

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

除以 √d_k 是为了防止内积过大导致 softmax 梯度消失——就像温度过高（内积值大），softmax 会变"硬"，只关注一两个词而忽略其他。

**Multi-Head Attention** 就像派多个学生同时查资料，每人关注不同角度：一个人关注语法结构，一个人关注语义关系，一个人关注位置信息。每个"学生"是一个独立的 Attention Head。

**GQA（Grouped Query Attention）** 是什么？假设图书馆有 32 个学生（Q Heads）同时查资料，但只给 8 本索引目录（KV Heads）。每 4 个学生（G=32/8=4）共享一本目录的不同部分。这和标准 MHA 的 32 套完整 QKV 相比，KV 目录数量减少了 4 倍。

**MLA（Multi-head Latent Attention）**：DeepSeek 的创新——不直接存储 KV，而是将 KV 压缩到一个低维的"潜在空间"（latent space），查询时再解压。相当于把索引目录用超强压缩算法存起来，用的时候解压——牺牲极少的精度换取巨大的显存节省。

---

## 数学公式 + 工程意义

### 1. Scaled Dot-Product Attention 的完整公式

```
A = softmax(QK^T / √d_k) · V

其中：
- Q ∈ R^{N × d_k} （N 个 position，d_k 维 query）
- K ∈ R^{N × d_k} （N 个 position，d_k 维 key）
- V ∈ R^{N × d_v} （N 个 position，d_v 维 value）
- A ∈ R^{N × d_v} （输出）
```

**关键数字**：对于 Llama 3 70B，d_k = 128（每个 head），h = 64 heads（Q），d_model = 8192。Attention 矩阵 QK^T 的大小是 N × N。对于 8K 上下文，这就是 8192² ≈ 67M 个元素，每个 FP16 占 2 bytes，共 **134MB**——这只是**一个 head 的一个 attention 矩阵**。64 个 head 就是 8.4GB，加上 multi-layer（80 layers），中间激活可达数百 GB。

### 2. GQA 和 MQA 的 KV 复用公式

**标准 MHA**：h 个 Q head，h 个 KV head（1:1 对应）
```
Q_i = XW_{Q_i}, K_i = XW_{K_i}, V_i = XW_{V_i}  (i = 1..h)
head_i = Attention(Q_i, K_i, V_i)
```

**GQA**：h_q 个 Q head，h_kv 个 KV head（h_q / h_kv = G 为组大小）
```
Q_{i} = XW_{Q_i}  (i = 1..h_q)
K_{j} = XW_{K_j}, V_{j} = XW_{V_j}  (j = 1..h_kv)
head_i = Attention(Q_i, K_{g(i)}, V_{g(i)})  其中 g(i) = floor(i / G)
```

**MQA** (GQA 的极端特例)：h_kv = 1，所有 Q head 共享一套 KV。

**工程意义**：KV Cache 大小从 MHA 的 `2 × h × d_k × N × layers` 降到 GQA 的 `2 × h_kv × d_k × N × layers`。以 Llama 3 70B（G=8）为例，KV Cache 缩小 8 倍：

| 配置 | h_q | h_kv | 80-layer KV Cache (8K context) |
|------|-----|------|-------------------------------|
| MHA | 64 | 64 | 2 × 64 × 128 × 8192 × 80 ≈ **10.7 GB** |
| GQA (8:1) | 64 | 8 | 2 × 8 × 128 × 8192 × 80 ≈ **1.34 GB** |
| MQA | 64 | 1 | 2 × 1 × 128 × 8192 × 80 ≈ **0.17 GB** |

### 3. MLA (DeepSeek V2/V3) 的压缩原理

MLA 的核心思想：不存储 K,V，存储其低维投影 C_{KV}。

```
C_{KV} = X · W_{down}           // 压缩到 d_c 维（d_c << d_kv = h_kv × d_k）
K = C_{KV} · W_{K,up}           // 使用时解压
V = C_{KV} · W_{V,up}           // 使用时解压（可与 K 共用 W_{up} 或独立）
```

**DeepSeek-V2 的配置**：d_c = 512（latent 维度），d_kv = 128 × 128 = 16384（全维度）。压缩比 = 16384/512 = **32x**！这意味着 KV Cache 从 MHA 的规模缩小了 32 倍——代价是 forward 时需要额外的矩阵乘法解压。

### 4. Sliding Window Attention (Mistral)

```
Attention(Q_i, K_{i-W..i}, V_{i-W..i})
```

只让每个 token 看到前后各 W 个 token（Mistral 用的是 W=4096）。复杂度从 O(N²) 降到 **O(NW)**。当 N=128K，W=4K 时，计算量减少 (128K/4K)² = 1024 倍。

### 5. Causal Mask 的数学形式

```
Mask_{ij} = -∞  if i < j, else 0
A = softmax((QK^T / √d_k) + Mask) · V
```

在 softmax 前加 mask，e^{-∞} = 0，确保 token i 无法 attend 到 token j（j > i）。这是自回归生成的基础。

---

## 工业界真实实现

### Llama 3 的 GQA

Llama 3 使用 8:1 的 Grouped Query Attention。具体配置：
- 8B 模型：32 Q heads，8 KV heads（G=4）
- 70B 模型：64 Q heads，8 KV heads（G=8）
- 405B 模型：128 Q heads，8 KV heads（G=16）

如此激进的 GQA 比率（G=8~16）意味着：KV Cache 极小，推理时 decode 阶段由 memory-bound 的 QK^T 计算变为更 compute-bound。**代价**：G 越大，attention 质量略有下降，但通过增加 Q heads 数量补偿。

### DeepSeek-V2/V3 的 MLA

DeepSeek-V2 首次在 236B MoE 模型中使用 MLA，deep dive 如下：

```
C_{KV} = X · W_dkv_a                           // [N, d_model] -> [N, d_c]
K_compressed = C_{KV} · W_dkv_b_NOPE            // NoPE: no positional encoding
K_positional = C_{KV} · W_uk_tum · W_kr_rope    // RoPE applied separately
K = [K_compressed, K_positional]                // concat along head_dim
V = C_{KV} · W_uk_tum · W_v_rope
```

**核心创新**：RoPE 的旋转矩阵不兼容低秩压缩，因此 DeepSeek 将 RoPE 部分单独处理（decoupled RoPE），只对非位置部分做低秩压缩。这样既保留了 RoPE 的序列外推能力，又实现了 KV 的深度压缩。

### Mistral 的 Sliding Window

Mistral 7B 使用 W=4096 的 sliding window attention。但注意——这**不是**唯一的 attention：Mistral 保留了**全局 attention 层**（每隔几层有一层不做 sliding window），确保模型仍能捕获长距依赖。

### FlashAttention 的 Tiling 思路

标准 Attention 的实现需要将 Q、K、V 全部加载到 SRAM 再计算，但 SRAM 只有 ~30MB（H100），放不下 N > 2048 时的 QK^T 矩阵（2048² × 2 bytes = 8MB 勉强，8192² = 128MB 不行）。

FlashAttention 的做法：
1. 将 Q 分块（tile），每块 Q_block 只与一个大 K 块计算 softmax
2. 维护**运行中的 softmax 统计量**（online softmax）：l（分母的和）、m（最大值）
3. 用 l 和 m 在块间做 re-scaling，实现正确的增量 softmax
4. **不写回 HBM** 中间的 QK^T 矩阵——直接在 SRAM 完成 Attention，只把最终结果写回 HBM

**结果**：HBM 读写量从 O(N²) 降到 O(N)，在 N=8K 时快 2-4 倍，显存节省与 O(N²) 相比是天壤之别。

---

## CUDA/GPU 视角

### Attention 的 Memory Bottleneck

标准 Attention 有三步 I/O 重灾区：
1. **QK^T 计算**：读 Q（N×d_k）、K（N×d_k），写 S = QK^T（N×N）。当 N=8192，S 是 64M 元素，写回需要 128MB HBM 带宽。
2. **Softmax**：读 S（N×N），写 P（N×N），又是 128MB+128MB。
3. **PV 计算**：读 P（N×N）和 V（N×d_v），写 O（N×d_v）。

总计 I/O ≈ 4N² + 4Nd，当 N=8K 时约 256MB（仅一个 head 一个 layer）。对 80 layers × 64 heads = 5120 次这样的操作，总 I/O 超过 1TB。

### FlashAttention 如何优化

FlashAttention 的核心：**Fused Kernel + Tiling + SRAM**。
- 把整个 Attention（matmul + softmax + matmul）融合为**一个 CUDA kernel**
- 在 SM 的 SRAM 中完成 softmax 的块间递增计算
- 只将最终 O 写回 HBM，中间矩阵（S, P）存在 SRAM

GPU 利用率方面：
- 原版 Attention：受 HBM 带宽限制（memory-bound），compute utilization < 10%
- FlashAttention：将 memory-bound 转为 compute-bound，利用 tensor core 做高效 matmul

| 实现 | N=2K | N=4K | N=8K |
|------|------|------|------|
| 标准 Attention | 1.0x | 1.0x | OOM |
| FlashAttention v2 | 2.2x | 3.0x | 4.0x |

---

## 本讲与整个 LLM 系统的关系

Attention 是 LLM 的计算核心和显存瓶颈，直接决定了：
- **模型质量**：GQA/MQA/MLA 的压缩比需要在质量和效率间精调
- **训练成本**：FlashAttention 是训练长上下文模型的必要条件——没有它，训练 128K 上下文会 OOM
- **推理吞吐**：KV Cache 大小决定 batch size 上限——GQA 的 8 倍压缩意味着 8 倍更大的 batch
- **架构设计**：从 MHA → MQA（PaLM）→ GQA（Llama）→ MLA（DeepSeek），KV 压缩是最近两年最大的架构创新方向

---

## 面试问题

1. **为什么要除以 √d_k？** 分析内积的方差为 d_k，softmax 在大值下的梯度饱和问题。

2. **GQA 和 MQA 区别是什么？什么时候选 GQA 而不是 MQA？** 讨论质量 vs 效率的 trade-off。

3. **MLA 为什么需要 decoupled RoPE？** 分析 RoPE 旋转矩阵对低秩分解的不兼容性。

4. **FlashAttention 为什么能减少 HBM 读写？** 描述分块 softmax 的在线统计算法。

5. **Sliding Window Attention 的缺点是什么？** 分析全局信息的丢失和补救方法。

6. **KV Cache 大小如何随 seq_len 和 batch_size 增长？** 给出公式并计算 Llama 3 70B 在 batch=32, seq_len=8K 时的 KV Cache。

7. **为什么 decode 阶段 Attention 是 memory-bound？** 从 arithmetic intensity 角度分析。
