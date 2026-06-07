# Lecture 10: 推理系统 Inference — Part 1: KV Cache & Arithmetic

## 本讲核心问题

LLM 推理和训练截然不同：训练是吞吐量优先（batch 大），推理是延迟敏感（用户等着）。本讲聚焦推理的**两个阶段**（Prefill 和 Decode）、**核心瓶颈**（KV Cache 的显存）和**核心指标**（TTFT、Latency、Throughput）。关键洞察：decode 阶段每步只处理 1 个 token，是极端的 memory-bound 场景——这就是为什么推理优化的核心是 KV Cache 管理和带宽利用。

---

## 通俗解释

把 LLM 推理比作写一篇作文：

**Prefill 阶段**（首次作答）：老师给了作文题目（prompt）——"请写一篇关于人工智能的议论文，800字"。学生**一次性读完**整个题目，理解要求，在脑海中建立上下文框架。这一步很快——学生同时看到所有 20 个字，可以并行处理。

**Decode 阶段**（逐字写作）：学生开始写作文，**一个字一个字**地写。写第 1 个字时需要回顾之前的全部内容；写第 2 个字时需要回顾题目 + 第 1 个字；写第 100 个字时需要回顾题目 + 前 99 个字。越写越慢——每写一个新字，都要重新回顾之前所有内容。

这就是 LLM 推理的两个阶段：prefill 一次性处理整个输入序列（compute-bound），decode 逐 token 生成（memory-bound）。

**KV Cache 的直觉**：
学生在写作时，不需要每次写新字都把之前的内容重新"想一遍"。聪明的做法是把之前想过的内容记在草稿纸上（KV Cache），写新字时直接从草稿纸查。这样：
- 没有 KV Cache：生成第 N 个 token 时，要重新计算前 N-1 个 token 的 K 和 V，计算量 O(N²)
- 有 KV Cache：只需算第 N 个 token 的 Q，然后和草稿纸上的 K 做 attention，计算量 O(N)

**Arithmetic Intensity 的直觉**：
- Prefill：一次性处理 1000 个 token，做大量矩阵乘法（Q×K^T 是 N×N），计算多、数据少 → **compute-bound**
- Decode：处理 1 个 token，读 1000 个 KV cache 的数据，几乎不怎么算 → **memory-bound**

---

## 数学公式 + 工程意义

### 1. Prefill 与 Decode 的 FLOPs 分析

设模型参数量为 P，序列长度为 N，每个 token 的 hidden dimension 为 d。

**Prefill 阶段**（处理 prompt 所有 N_prompt 个 token）：

```
FLOPs_prefill ≈ 2 × P × N_prompt           （每个参数做一次乘加 = 2 FLOPs）
```

例如 Llama 3 70B（P=70B），处理 4096 个 prompt token：
```
FLOPs_prefill = 2 × 70B × 4096 ≈ 5.73 × 10^14 = 573 TFLOPs
```

在 H100（989 TFLOPS BF16）上，理论最快：573 / 989 ≈ 0.58 秒。

**Decode 阶段**（每生成 1 个 token）：

```
FLOPs_decode ≈ 2 × P × 1                   （只处理 1 个新 token）
```

即 140 GFLOPs。但在 H100 上实际时间远大于 140G/989T ≈ 0.14ms。**为什么？** 因为瓶颈不在计算，而在读取 KV Cache 的带宽。

### 2. KV Cache 的内存计算

KV Cache 大小公式（GQA 的通用情况）：

```
KV_Cache_size = 2 × num_layers × num_kv_heads × head_dim × seq_len × dtype_size
```

以 Llama 3 70B（80 layers, 8 KV heads, head_dim=128, BF16 dtype=2 bytes）为例：

| seq_len | KV Cache 大小 | 备注 |
|---------|--------------|------|
| 4K | 2 × 80 × 8 × 128 × 4096 × 2 = **1.34 GB** | 一个请求 |
| 8K | 2 × 80 × 8 × 128 × 8192 × 2 = **2.68 GB** | — |
| 32K | 2 × 80 × 8 × 128 × 32768 × 2 = **10.7 GB** | Llama 3.1 支持 |
| 128K | 2 × 80 × 8 × 128 × 131072 × 2 = **42.9 GB** | Llama 3.1 长上下文 |

**关键约束**：H100 的 80GB HBM 中，模型参数占 140GB（需要量化或 TP 分片），KV Cache 还要占 1-10+ GB per request。如果同时服务 32 个请求（batch=32, seq_len=8K），每请求 2.68GB → 总共需要 **85.8GB KV Cache**——已经超过 80GB！

### 3. Arithmetic Intensity 计算

Arithmetic Intensity = FLOPs / Bytes_loaded_from_HBM

**Decode 阶段**的 Arithmetic Intensity：
- 计算量：约 2P FLOPs（每层计算 Q、K、V、Attention、FFN，但大部分操作量很小）
- 关键：Attention 部分需要读取整个 KV Cache。对一层 Attention：
  - 读 Q (1 × head_dim × num_q_heads × 2 bytes)
  - 读 K_cache (seq_len × head_dim × num_kv_heads × 2 bytes)
  - 读 V_cache (seq_len × head_dim × num_kv_heads × 2 bytes)

当 seq_len = 8192 时，K_cache + V_cache ≈ 2 × 8 × 128 × 8192 × 2 ≈ 33.5 MB/层（Llama 3 70B）。

80 layers 的 KV Cache 总读取量 = 80 × 33.5 MB ≈ **2.68 GB**。

而计算量约 140 GFLOPs。因此：

```
Arithmetic Intensity ≈ 140 × 10^9 FLOPs / 2.68 × 10^9 bytes ≈ 52 FLOPs/byte
```

H100 的峰值 compute throughput 是 989 TFLOPS BF16，HBM 带宽是 3.35 TB/s。算力/带宽比 = 989T / 3.35T ≈ **295 FLOPs/byte**。

52 << 295，说明 decode 是 **memory-bound**——大部分时间在等 HBM 把 KV Cache 搬进 SM，而不是在计算。

**Prefill 阶段**的 Arithmetic Intensity：
- 处理 N_prompt 个 token 同时计算，矩阵运算占主导
- QK^T 是 N×N 矩阵乘法，每加载 (N×d) bytes 做约 O(N²×d) FLOPs
- 当 N 很大时，Arithmetic Intensity >> 295 → **compute-bound**

### 4. 推理指标详解

| 指标 | 定义 | 公式/解释 | 目标值 |
|------|------|-----------|--------|
| **TTFT** (Time To First Token) | 提交 prompt 到第一个 token 出现的时间 | prefill_time | < 500ms（用户体验阈值） |
| **TPOT** (Time Per Output Token) | decode 每生成一个 token 的时间 | decode_step_time | < 50ms（阅读速度匹配） |
| **Latency** | 整个请求的总时间 | TTFT + TPOT × num_output_tokens | — |
| **Throughput** | 每秒处理的请求数或 token 数 | tokens_per_second = batch_size / avg_latency_per_token | 尽量高 |
| **QPS** | 每秒查询数 | requests_per_second | — |

**为什么 TTFT 重要**：用户看不到输出就开始焦虑。研究表明 TTFT > 1 秒时，用户满意度大幅下降。

---

## 工业界真实实现

### vLLM：PagedAttention 驱动的推理引擎

vLLM 的核心理念：KV Cache 不应该像传统实现那样分配为**连续的、固定大小的** tensor，而应该像操作系统管理虚拟内存一样**分页**。

传统 KV Cache 管理的问题：
```python
# Naive: pre-allocate max_seq_len for KV cache
k_cache = torch.zeros(max_batch, max_seq_len, num_kv_heads, head_dim, dtype=bf16)
# 问题：实际只用到当前 seq_len，剩余空间全部浪费（内部碎片）
```

vLLM 的做法：
- KV Cache 被分成固定大小的 **block**（如一页 16 个 token）
- 请求按需分配 block，通过 **block table** 做逻辑到物理的映射
- 不连续分配，零内部碎片，内存利用率 > 95%

vLLM 的吞吐提升：
```
Throughput_vLLM / Throughput_Naive ≈ 1 / (1 - fragmentation_rate)
```
当 fragmentation_rate = 60% 时（常见于长短混合请求），vLLM 可提升 1/(1-0.6) = 2.5x。

### TensorRT-LLM：推理的全栈优化

NVIDIA 的 TensorRT-LLM 提供从图优化到 kernel 的完整优化栈：
1. **Graph optimization**：算子融合（LayerNorm + Attention QKV proj = 1 个 GEMM）
2. **In-flight batching**：请求动态加入和离开 batch（和 vLLM 的 continuous batching 同概念）
3. **Quantization**：FP8/INT8/INT4 KV Cache，减少带宽需求
4. **Custom kernels**：fused MHA、fused MLP

### SGLang：RadixAttention

SGLang 的核心创新是 **RadixAttention**——通过 Radix Tree 数据结构实现 prefix 级别的 KV Cache 自动共享。例如：
- 请求 A："Translate to French: Hello, how are you?"
- 请求 B："Translate to French: Hello, what time is it?"

两个请求共享 "Translate to French: Hello, " 这个 prefix 的 KV Cache。SGLang 自动识别并复用，避免重复 prefill。

---

## CUDA/GPU 视角

### Decode 阶段的 Memory Bottleneck 深度分析

Decode 每步的完整数据流（以 Llama 3 70B 某层为例）：
1. 加载新 token 的 embedding（小，可忽略）
2. **加载该层 KV Cache**：K_cache + V_cache = 每层 33.5 MB → 80 layers = 2.68 GB 全部 HBM 读
3. 计算 Q（1 token 的投影，小）
4. QK^T：Q(1×head_dim) × K_cache(seq_len×head_dim)^T → 小计算，大读取
5. softmax + × V_cache → 小计算，大读取
6. Output projection + FFN：这部分计算量大一些，但模型权重也需要从 HBM 读

**为什么这个问题难**：GPU SM 的计算能力过剩，但 HBM 带宽是硬限制。H100 理论 decode 最大吞吐 = HBM_bandwidth / bytes_per_token。

```
max_tokens_per_second = 3.35 TB/s / bytes_per_token

其中 bytes_per_token ≈ 2 × P_total (模型权重 + KV cache 读取)
对于 KV Cache 很大的场景（长序列），KV 读取主导
```

### 量化如何减少 Memory Bottleneck

| 精度 | KV Cache 每元素字节 | Llama 3 70B 8K KV Cache | decode 步读取 |
|------|---------------------|------------------------|--------------|
| FP16/BF16 | 2 bytes | 2.68 GB | 2.68 GB/step |
| FP8 | 1 byte | 1.34 GB | 1.34 GB/step |
| INT8 | 1 byte | 1.34 GB | 1.34 GB/step |
| INT4 | 0.5 bytes | 0.67 GB | 0.67 GB/step |

FP8 KV Cache 可以**直接将 decode 延迟减半**（在 memory-bound 场景下）。这也是为什么 NVIDIA 在 H100 中加入了 FP8 tensor core 支持。

---

## 本讲与整个 LLM 系统的关系

推理是 LLM 落地的"最后一公里"。训练决定了模型的上限，推理决定了模型能否真正为用户服务：
- **成本**：推理成本（GPU 小时 × 请求数）远超训练成本——ChatGPT 每天推理成本估计是训练成本的 10-100x
- **用户体验**：TTFT > 1s 用户就会流失
- **架构选择**：GQA/MQA 本质上是为了缩小 KV Cache——这是推理驱动的架构决策
- **量化方向**：FP8/INT4 在推理中广泛使用，在训练中还很罕见——推理容忍更低的精度

---

## 面试问题

1. **Prefill 和 Decode 阶段的核心区别是什么？** 从 compute-bound vs memory-bound 角度分析。

2. **为什么需要 KV Cache？没有 KV Cache 的 decode 计算量是多少？** 推导 O(N²) vs O(N) 的复杂度差异。

3. **KV Cache 大小如何计算？** 给出完整公式，用 Llama 3 70B 在 32K 上下文下举例。

4. **为什么 decode 阶段是 memory-bound？** 计算 arithmetic intensity 并与 H100 的硬件比例对比。

5. **TTFT 和 TPOT 分别对应哪个阶段？为什么优化目标不同？**

6. **如何估算一个推理系统的最大吞吐？** 从 HBM 带宽 + 模型大小 + batch size 推导。

7. **量化 KV Cache（如 FP8）为什么能加速 decode？** 分析 memory-bound 场景下带宽节省的收益。
