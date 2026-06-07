# Lecture 03: Transformer 架构

## 本讲核心问题

1. Self-Attention 的 Q、K、V 到底在做什么？为什么是三个矩阵而不是两个？
2. Multi-Head Attention 和 Single-Head 的本质区别是什么？
3. Pre-Norm vs Post-Norm：为什么大模型都改成了 Pre-Norm？
4. SwiGLU 凭什么取代 ReLU？那个额外的门控有什么作用？
5. 为什么说 Transformer 的 O(N^2) attention 是其最大的短板，也是最大的优化空间？

## 通俗解释

### Attention ≈ 开卷考试时动态查重点

想象你在参加一场开卷考试，手里有一整本教科书（context）。每遇到一道题，你不会从头到尾翻一遍，而是：

1. 根据题目要求（Query = "今年 GDP 增长多少？"）快速扫描目录
2. 找到相关章节（Key = 目录中的"第三章：宏观经济"）
3. 到那一章仔细读（Value = 详细的 GDP 数据表格）

Self-Attention 完全一样：
- **Q（Query）**：这个位置"想查什么"（比如代词 "it" 想知道它指代谁）
- **K（Key）**：每个位置"能提供什么"（每个词给出自己的"标签"）
- **V（Value）**：每个位置"真正的内容是什么"（词的实际语义信息）

### 为什么 Q、K、V 是三个独立矩阵？

这是一个很好的面试问题。核心原因是**解耦**：

- **Q 和 K 在同一个空间比较相似度**：`softmax(Q * K^T)` 衡量"这个位置应该关注哪些位置"
- **V 在另一个空间提供内容**：`Attention(Q,K,V) = softmax(Q*K^T) * V`，V 的投影决定了"关注到了什么内容"

如果 K=V（共享矩阵），相当于强制"标签"和"内容"完全一样，限制了表达能力。多任务场景下，一个词可能同时扮演不同的角色——比如 "bank" 作为 K 表示"金融机构"，但作为 V 传递的可能是"河岸"的语义。

### Multi-Head Attention ≈ 多个评委给分

一个注意力头可能会关注"语法关系"（主语-谓语），另一个头关注"长距离指代"（it -> elephant），还有一个头关注"语义相似度"。多个头的结果拼起来再做一次线性变换，相当于"综合所有评委的意见给出最终得分"。

### Transformer 比 RNN 好在哪里

RNN 处理序列必须按顺序：处理第 100 个词时，必须等前面 99 个词都处理完。Transformer 可以一次性看到整个序列，所以：

1. **并行化**：所有位置同时计算，不像 RNN 要一个接一个
2. **长距离依赖**：第 1 个词和第 1000 个词之间的信息只需要 O(1) 步 attention，而 RNN 需要 O(1000) 步
3. **可解释性**：attention weights 可以直观看到模型在关注什么

打个比方：RNN 是排队过安检，一个人一个人过；Transformer 是所有人同时出示身份证，安检员同时核对所有人的信息。

## 数学公式 + 工程意义

### Self-Attention 公式

```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
```

- **`Q * K^T`**：计算每对位置之间的相似度，得到一个 N x N 矩阵
- **`/ sqrt(d_k)`**：缩放因子。如果不除，当 d_k=64 时 `Q*K^T` 的元素方差是 64，softmax 后会趋近 one-hot（梯度消失）。除以 sqrt(d_k) 把方差压回 1
- **softmax**：把相似度变成概率分布（每行和为 1）
- **`* V`**：用概率分布对 V 做加权平均，得到每个位置的"上下文表示"

工程意义上，`softmax(Q*K^T)` 这个 NxN 矩阵就是 attention 的 O(N^2) 来源——一个 128K 上下文的 attention matrix 有 16B 个元素，fp16 下就是 32GB。

### Layer Normalization

```
LayerNorm(x) = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta
```

为什么 Transformer 需要 LayerNorm？因为深度网络中的"梯度爆炸/消失"问题——第 30 层的激活值可能比第 1 层大 100 倍。LayerNorm 强制每层的输出具有稳定的均值和方差，让梯度平稳传播。

**Pre-Norm** (先 Norm 再 attention/MLP)：

```
x = x + Attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

**Post-Norm** (先 attention/MLP 再 Norm，原始论文的做法)：

```
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + MLP(x))
```

为什么大模型都用 Pre-Norm？因为 pre-norm 在初始化时让残差路径近于 identity，训练初期梯度传播更稳定。Post-Norm 在大模型（>1B）上训练时容易梯度爆炸/消失。这是 LLM 扩展的**关键经验发现**。

### SwiGLU

```
SwiGLU(x) = Swish(x * W_g) * (x * W_u)  # gate * value
          = (x * W_g * sigmoid(x * W_g)) * (x * W_u)
```

对比 ReLU：`ReLU(x) = max(0, x)`。SwiGLU 的优势：

- ReLU 在 x<0 时梯度为 0（dead neurons），SwiGLU 的 Swish 在负区间有非零梯度
- 门控机制（gate * value）让网络学会"选择性通过信息"，而非粗暴截断
- PaLM 论文的消融实验：SwiGLU 在同参数量下比 ReLU 提升 ~1% perplexity

代价是参数量增加——SwiGLU 需要额外的 gate 矩阵，d_ff 通常从 4x 降到 8/3 * d_model 来平衡参数量。

### Residual Connections

```
output = Layer(x) + x  # Layer = Attention or MLP
```

残差连接让梯度有"短路路径"直接回传。没有残差，100 层的网络梯度要乘 100 次雅可比矩阵，大概率消失或爆炸。有残差，梯度至少有一条直接通道。这就像高速公路旁边有一条备用小路——高速堵了（梯度消失），还能走小路（残差）。

## 工业界真实实现

### Llama 架构的精简设计

Llama 2 在原始 Transformer 基础上的改进：

```
# Pseudo-code for one Llama block
def llama_block(x):
    # Pre-norm with RMSNorm (NOT LayerNorm)
    normed = rms_norm(x)

    # Query, Key, Value projections
    q, k, v = W_q(normed), W_k(normed), W_v(normed)

    # Rotary Position Embedding (RoPE), NOT learned positional encoding
    q, k = apply_rotary_embedding(q, k, position_ids)

    # Attention + residual
    attn_out = scaled_dot_product_attention(q, k, v, mask=causal_mask)
    x = x + attn_out

    # MLP with SwiGLU
    normed = rms_norm(x)
    gate = sigmoid(W_g(normed) * W_u(normed))  # Silu gate
    mlp_out = gate * W_d(W_u(normed))
    x = x + mlp_out

    return x
```

关键差异：

| 特性 | 原始 Transformer | Llama |
|------|-----------------|-------|
| Normalization | LayerNorm (post-norm) | RMSNorm (pre-norm) |
| Activation | ReLU | SiLU (Swish) |
| MLP | FFN(ReLU) | SwiGLU |
| Position encoding | Sinusoidal / Learned | RoPE |
| Weight tying | 无 | 无（input/output embedding 分开） |

### RMSNorm 为什么比 LayerNorm 快

```
LayerNorm:  y = gamma * (x - mean) / std + beta  # needs mean AND std
RMSNorm:    y = gamma * x / rms(x)                # only needs RMS
```

RMSNorm 省去了减均值的操作（约 15-20% layer normalization 计算量），实验表明效果一样甚至更好。在推理阶段，每个 token 的 LayerNorm 是 memory bound 的——省掉减均值意味着少一次 HBM 读写。

### Parallel Attention + FFN (PaLM 风格)

Llama 和标准 Transformer 都是串行的——先 attention 再 MLP。PaLM 提出并行版本：

```
x = x + Attention(RMSNorm(x)) + MLP(RMSNorm(x))  # 共享同一个 norm
```

这样可以同时计算 attention 和 MLP，进一步利用 GPU 并行性。不过 Llama 没有采用，可能因为独立 norm 的灵活性更重要。

### DeepSeek-V2/V3 的 MLA (Multi-head Latent Attention)

DeepSeek 引入 MLA 来降低 KV cache 大小。核心思想：

```
k = W_dk * c_kv  # d_kv << d_model, 先压缩
v = W_dv * c_kv  # shared latent representation

# KV cache 存的是 c_kv (latent vector)，不是完整的 k, v
# 推理时从 latent -> k, v 只需要一次小 matmul
```

MLA 将 KV cache 从 `2 * layers * heads * d_head` 降到 `layers * d_c`（d_c 是 latent 维度），在长上下文推理中节省 5-10x KV cache 显存。

### MQA (Multi-Query Attention) 和 GQA (Grouped-Query Attention)

- **MHA**：head 数量 = KV head 数量（标准做法）
- **MQA**：所有 Q heads 共享 1 组 K,V → KV cache 减少 head 数量倍，但质量略有下降
- **GQA**：Q heads 分组共享 K,V → 折中方案。Llama 2-70B 使用 GQA (8 KV heads for 64 Q heads)

GQA 在质量和效率之间取得了最佳平衡：

```
KV cache size (GQA) = KV_cache (MHA) * (n_kv_heads / n_q_heads)
```

## CUDA/GPU 视角

### Attention 的 O(N^2) memory 问题

一个 batch 中单个 head 的 attention score 矩阵：

```
scores = Q @ K.T  # shape: [batch, head, S, S]
```

对于 S=128K, fp16：

```
size = 2 * 128K^2 = 2 * 16,384,000,000 = 32 GB per head!
```

这就是为什么**标准 attention 在长上下文时根本没法在 GPU 上跑**。FlashAttention 的解决方案是不把完整的 NxN attention matrix 写入 HBM，而是在 SRAM 内 tiling 计算——本质上是用 compute 换取 memory。

### Pre-norm 的 CUDA 效率

Pre-norm 在 forward 时，norm 的输出可以直接送入 attention kernel，不需要额外的 HBM 写入——这就是 **kernel fusion** 的基础。相比之下 post-norm 必须在残差加法之后才能 norm，增加了一次 HBM round-trip。

### Attention kernel 的 memory access pattern

标准 attention 的 memory access：

```
QK^T:  Read Q from HBM (B*H*S*d_head)
       Read K from HBM (B*H*S*d_head)
       Write attention scores (B*H*S*S) to HBM  <-- the bottleneck!

Softmax: Read scores from HBM, write probs to HBM

Prob * V: Read probs, read V, write output
```

FlashAttention 通过 tiling 将中间结果（scores, probs）完全留在 SRAM 内，只把最终输出写回 HBM。这消除了 O(N^2) 的 HBM 写入，将 memory access 从 O(N^2) 降到 O(N)。

## 本讲与整个 LLM 系统的关系

```
Tokenizer -> Embedding -> Attention -> MLP -> Loss -> Optimizer -> Distributed -> Inference
                            ^^^^^^^^^^^^^^
                             本讲核心 |
```

Transformer 架构是 LLM 的"发动机"——决定模型怎么从输入计算输出。它的设计选择直接影响：
- **训练速度**：Pre-norm vs post-norm 决定训练稳定性
- **推理成本**：MHA vs GQA vs MLA 决定 KV cache 大小
- **扩展性**：O(N^2) attention 决定 context length 上限

理解 Transformer 不是为了"会推导公式"，而是为了**知道每个组件的计算/显存瓶颈在哪里**——这才是做系统优化的前提。

## 面试问题

**Q1: 为什么 Pre-norm 让大模型训练更稳定？**

A: Pre-norm 中残差路径是 identity + f(Norm(x))。初始化时 f(Norm(x)) 接近 0，残差路径近于恒等映射。梯度反向传播时，即使 f(Norm(x)) 的梯度很小，残差路径也有恒等梯度 1。Post-norm 是 Norm(x + f(x))，在训练初期 f(x) 可能很大，Norm 后梯度被压缩。对于深网络，pre-norm 每层梯度的方差更稳定，post-norm 梯度在前几层容易爆炸。

**Q2: MHA、MQA、GQA 的显存和计算对比？**

A: 以 70B 模型 (64 Q heads, h_dim=128, n_kv_heads 可变) 为例：

| 类型 | n_kv_heads | KV Cache per token | 质量影响 |
|------|-----------|-------------------|----------|
| MHA  | 64        | 64*4*128*80 = 2.6 MB | Baseline |
| GQA  | 8         | 8*4*128*80 = 327 KB | 几乎无损 |
| MQA  | 1         | 1*4*128*80 = 41 KB  | 部分任务下降 |

GQA 用 8x 节省取得 ~99% 的质量。MQA 虽然更省，但在多任务 fine-tuning 时质量下降明显。

**Q3: RoPE 为什么比绝对位置编码好？**

A: RoPE 通过旋转矩阵编码相对位置——两个 token 的 attention score 只取决于它们的相对距离，不依赖绝对位置。这意味着模型天然支持外推（训练时 S=4096，推理时可以扩展到 32K+）。绝对位置编码（learned/sinusoidal）把位置信息加到 embedding 上，外推能力弱。RoPE 的数学本质是：通过在复数空间旋转 Q 和 K 向量，使点积结果仅依赖相对角度差。

**Q4: FlashAttention 为什么不在所有情况下都更快？**

A: FlashAttention 通过 tiling 避免 HBM 读写 O(N^2) 的 attention matrix，但增加了计算量（每个 tile 需要 partial softmax rescaling）。当序列较短（N < 512）时，计算开销可能超过内存节省，标准 attention 反而更快。此外，FlashAttention 要求 batch 和 head 维度完全并行，对小 batch 推理场景的加速有限。但对于长上下文（N > 2K），FlashAttention 通常是严格更优的。
