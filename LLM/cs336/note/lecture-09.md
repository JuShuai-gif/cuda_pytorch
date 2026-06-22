# Lecture 09: Normalization & Activation

## 本讲核心问题

深度神经网络的训练稳定性极其脆弱。当网络变深（LLM 常达 80-120 层），各层的激活值分布会逐渐偏移——中间层的值可能爆炸到数千，也可能坍缩为零。Normalization（归一化）和 Activation（激活函数）就是解决这一问题的两把钥匙。本讲回答：(1) LayerNorm/BatchNorm/RMSNorm 的原理和差异？(2) Pre-norm 和 Post-norm 哪个更稳定？(3) 为什么 SwiGLU 成了现代 LLM 的标配激活函数？

---

## 通俗解释

**Normalization（归一化）** 说白了就是"统一标准"。

想象一个大厨房里，每个厨师（每个 Layer）在做一道菜。如果不做归一化，厨师 A 做的菜咸得要命（输出值 1000），厨师 B 做的菜淡得没味（输出值 0.001）。传到下一个厨师那里，他拿到 1000 和 0.001 混在一起，完全无法正常工作——要么被大值主导，要么小值被忽略。

**LayerNorm** 的做法是：不管输入端给了什么，每个厨师把自己的所有原材料（这一层的所有 hidden dimension）重新调整到均值 0、方差 1，再乘以可学习的 scale（γ）和 shift（β）。相当于说："不管上道工序怎么样，到我这里先标准化，再按我的配方调味。"

**BatchNorm** 的做法不同：不是按每个厨师自己标准化，而是按**同一道菜在不同顾客那儿的做法**标准化。比如红烧肉在顾客 A、B、C 那里，BatchNorm 会统计这三个例子里红烧肉特征的平均值和方差。问题是：如果只有 1 个顾客（batch_size=1），就没法统计。这就是 BatchNorm 在 LLM 中不好用的原因——长序列训练时 batch 很小。

**RMSNorm** 是 LayerNorm 的简化版：只做"缩放"不做"平移"——只除以 RMS（均方根），不减去均值。计算更快（少一次 reduce），效果差不多。

**Pre-norm vs Post-norm**：
- Pre-norm：先归一化，再进 Sublayer。就像做菜前先洗手——保证干净的材料进入加工工序。
- Post-norm：先进 Sublayer，再归一化。就像做完菜再尝味道——可能已经糊了。

原始的 Transformer（"Attention Is All You Need"）用的是 Post-norm，后来发现 Pre-norm 训练**稳定得多**——因为梯度不需要穿过未归一化的残差连接。

**SwiGLU** 的直觉：把简单的一刀切（ReLU：x>0 时 x，否则 0）换成平滑的"门控"——一个门（Swish）控制另一个通道（线性投影）的信息流，就像水龙头控制水流。

---

## 数学公式 + 工程意义

### 1. LayerNorm 的公式

对于输入 x ∈ R^d（一个 token 的 d 维表示）：

```
μ = (1/d) · Σ_{i=1}^{d} x_i                      // 均值
σ² = (1/d) · Σ_{i=1}^{d} (x_i - μ)²              // 方差

x_hat = (x - μ) / √(σ² + ε)                       // 标准化
y = γ · x_hat + β                                  // affine transform

其中 γ, β ∈ R^d 是可学习参数，ε 是数值稳定小量（如 1e-5）
```

### 2. RMSNorm 的公式

```
RMS(x) = √((1/d) · Σ_{i=1}^{d} x_i²)             // 只算均方根，不算均值

x_hat = x / (RMS(x) + ε)                          // 不做去均值
y = γ · x_hat                                     // 没有 β（shift）

```

**工程差异**：LayerNorm 前向需要 2 次 reduce（均值 + 方差），RMSNorm 只需 1 次 reduce（平方和）。在 8192 维的 LLM hidden dim 上，这一步差异虽然绝对值不大，但当有 80 layers × 2 Norm/layer = 160 个 Norm 时，累积效应可节省 2-5% 的 latency。

### 3. Pre-norm vs Post-norm 的梯度分析

**Pre-norm**（Llama 风格）：
```
y = x + Sublayer(Norm(x))
```

**Post-norm**（原始 Transformer）：
```
y = Norm(x + Sublayer(x))
```

Pre-norm 的核心优势：残差连接 x 直接连到输出，梯度 ∂y/∂x = I（无衰减）是恒等映射的，只有通过 Sublayer 的梯度受影响。而 Post-norm 中，归一化把残差输入也做了变换，∂y/∂x 被 Norm 的 Jacobian 修改——在深层网络中，Norm 的 Jacobian 可能放大或缩小梯度，导致不稳定。

**实验结果**：Post-norm 训练 65B 参数模型需要非常小心的 warmup（前几千步用小学习率），否则直接 loss spike。Pre-norm 不需要任何 warmup。

### 4. Activation 函数对比

| 激活函数 | 公式 | 特点 |
|----------|------|------|
| **ReLU** | max(0, x) | 简单，但 x<0 时梯度为 0（dead neuron） |
| **GeLU** | x · Φ(x)（Φ 是标准正态 CDF） | 平滑，有非零梯度在 x<0，GPT-3 使用 |
| **Swish/SiLU** | x · σ(x)（σ 是 sigmoid） | 平滑，自门控，B=1 时等价 |
| **SwiGLU** | SiLU(xW_1) ⊙ (xW_2) | 门控线性单元，最先进的激活方案 |

**SwiGLU 的详细公式**：

```
SwiGLU(x) = SiLU(xW_gate) ⊙ (xV · W_up)

其中：
- W_gate ∈ R^{d_model × d_intermediate}  // gate projection
- W_up ∈ R^{d_model × d_intermediate}    // value projection
- ⊙ 表示逐元素乘
```

**为什么 intermediate dim 设为 8/3 × d_model 而不是 4 × d_model？**

因为 SwiGLU 有**3 个矩阵**（gate, up, down）而非传统 FFN 的 2 个矩阵（up, down）。为了保持总参数量不变，工业界把 d_intermediate 设为 (2/3) × 4d = 8d/3：

| 配置 | FFN 参数（d_intermediate=4d） | SwiGLU 参数（d_intermediate=8d/3） |
|------|-----------------------------|-------------------------------------|
| 2 矩阵（up + down） | 2 × d × 4d = 8d² | — |
| 3 矩阵（gate + up + down） | — | 3 × d × (8d/3) = 8d² |

**效果**：PaLM 实验表明 SwiGLU 在相同计算量下比 ReLU 的 perplexity 低 0.2-0.5，这对 LLM 来说是很显著的提升。

### 5. QK Normalization

在 Attention 中，Q 和 K 各自做 LayerNorm 或 RMSNorm：

```
Q_norm = RMSNorm(Q)
K_norm = RMSNorm(K)
Attention(Q_norm, K_norm, V)
```

**为什么有用**：训练大模型时，Q 和 K 的数值范围可能差异很大，导致 attention logits 不稳定。QK Norm 确保二者的范数始终在 1 附近，使 attention 分布更"温和"。DeepSeek-V3 和 Llama 3 都使用了 QK Normalization。

---

## 工业界真实实现

### Llama 3 的 Norm 配置

Llama 3 全部使用 **Pre-norm + RMSNorm**：
- 每个 Attention 前有 1 个 RMSNorm（输入归一化）
- 每个 FFN 前有 1 个 RMSNorm
- 最后的输出前有 1 个 RMSNorm
- **没有** QK Norm（Llama 3 选择不用，依赖 RMSNorm 的输入归一化）

```
# Pseudocode for a Llama 3 Transformer Block
def transformer_block(x):
    # Attention sublayer (Pre-norm)
    residual = x
    x = rms_norm(x)                 # RMSNorm BEFORE attention
    x = attention(x)                # Multi-head GQA
    x = x + residual                # Residual connection

    # FFN sublayer (Pre-norm)
    residual = x
    x = rms_norm(x)                 # RMSNorm BEFORE FFN
    x = swiglu_ffn(x)               # SiLU(gate) * up, then down proj
    x = x + residual                # Residual connection
    return x
```

### DeepSeek-V3 的 Norm 配置

DeepSeek-V3 同样使用 RMSNorm，但额外加入了**QK-RMSNorm**和**DeepNorm**初始化策略：

1. **QK RMSNorm**：Attention 计算前分别对 Q 和 K 做 RMSNorm
2. **DeepNorm**：在初始化时，根据网络深度 α 缩放参数，确保每层的输出方差不随层数增长：
   ```
   α = (2N)^(1/4) / sqrt(2)   其中 N 是层数
   ```
   这种初始化使 60+ 层的 DeepSeek 也无需 lr warmup 就能稳定训练。

### SwiGLU 在各模型中的使用

| 模型 | 激活函数 | d_intermediate | 备注 |
|------|----------|----------------|------|
| GPT-3 (175B) | GeLU | 4d (49152) | 旧方案 |
| PaLM (540B) | SwiGLU | 8d/3 (≈ 2.67d) | 首次大规模使用 SwiGLU |
| Llama 2/3 | SwiGLU | 8d/3 | 标配 |
| Gemma | GeGLU | 8d/3 | GeLU 门控变体 |
| DeepSeek-V3 | SwiGLU | MoE（多个 expert）| 结合 MoE 的门控 |

---

## CUDA/GPU 视角

### RMSNorm 的 Fused Kernel 实现

RMSNorm 有三步计算：
1. **Reduce**：计算 x² 的和，得到 RMS
2. **Normalize**：每个元素除以 RMS
3. **Scale**：乘以 γ

在 GPU 上，这三步如果分三个 kernel 执行，需要从 HBM 读 x 3 次，写结果 3 次。**Fused RMSNorm kernel** 的做法：

```cuda
// Fused RMSNorm CUDA kernel (simplified)
__global__ void fused_rms_norm_kernel(
    float* out, const float* x, const float* gamma,
    int n, float eps
) {
    // Step 1: Warp-level reduction to compute sum(x^2)
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        sum_sq += x[i] * x[i];
    }
    sum_sq = warp_reduce_sum(sum_sq);       // Warp shuffle reduction
    // ... block-level reduction ...

    // Step 2 & 3: Normalize and scale in the same pass
    float rms = sqrtf(sum_sq / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out[i] = (x[i] / rms) * gamma[i];
    }
}
```

这种 fused 实现将 HBM 读写从 O(3N) 降到 O(N)，利用了 **warp shuffle** 做高效的 warp 内 reduction（比共享内存快）。

### SwiGLU 的 Fused 实现

SwiGLU 的计算是 `SiLU(xW_gate) ⊙ (xW_up)`，然后 down projection。在 GPU 上，SiLU 是一个 element-wise 算子和 matmul 分开执行的话需要额外的 kernel launch 和 HBM 读写。将 SiLU 和乘加融合为 **Fused SiLU Mul**：

```cuda
// Fused SiLU + element-wise multiplication
out[i] = x_gate[i] * sigmoid(x_gate[i]) * x_up[i];
//        ^^^^^^^^^^^^^^^^^^^^^^^^^^    ^^^^^^^
//           SiLU(x_gate)               element-wise multiply
```

**效果**：节省 1 次 kernel launch 延迟（~2-5 μs）和 1 次 HBM 读写（d × 2 bytes），在 decode 的 memory-bound 场景下尤其重要。

---

## 本讲与整个 LLM 系统的关系

Normalization 和 Activation 看似是"细节"，实则决定训练稳定性和模型质量：

- **训练稳定性**：Pre-norm + RMSNorm 是现代 LLM 不需要复杂 lr warmup 的根本原因。在 1000+ GPU 训练时，一次 crash 重启的代价是数万美元。
- **推理效率**：RMSNorm 的 fused kernel 在 decode 阶段（memory-bound）是关键优化——少一次 HBM 读写可能意味着 2-5% 的吞吐提升。
- **模型能力**：SwiGLU 的门控机制提供了比 GeLU/ReLU 更丰富的非线性表达，被视为从 GPT-3 → PaLM → Llama 时代 perplexity 持续下降的关键因素之一。
- **与 Attention 的关系**：QK Norm 直接影响 attention 分布的"温度"——不归一化时 softmax 容易过饱和，归一化后 attention 分布更分散，有利于长上下文建模。

---

## 面试问题

1. **LayerNorm 和 BatchNorm 的本质区别是什么？为什么 NLP/LLM 只用 LayerNorm？** 从归一化维度和 batch size 依赖性两方面回答。

2. **RMSNorm 相比 LayerNorm 省略了什么？为什么可以省略？** 分析去均值操作的必要性和训练稳定性的 trade-off。

3. **Pre-norm 为什么比 Post-norm 训练更稳定？** 从残差连接和梯度流的角度推导。

4. **SwiGLU 为什么比 ReLU 好？为什么 FFN 的 intermediate dim 从 4d 变成了 8d/3？** 分析参数量匹配和门控机制。

5. **QK Normalization 解决了什么问题？** 分析 attention logits 的数值不稳定性。

6. **RMSNorm 在 GPU 上如何实现 fused kernel？** 描述 warp shuffle 做 reduction 的过程。

7. **如果不用任何 Normalization，深度 Transformer 会发生什么？** 分析内部协变量偏移和梯度爆炸/消失。
