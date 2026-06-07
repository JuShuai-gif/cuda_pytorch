# Paper-05: Megatron-LM — 万亿参数模型的分布式训练框架

> Shoeybi et al., 2019. "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism."
> Narayanan et al., 2021. "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM."

---

## 1. 解决什么问题

2019 年，BERT-Large 有 340M 参数，GPT-2 有 1.5B 参数。当你想训练一个 8B 甚至 175B 的模型时，单个 GPU 的显存根本放不下——即使是最强的 V100-32GB。

一个 175B 参数的模型（GPT-3 大小）：
- 参数存储：175B × 2 bytes (FP16) = 350 GB
- 梯度存储：同样 350 GB
- 优化器状态（Adam）：每个参数需要 m 和 v 两个状态，共 700 GB
- **总需求：约 1.4 TB——需要 44 个 V100-32GB 才能勉强放下**（不包括激活值！）

这就是"model parallelism"（模型并行）的动机。但传统的模型并行方式（如 Mesh-TensorFlow、GPipe）是把模型按层切开放到不同 GPU 上，同一时刻只有一个 GPU 在工作——**GPU 利用率为 1/切分份数**，极度浪费。

Megatron-LM 要解决的核心问题是：**如何让模型并行不仅可行，而且高效——让 GPU 利用率接近数据并行的水平（> 50%）？**

---

## 2. 核心创新

Megatron-LM 引入了两种互补的模型并行策略，与数据并行组合成 **3D parallelism**：

### 2.1 Tensor Parallelism (TP)

核心思想：**将单个 Transformer 层的权重矩阵沿列或行切分到多个 GPU 上**。

具体以 Self-Attention 为例。每个 attention head 的计算是独立的：

$$\text{head}_i = \text{Attention}(X W_{Q_i}, X W_{K_i}, X W_{V_i})$$

如果模型有 64 个 head，可以将 $W_Q$ 沿列切成 8 份（每份对应 8 个 head），分到 8 个 GPU 上。每个 GPU 只计算自己那 8 个 head 的 attention，然后通过 **all-reduce** 将结果合并。

数学上：

$$Y = \text{Concat}(\text{head}_1, ..., \text{head}_h) W_O$$

等效为：

$$Y = \sum_{i=1}^{TP} \text{Attention}(X W_{Q_i}, X W_{K_i}, X W_{V_i}) W_{O_i}$$

其中 $W_{Q_i}$ 是 Q 权重矩阵的一个"列切片"，$W_{O_i}$ 是 O 权重的一个"行切片"。关键点在于：**分解后的每份计算是独立的**，不需要跨 GPU 通信（除了最后的 all-reduce）。

FFN 层也类似地切分：

$$Y = \text{GeLU}(X W_1) W_2$$

将 $W_1$ 沿列切成 TP 份，$W_2$ 沿行切成 TP 份。每份在前向传播中独立计算（除了输入 X 的 broadcast 和输出的 all-reduce）。

### 2.2 Pipeline Parallelism (PP)

将模型按层分成多个 stage，每个 stage 放在不同的 GPU 上。例如 64 层的 Transformer，分成 8 个 stage，每个 stage 有 8 层。

但这引入了"气泡"（bubble）问题：如果每个 stage 必须等上一个 stage 的计算结果，则 pipeline 中总有 GPU 在空闲等待。

Megatron 的解决方案是 **1F1B (1 Forward, 1 Backward) scheduling**：

- 传统方案：完成所有 micro-batch 的 forward → 再完成所有 backward。气泡巨大。
- 1F1B：warm-up 阶段逐步填充 pipeline（先发几个 micro-batch 的 forward），然后每个 stage 交替执行 1 个 forward 和 1 个 backward。稳态下 GPU 利用率极高。

Bubble 比例为：

$$\text{bubble ratio} = \frac{PP - 1}{M}$$

其中 M 是 micro-batches 数量。M 越大（micro-batch 越小），bubble 越小。但 micro-batch 太小会降低 GPU 利用率（单个 micro-batch 的计算量不足以淹没 GPU 的 kernel launch overhead）。

### 2.3 Data Parallelism (DP)

每个 GPU（组）有完整模型副本，不同副本处理不同的 mini-batch 数据。梯度通过 all-reduce 同步。

### 2.4 3D Parallelism 的组合

Megatron-LM 将三种策略组合：

- **TP**（Tensor Parallelism）：层内切分，通信频繁但量小，适合同一节点内的 NVLink 互联（高带宽低延迟）
- **PP**（Pipeline Parallelism）：层间切分，通信量小（只传激活值和梯度），适合跨节点通信
- **DP**（Data Parallelism）：模型副本，通信也在反向传播结束时（all-reduce 梯度），适合跨节点

典型配置（以训练 175B 模型为例）：
- TP = 8（同一 DGX 节点的 8 个 GPU）
- PP = 4（4 个 stage）
- DP = 64（64 个数据副本）
- 总 GPU 数 = 8 × 4 × 64 = 2048

---

## 3. 为什么有效

Tensor Parallelism 有效的核心数学洞察：**Transformer 层的矩阵乘法天然可分**。$Y = XW$ 可以在列维度切分 $W$（$Y$ 的列也相应切分，不同 GPU 计算不同列）。这种切分不需要任何近似——输出与单卡完全一致。

但这有一个前提：需要保证切分后的子矩阵乘法的输入 X 在各 GPU 上一致。这就解释了为什么在 Self-Attention 和 FFN 之间需要 **all-reduce**：Self-Attention 输出的每个 token 在不同的 GPU 上有不同的列片段，但 FFN 需要在全维度上计算，所以必须通过 all-reduce 合并。

Pipeline Parallelism 有效的直觉：**通信和计算可以 overlap**。1F1B 调度在稳态阶段让每个 GPU 都在忙——前面的 GPU 在执行 forward，后面的 GPU 在执行 backward（不同 micro-batch）。理想情况下，bubble ratio 趋近于 0。

---

## 4. GPU/硬件角度解释

这里的硬件考量极为重要，因为它们决定了三种并行策略的部署方式：

**通信拓扑 vs 并行策略**：

| 策略 | 通信模式 | 带宽需求 | 延迟敏感 | 部署层级 |
|------|----------|---------|----------|---------|
| TP | all-reduce (每层) | 极高 (~200 GB/s) | 是 | 节点内 (NVLink, 900 GB/s) |
| PP | point-to-point (每层边界) | 中 (~5-25 GB/s) | 否 | 节点间 (InfiniBand, 50-400 GB/s) |
| DP | all-reduce (每步) | 高 (~25-100 GB/s) | 否 | 节点间 |

**为什么 TP 只能在节点内**：每次 forward 中 TP 需要至少 4 次 all-reduce（attention 的 QKV + output + FFN 的 W1 + W2），总通信量约为 2×batch×seq_len×hidden_dim。对于 8B 参数模型，这是 ~2 GB 的数据。NVLink 在 900 GB/s 下只需 ~2ms，而 InfiniBand 200 Gbps 需要 ~80ms——差异是 40 倍。80ms 对于一次 layer 的 forward（通常 5-15ms）来说太长了。

**1F1B 调度与 GPU 利用率**：假设 PP=4, M=32（32 个 micro-batches）：
- Bubble = (4-1)/32 = 9.4%
- GPU 利用率 ≈ 90.6%

如果张量并行将 TP=8 节点内的 8 个 GPU 视为一个逻辑 GPU（"virtual pipeline stage"），则 TP 内部的通信被 NVLink 吸收，PP 跨节点的通信量显著减少（每次只传 batch×seq_len×hidden_dim 的边界数据，而不是完整的矩阵乘法结果）。

---

## 5. 工业意义

Megatron-LM 的直接和间接影响极为深远：

1. **使大规模模型训练可行**：GPT-3、Megatron-Turing NLG（530B）、Bloom（176B）都是用 Megatron-LM 训练的。没有 3D parallelism，这些模型根本无法训练。

2. **定义了分布式训练的术语和范式**：TP、PP、DP 的三维分解成为了整个工业界的标准语言。之后的 DeepSpeed ZeRO、FSDP、Alpa 等都是在这个框架上的扩展。

3. **催生了 sequence parallelism**：Megatron-LM v2 引入将 LayerNorm 和 Dropout 的激活值沿 sequence 维度切分，进一步减少了每张卡的激活存储，使激活存储不再成为 scaling 瓶颈。

4. **Selective activation recomputation**：不是重计算所有中间激活，而是只保留 attention 的 softmax output 和 layernorm 的 input，中间结果在 backward 时重算。相比于全重算节省了约 2/3 的显存，代价是约 30% 的额外计算。

5. **证明了"通信隐藏"的关键性**：PP 的 1F1B 调度、DP 中 gradient all-reduce 与 backward 的 overlap——这些技巧不是减少了通信量，而是让通信时间被计算覆盖了。

---

## 6. 如何复现

关键实现细节：

1. **TP 中的 column-parallel 和 row-parallel 线性层**：
   ```python
   # Column-parallel: W 沿列切分，每个 GPU 算一部分输出列
   # f(x) 和 g(x) 是对偶操作
   class ColumnParallelLinear:
       def forward(self, x):
           # x 在所有 GPU 上相同（由上一个 all-reduce 保证）
           y = x @ self.W  # self.W 只有 part 的列
           return y  # y 也是分片的
   
   # Row-parallel: W 沿行切分，需要 all-reduce 合并输出
   class RowParallelLinear:
       def forward(self, x):
           # x 在 GPU 上分片
           y = x @ self.W  # self.W 只有 part 的行
           return all_reduce(y)  # 合并
   ```

2. **Communication hooks for gradient synchronization**：在 PP 中，每个 micro-batch 的 backward 完成后需要发送 gradient 到上一个 stage。使用 `torch.autograd.Variable.register_hook` 来在梯度计算完成后立即触发通信，与后续 micro-batch 的计算重叠。

3. **Micro-batch size 选择**：需要权衡 bubble ratio 和 GPU 效率。M 太小 → bubble 大。M 太大 → micro-batch 太小 → GPU 上的矩阵乘太小 → GPU 利用率低。经验法则是 micro-batch size 至少为 1 token（纯数据并行）或至少为 GPU 能饱和的最小矩阵尺寸（通常 ~128 tokens）。

4. **Random seed alignment**：在 DP 中，每个副本需要用不同的随机种子（或动态调整 dropout mask），否则所有副本计算相同的梯度，DP 退化。

---

## 7. 面试要点

**必问题**：

1. **TP、PP、DP 的区别和应用场景？**
   答：TP 层内切权重矩阵，通信频繁量小，适合节点内 NVLink。PP 层间切模型，通信低频量中，适合跨节点。DP 复制模型副本，通信在梯度同步（每步一次），适合跨节点。

2. **Pipeline bubble 是什么？如何减小？**
   答：PP 中由于阶段间的数据依赖，前面的 GPU 必须等待，产生空闲期。Bubble ratio = (PP-1)/M。通过增大 M（micro-batch 数）和使用 1F1B 调度来减小。

3. **Tensor Parallelism 中为什么需要 row-parallel linear 后的 all-reduce？**
   答：Row-parallel 中每个 GPU 只算了一部分输出维度。attention 之后的 FFN 需要在全维度上计算，所以必须 all-reduce 合并各部分结果，使每个 GPU 上的输入再次一致。

4. **为什么 TP 不适合跨节点？**
   答：TP 每层需要多次 all-reduce，通信量约 2×batch×seq_len×hidden_dim，对于大模型每层约 2 GB。在 NVLink 上 ~2ms（可接受），在 InfiniBand 上 ~80ms（远超过单层计算时间 5-15ms）。

5. **组合并行（3D）时，通信量如何分布？**
   答：TP 通信 > DP 通信 > PP 通信。TP 每层都在通信，DP 每步结束后通信，PP 只在 stage 边界通信。所以 TP 放在节点内，DP/PP 放在节点间。

6. **怎么确定一个模型用什么样的 TP/PP/DP 组合？**
   答：首先看单张卡能放模型的比例（确定最小 TP×PP），然后尽量使用 DP 来填满集群。原则：节点内用 TP（NVLink 带宽充足），节点间优先 DP（通信只发生在梯度同步），剩余用 PP。
