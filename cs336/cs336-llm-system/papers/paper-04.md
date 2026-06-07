# Paper-04: DeepSeek-V2/V3 — MLA 注意力与 DeepSeekMoE 的极致效率

> DeepSeek-AI, 2024. "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model."
> DeepSeek-AI, 2024. "DeepSeek-V3: Pushing the Limits of Open-Source LLMs."

---

## 1. 解决什么问题

大模型推理面临一个尖锐的矛盾：**模型能力和推理成本之间存在指数级的关系**。

具体来说，Standard Multi-Head Attention 在推理时，每生成一个 token，都需要存储和加载所有历史 token 的 Key 和 Value（KV cache）。KV cache 的大小为：

$$\text{KV cache size} = 2 \times \text{layers} \times \text{heads} \times \text{seq_len} \times d_{head} \times \text{precision}$$

对于一个 70B 参数、40 层、64 个 head、d_head=128 的模型，推理 128K tokens 时，仅 KV cache 就需要：2 × 40 × 64 × 131072 × 128 × 2 bytes ≈ 172 GB 的显存！

这意味着推理的瓶颈不在计算（compute），而在显存（memory）和显存带宽。更大的 KV cache → 更小的 batch size → 更低的 GPU 利用率 → 更高的每 token 成本。

DeepSeek-V2 要解决的问题是：**如何在不显著损失模型质量的前提下，将 KV cache 压缩一个数量级（~10x）？**

同时，MoE（Mixture of Experts）虽然可以通过稀疏激活降低推理计算量，但传统的负载均衡方法（auxiliary loss）会引入破坏梯度的"路由噪声"。DeepSeek 还需要解决：**如何实现不需要 auxiliary loss 的 MoE 负载均衡？**

---

## 2. 核心创新

### 2.1 Multi-head Latent Attention (MLA)

MLA 的核心思想是**低秩压缩 KV cache**。标准 MHA 中：

$$K = XW_K \in \mathbb{R}^{n \times h \cdot d_k}$$

MLA 将 K 和 V 通过一个共享的低维 latent 空间投影：

$$C_{KV} = XW_{DKV} \in \mathbb{R}^{n \times d_c}$$

其中 $d_c \ll h \cdot d_k$（例如 d_c=512，而 h·d_k=8192）。然后从这个低维 latent 向量恢复出 K 和 V：

$$K = C_{KV}W_{UK}, \quad V = C_{KV}W_{UV}$$

推理时，只需缓存 $C_{KV}$（一个 $d_c$ 维向量 per token），而不是 $h \cdot (d_k + d_v)$ 维。KV cache 的大小变为：

$$\text{KV cache size}_{MLA} = (d_c + d_c) \times \text{layers} \times \text{seq_len} \times \text{precision}$$

对应前面的例子，KV cache 从 172 GB 降到了约 15 GB——**压缩了 10 倍以上**。

Q 也被类似地压缩到 latent 空间：$C_Q = XW_{DQ}$，然后 $Q = C_Q W_{UQ}$。注意 Q 不需要缓存（每轮只算当前 token 的 Q），但 low-rank Q 减少了 attention 计算的 FLOPs。

更精妙的是，**RoPE 的解耦处理**：RoPE 的位置旋转在线性空间中不能保持低秩性质——也就是说，如果将低秩 K 做 RoPE 旋转，再左乘 $W_{UK}$，结果不等于先左乘再旋转。MLA 的解法是将 Q 和 K 拆分为两部分：一部分经过低秩投影（不旋转），另一部分不经过低秩投影（用于旋转），最后拼接。

### 2.2 DeepSeekMoE: Auxiliary-Loss-Free Load Balancing

传统 MoE 使用 auxiliary loss 来鼓励均匀路由：
$$\mathcal{L}_{aux} = \alpha \sum_{i=1}^{N} f_i \cdot P_i$$

其中 $f_i$ 是 expert i 的实际使用频率，$P_i$ 是平均路由概率。这个 loss 强制让 routing 更均匀，但它引入了梯度噪声——auxiliary loss 与主 loss 梯度可能方向不同。

DeepSeekMoE 的创新是使用 **expert-level bias** 来调节负载，而不是 loss：

1. 每个 expert 有一个可学习的 bias $b_i$，加到路由分数上
2. 如果某个 expert 负载过高，减小其 bias（减少路由到它的概率）
3. 如果负载过低，增大其 bias
4. Bias 的更新不通过梯度下降，而是一个启发式的在线调整机制

这完全消除了 auxiliary loss 对主训练梯度的干扰，同时保持了负载均衡。

DeepSeek-V3 进一步引入了 **shared experts**：除了 routed experts（被 router 选择的），还有几个 shared expert 始终激活。Shared experts 捕捉跨所有 token 的通用知识（如语言模型的基础语法），而 routed experts 处理专家级知识（如代码、数学、特定领域）。

### 2.3 Multi-Token Prediction (MTP)

训练时不仅预测下一个 token，而是预测未来的多个 token（2-4 个）。每个未来 token 的预测使用独立的 output head。这在训练时增加了计算量（约 10%），但显著提高了模型对长程依赖的建模能力，且推理时完全不需要（因为推理仍是逐 token 自回归）。

---

## 3. 为什么有效

MLA 有效的直觉：**attention 中的 K 和 V 矩阵天然是低秩的**。对于每个 head，K 和 V 的有效信息维度远小于其表观维度 d_k。这是因为：

1. **序列中有大量冗余信息**：相邻 token 的 K、V 向量高度相似（尤其是像逗号、句号、常见词）
2. **不同 head 之间有信息重叠**：多个 head 可能关注相似的 pattern
3. **位置信息与语义信息可以分离**：RoPE 捕获的旋转信息与语义向量无关

低秩压缩将这些冗余"蒸馏"到紧凑的 latent 表示中，去掉了信息论意义上的冗余。

DeepSeekMoE 为什么不需要 auxiliary loss：expert bias 的调节是对"症状"的直接反应，而不是通过梯度间接传信号。类比一下：auxiliary loss 像是在告诉模型"你应该均匀地使用专家"（可能导致模型为了降低 loss 而选择不合适的专家），而 bias adjustment 是在均衡器后面加了一个可调电阻——改变的是路由条件，不是学习信号。

MTP 为什么有效：next-token prediction 的损失函数是 myopic（短视的）——只关心下一个 token，不关心后面的 token。预测多个 token 迫使模型在训练时就对更远的未来建立更好的内部表示。

---

## 4. GPU/硬件角度解释

MLA 对推理成本的影响是巨大的——确切地说，它让推理从 **memory-bandwidth bound 变成了更均衡的状态**。

在 A100-80GB 上做自回归推理：每生成一个 token 的 compute FLOPs 约为 2×参数量的 1-2%（因为 MoE 稀疏激活），但 KV cache 的加载是瓶颈。标准 MHA 中，加载 128K tokens 的 KV cache 约需 172 GB（如果装得下的话），对应约 86ms 的显存读取时间（按 2TB/s 带宽）。这已经超过了 attention 计算本身的时间（~10ms）。

MLA 将 KV cache 加载量降低到约 15 GB，对应约 7.5ms——不再主导 latency。这意味着：
- **更大的有效 batch size**：同样的显存可以同时服务更多请求
- **更高的 GPU 计算利用率**：compute 不再等 memory
- **更好的吞吐（throughput）**：在在线推理场景中，continuous batching 的效率更高

对于 DeepSeekMoE，稀疏激活（每个 token 只激活约 5-8% 的参数）意味着训练和推理的 FLOPs 远低于同规模的 dense 模型。但这里的重点是**通信**：在分布式训练中，MoE 的 all-to-all 通信是瓶颈（tokens 需要被路由到不同的 experts，它们坐落在不同的 GPU 上）。DeepSeek 的负载均衡策略确保了 all-to-all 通信量在各个 GPU 上大致均衡，避免了某些 GPU 的通信"热点"。

---

## 5. 工业意义

1. **定义了 MoE 推理效率的新标杆**：DeepSeek-V2 的 API 价格是 GPT-4-Turbo 的 1/50，同时质量相当。这不是因为 DeepSeek 在烧钱补贴，而是因为 MLA + MoE 的架构确实把推理成本降了一个数量级。

2. **Auxiliary-loss-free 负载均衡成为 MoE 新范式**：之前所有的 MoE 模型（Mixtral、Qwen-MoE、GPT-4 MoE）都依赖 auxiliary loss。DeepSeek 证明了不需要它也能训练稳定的大型 MoE 模型——这对训练效率有直接影响（去掉了干扰梯度）。

3. **MLA 的压缩思路启发后续工作**：KV cache 压缩通过 low-rank 分解成为一种通用技术，后续的 GQA、MQA 也可以用 MLA 统一解释——MQA 是 d_c=1 的 MLA，GQA 是 d_c=g 的 MLA。

4. **Cost-performance 比开启了 LLM 普惠化**：DeepSeek-V3 的训练成本据称约 $5.6M（对比 GPT-4 据传的 $100M+），开源后人人可以私有化部署。这是"AI 民主化"的标志性事件。

---

## 6. 如何复现

关键实现细节：

1. **MLA 中的 latent dimension 选择**：d_c 是 MLA 中最重要的超参数。d_c 太小则信息损失严重，太大则失去了压缩的意义。DeepSeek-V2 中 d_c=512 是经过大量消融实验确定的，约为 d_k 的 1/16。

2. **RoPE 的解耦投影**：这是 MLA 实现中最容易出错的地方。伪代码逻辑：
   ```python
   # Query projection
   c_q = x @ W_DQ           # [n, d_c]
   q_nope = c_q @ W_UQ      # [n, h * d_q_no_rope]  不旋转部分
   q_pe = x @ W_QR          # [n, h * d_q_rope]      旋转部分
   q = concat(q_nope, q_pe) # [n, h * d_k]
   
   # Key projection (推理时只存 c_kv)
   c_kv = x @ W_DKV         # [n, d_c]
   k_nope = c_kv @ W_UK     # [n, h * d_k_no_rope]
   k_pe = x @ W_KR          # [n, h * d_k_rope]
   k = concat(k_nope, k_pe)
   ```

3. **MoE expert bias update rule**：每个 training step 后，统计每个 expert 处理的 token 数。设定目标负载 $L_{target}$，更新：$b_i \leftarrow b_i - \text{sign}(L_i - L_{target}) \cdot \Delta$。$\Delta$ 是一个小的常数（约 1e-3）。

4. **Shared experts 的数量**：DeepSeek-V3 使用 1 shared expert vs 256 routed experts（top-8）。Shared expert 的参数量约占总量的不到 1%，但有效提升了基础语言能力。

5. **MTP 的 loss 权重**：主 loss（next token）+ λ × 辅助 losses（future tokens）。λ 通常设为 0.3-0.5，过高会损害主任务，过低则不起作用。

---

## 7. 面试要点

**必问题**：

1. **MLA 相比 MHA 的 KV cache 节省比例是多少？**
   答：约 6-12x，取决于 d_c 的选取（典型 d_c=512 时约为 10x）。节省来自将 h·(d_k+d_v) 压缩为 2·d_c。

2. **为什么 RoPE 不能直接应用在 MLA 的低秩 K 上？**
   答：RoPE 的旋转是逐对元素的操作，破坏了低秩性质——R(K_lowrank) 不等于 lowrank(R(K))。因此需要将 K 拆分：nope 部分走低秩，pe 部分直接旋转。

3. **DeepSeekMoE 如何在不使用 auxiliary loss 的情况下实现负载均衡？**
   答：通过 per-expert 的可调 bias，根据实时负载情况在线调整。overloaded expert → 减小 bias，underloaded expert → 增大 bias。Bias 修改的是路由条件，不影响 loss 梯度。

4. **Shared experts 和 routed experts 的区别？为什么要分开？**
   答：Shared experts 始终激活，处理通用知识（语法、常识）。Routed experts 按路由选择，处理专业知识（代码、数学）。分开避免通用知识被路由"稀释"，也避免路由将通用样本全送进同一个 expert 造成热点。

5. **Multi-Token Prediction 为什么能提升模型能力？**
   答：迫使模型在训练时建立更长远的规划能力，而不是只优化一步。MTP 的辅助 loss 提供了额外的训练信号，类似于 intermediate supervision。推理时回归标准自回归，无额外开销。

6. **MLA 和 GQA/MQA 的关系是什么？**
   答：MQA（Multi-Query）是 d_c=1 的特殊情况（K 和 V 只有 1 个 head）。GQA 是 d_c=g 的特殊情况（K 和 V 有 g 个 head）。MLA 比这两者更通用——latent 维度 d_c 可以任意选择，不限于整数个 head。
