# Lecture 20: 分布式训练 II — 混合并行、Ring Attention 与梯度压缩

## 1. 本讲核心问题

当模型继续增大（100B+ 参数）且序列长度极长（128K+ tokens），单一并行策略不再够用。本讲回答三个核心问题：

1. **如何组合多种并行策略？** 2D/3D 混合并行（数据 + 流水线 + 张量），PTD（Pipeline + Tensor + Data）架构
2. **长序列训练的通信瓶颈如何解决？** Ring Attention（序列并行），利用环状通信在注意力计算中分摊内存和通信
3. **如何压缩梯度以降低通信开销？** Deep Gradient Compression（DGC）、1-Bit SGD、梯度稀疏化/量化

## 2. 通俗解释

**混合并行的直觉**：不同并行策略各有优劣。数据并行的通信量最小但内存不节省，张量并行的通信量最大但无气泡，流水线并行有气泡但通信少。3D 混合并行就是**在三维空间里同时切**：维度 1（数据并行，切数据）、维度 2（流水线并行，切层）、维度 3（张量并行，切矩阵）。这就像同时把蛋糕横着切、竖着切、斜着切——每块都很小，但合起来还是完整的蛋糕。

**Ring Attention 的直觉**：传统注意力计算需要 $O(L^2)$ 内存（$L$ 为序列长度）。当 $L=128$K 时，注意力矩阵就有 $128K \times 128K = 16$B 个元素！Ring Attention 的思路是：把 $Q, K, V$ 按序列维度分给多张 GPU，每张 GPU 只算一部分注意力，然后用环状通信传递 $K$ 和 $V$。就像 8 个人围成一圈做一道大拼图，每人只需要当前的碎片，拼完就传给下一位。

**梯度压缩的直觉**：每轮训练中，很多梯度的值非常小（接近 0），传输它们浪费带宽。DGC 的思想是：**只传大的梯度，小的先攒着，攒大了再一起传**。Momentum Correction 确保被延迟的小梯度不会丢失方向。1-Bit SGD 更激进：每个梯度只传一个比特（+1 或 -1），接收端用累积的误差修正来恢复精度。这就像汇报工作：大事马上报，小事攒一个月一起报——效率高，但可能有时效性问题。

## 3. 关键公式

**混合并行的总 GPU 数分配**：
$$
N_{\text{total}} = N_{\text{DP}} \times N_{\text{PP}} \times N_{\text{TP}}
$$
其中 $N_{\text{DP}}$ 为数据并行组数，$N_{\text{PP}}$ 为流水线段数，$N_{\text{TP}}$ 为张量并行组大小

**Ring Attention 计算分解**：
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$
Ring Attention 将 $K, V$ 沿序列维度分块，在 GPU 间轮转：
$$
S_i^{(t)} = S_i^{(t-1)} + Q_i K_{(i+t) \bmod N}^T
$$
$$
O_i^{(t)} = O_i^{(t-1)} + \text{softmax}(S_i^{(t)}) V_{(i+t) \bmod N}
$$

每个 step $t$ 中，GPU $i$ 从邻居接收 $K, V$ 块，计算局部注意力，然后传给下一个邻居。

**Deep Gradient Compression（DGC）核心机制**：
$$
\mathbf{v}_{k,t} = \begin{cases}
\mathbf{g}_{k,t} & \text{if } |\mathbf{g}_{k,t}| > \text{threshold} \\
0 & \text{otherwise}
\end{cases}
$$
$$
\mathbf{u}_{k,t} = \mathbf{g}_{k,t} - \mathbf{v}_{k,t} \quad \text{(残差累积)}
$$
$$
\mathbf{g}_{k,t+1}^{\text{effective}} = \mathbf{g}_{k,t+1} + \mathbf{u}_{k,t} \quad \text{(动量修正)}
$$

其中 $\mathbf{v}_{k,t}$ 为实际传输的稀疏梯度，$\mathbf{u}_{k,t}$ 为未传输的残差，在下一轮叠加

**1-Bit SGD 量化**：
$$
\tilde{\mathbf{g}}_t = \|\mathbf{g}_t\|_1 \cdot \text{sign}(\mathbf{g}_t) \quad \text{(1-bit 量化)}
$$
$$
\mathbf{r}_t = \mathbf{r}_{t-1} + \mathbf{g}_t - \tilde{\mathbf{g}}_t \quad \text{(残差误差累积)}
$$
$$
\mathbf{g}_t^{\text{transmit}} = \tilde{\mathbf{g}}_t + \mathbf{r}_{t-1} \quad \text{(误差补偿)}
$$

通信压缩比：
$$
\text{Compression Ratio} = \frac{32 \text{ bits (FP32)}}{1 \text{ bit (sign)} + 32 \text{ bits (norm)}} \approx 32\times
$$

**梯度稀疏化的通信量**：
$$
T_{\text{sparse comm}} = \gamma \cdot |\mathbf{g}| \cdot 32 \text{ bits} \quad \text{where } \gamma \approx 0.001\text{-}0.01
$$
$\gamma$ 为稀疏率（top-$k$ 选取的梯度比例）

## 4. 公式背后的直觉

- **3D 并行的"正交性"**：数据并行、流水线并行、张量并行之所以能同时使用，是因为它们在不同的"维度"上切分。数据并行在批次维度切，流水线并行在深度维度切，张量并行在隐藏维度切。这三种切割互不影响（正交），就像 $x, y, z$ 三个坐标轴。最终通信模式是三者的叠加。

- **Ring Attention 的 $O(L^2/N)$ 内存**：传统注意力每张 GPU 存储完整的 $L \times L$ 注意力矩阵。Ring Attention 通过序列切分，每张 GPU 只需 $\frac{L}{N} \times L$ 的子矩阵。同时，通过环状通信传递 $K$ 和 $V$ 块，实现了 `softmax` 的分块计算（需要额外的 rescaling 来修正分母）。这本质上是用通信换内存。

- **梯度压缩为什么"免费"**：DGC 不损失精度的直观解释是：小的梯度累积起来最终也会被传输（通过残差 $\mathbf{u}$）。Momentum Correction 确保这些"延迟"的梯度不会改变优化方向。类比：你打车去机场，每次只传"往东 100 米"这种大位移，3 厘米的微调先忽略；但 3 厘米攒到 100 米时一起报，总路线不变。

- **1-Bit SGD 的有效性**：如果所有 GPU 的梯度符号（sign）一致（大家都同意该参数应该增大），那用 +1/-1 就够了。如果符号不一致（有分歧），残差累积机制会保留差异信息。实际上，在大 batch 训练中梯度符号的共识度很高（因为同一个大 batch 的不同 micro-batch 的梯度方向相似），所以 1-bit 量化效果超预期。

- **通信瓶颈的计算**：多节点训练中，梯度 AllReduce 的通信时间可能超过计算时间：
  $$
  T_{\text{comp}} = \frac{\text{FLOPs}}{\text{GPU FLOPS}}, \quad T_{\text{comm}} = \frac{K_{\text{grad}}}{B_{\text{network}}}
  $$
  当 $T_{\text{comm}} > T_{\text{comp}}$ 时，训练变成通信瓶颈。GPT-3 规模的训练中，跨节点的梯度通信可占总时间的 30-40%。

## 5. 工业界用途

| 技术 | 压缩率/效率提升 | 代表系统 | 典型部署 |
|------|----------------|---------|---------|
| 3D 混合并行（PTD） | 支持 1T+ 参数 | Megatron-LM, DeepSpeed | GPT-3 (175B), 10000+ A100 |
| Ring Attention | $O(L^2) \to O(L^2/N)$ 内存 | RingAttention, StripedAttention | 128K+ 序列训练 (LLaMA-3) |
| Deep Gradient Compression | 270-600× 压缩 | DGC (PowerSGD 改进) | 1Gbps 以太网多节点训练 |
| 1-Bit SGD | ~32× 压缩 | 1-Bit Adam, 0/1 Adam | 低带宽集群（如云端 Spot 实例） |
| PowerSGD | 秩 $r$ 低秩分解 | PowerSGD (PyTorch 集成) | 通用梯度压缩，$r=1$ 时效果最好 |

**具体实践**：
- **Megatron-Turing NLG 530B**（Microsoft/NVIDIA）：3D 并行，数据并行跨 280 个 DGX 节点，流水线并行 16 段，张量并行 8 路
- **LLaMA-3 405B**（Meta）：FSDP + TP + PP 混合，支持 128K 上下文窗口（Ring Attention 或类似技术）
- **BLOOM 176B**（BigScience）：ZeRO-3 + 流水线并行跨 48 个节点的 384 张 A100，使用 1Gbps 以太网（慢速互联）
- **Hugging Face Accelerate**：简化了混合并行的配置，支持 FSDP + 流水线 + 张量并行的声明式组合

## 6. PyTorch 实现思路

```python
# ====================== 3D 混合并行配置 ======================
# 典型配置：TP 在节点内（NVLink），PP 跨节点（InfiniBand），DP 跨节点组

def configure_3d_parallelism():
    """PTD 并行：Tensor (node) + Pipeline (across) + Data (groups)"""
    world_size = dist.get_world_size()
    tp_size = 8   # 张量并行：单机 8 GPU
    pp_size = 4   # 流水线并行：4 个节点
    dp_size = world_size // (tp_size * pp_size)  # 数据并行：剩余

    # 构建 3D 进程网格
    # dp_rank: 数据并行组内 ID
    # pp_rank: 流水线并行组内 ID
    # tp_rank: 张量并行组内 ID
    mesh = torch.distributed.init_device_mesh(
        "cuda",
        (dp_size, pp_size, tp_size),
        mesh_dim_names=("dp", "pp", "tp")
    )
    return mesh

# ====================== Ring Attention ======================
def ring_attention(q, k, v, world_size, rank):
    """Q, K, V shape: (batch, num_heads, seq_len // world_size, head_dim)"""
    seq_len_per_gpu = q.shape[2]
    head_dim = q.shape[3]
    scale = head_dim ** -0.5

    # 输出和 softmax 归一化因子
    out = torch.zeros_like(q)
    lse = torch.full(
        (q.shape[0], q.shape[1], q.shape[2]),
        -float('inf'), device=q.device
    )  # log-sum-exp

    # Ring: 轮转 world_size 次
    for step in range(world_size):
        # 当前 KV 块
        kv_src = (rank - step) % world_size  # 当前 step 使用哪个 GPU 的 KV

        # 如果当前 KV 不在本 GPU，需要接收
        if kv_src != rank:
            k_chunk = recv_from_neighbor()
            v_chunk = recv_from_neighbor()
        else:
            k_chunk, v_chunk = k, v

        # 计算局部注意力分数
        scores = torch.matmul(q, k_chunk.transpose(-2, -1)) * scale
        block_lse = scores.logsumexp(dim=-1)

        # 更新全局归一化（flash attention 风格的 online softmax）
        new_lse = torch.logaddexp(lse, block_lse)
        out = out * torch.exp(lse - new_lse).unsqueeze(-1)
        out += torch.matmul(torch.softmax(scores, dim=-1), v_chunk)

        # 重归一化
        lse = new_lse

        # 发送 KV 给下一个邻居（环形传递）
        if step < world_size - 1:
            send_to_neighbor(k, v)

    return out

# ====================== Deep Gradient Compression ======================
class DeepGradientCompression:
    def __init__(self, compress_ratio=0.001, momentum=0.9):
        """
        compress_ratio: 保留的梯度比例（top-k），0.001 = 只传输 0.1% 梯度
        """
        self.compress_ratio = compress_ratio
        self.momentum = momentum
        self.residual = {}   # 残差累积：未传输的小梯度
        self.momentum_buf = {}  # 动量修正缓冲区

    def compress(self, model, iter_num):
        """压缩梯度，返回稀疏梯度 + 索引"""
        sparse_tensors = []
        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            # 初始化残差
            if name not in self.residual:
                self.residual[name] = torch.zeros_like(param.grad)
                self.momentum_buf[name] = torch.zeros_like(param.grad)

            # 梯度 + 累积残差
            grad = param.grad.data + self.residual[name]

            # Top-k 稀疏化
            k = max(1, int(grad.numel() * self.compress_ratio))
            topk_values, topk_indices = torch.topk(
                grad.abs().flatten(), k
            )

            # 构建稀疏梯度
            sparse_grad = torch.zeros_like(grad.flatten())
            sparse_grad[topk_indices] = grad.flatten()[topk_indices]

            # 残差 = 原梯度 - 已传输的稀疏梯度
            self.residual[name] = grad - sparse_grad.reshape(grad.shape)

            # Momentum Correction: 对未传输参数的动量做补偿
            mask = (sparse_grad.reshape(grad.shape) != 0).float()
            self.momentum_buf[name] = (
                self.momentum * self.momentum_buf[name] * (1 - mask)
                + mask * grad  # 传输的部分直接更新动量
            )

            sparse_tensors.append((sparse_grad, topk_indices, grad.shape))

        return sparse_tensors

    def decompress_and_apply(self, sparse_tensors, model, optimizer):
        """接收端：从稀疏梯度恢复到 dense 并更新参数"""
        for (sparse_grad, indices, shape), (name, param) in \
                zip(sparse_tensors, model.named_parameters()):
            # 恢复 dense 梯度
            dense_grad = torch.zeros(shape.numel(), device=param.device)
            dense_grad[indices] = sparse_grad[indices]
            param.grad.data = dense_grad.reshape(shape)

        optimizer.step()
        optimizer.zero_grad()

# ====================== 1-Bit SGD 实现思路 ======================
class OneBitSGD:
    def __init__(self, world_size):
        self.residual = {}
        self.world_size = world_size

    def compress(self, param, name):
        """将梯度量化为 1-bit: [sign_vector, norm]"""
        grad = param.grad.data
        if name not in self.residual:
            self.residual[name] = torch.zeros_like(grad)

        # 误差补偿：累积上一轮的量化误差
        corrected_grad = grad + self.residual[name]

        # 1-bit 量化
        sign_vector = torch.sign(corrected_grad)  # +1 or -1
        norm = corrected_grad.norm(p=1) / corrected_grad.numel()

        # 更新残差
        self.residual[name] = corrected_grad - norm * sign_vector

        return sign_vector, norm

    def decompress(self, sign_vectors, norms, param_shape):
        """ALL_GATHER 收集所有 GPU 的符号和 norm，取平均"""
        # AllGather sign_vectors (1-bit per element per GPU)
        # AllGather norms (1 float per GPU)
        # 平均梯度 = mean(norm_i * sign_vector_i)
        all_signs = all_gather(sign_vectors)  # [world_size, B, *shape]
        all_norms = all_gather(norms)

        avg_grad = 0
        for i in range(self.world_size):
            avg_grad += all_norms[i] * all_signs[i]
        return avg_grad / self.world_size

# ====================== 通信性能分析 ======================
def analyze_communication(model, world_size, bandwidth_gbps):
    """分析梯度通信的开销"""
    total_params = sum(p.numel() for p in model.parameters())
    grad_size_gb = total_params * 4 / 1e9  # FP32 gradient in GB

    # AllReduce 时间 (Ring AllReduce)
    allreduce_time_ms = 2 * (world_size - 1) * grad_size_gb / bandwidth_gbps * 1000

    # 1-Bit SGD 时间
    compressed_size = total_params / 8 / 1e9  # 1 bit = 1/8 byte per param
    onebit_time_ms = compressed_size / bandwidth_gbps * 1000

    print(f"Gradient size: {grad_size_gb:.2f} GB")
    print(f"AllReduce time (Ring): {allreduce_time_ms:.1f} ms")
    print(f"1-Bit SGD time: {onebit_time_ms:.1f} ms")
    print(f"Speedup: {allreduce_time_ms / onebit_time_ms:.1f}x")
```

## 7. TinyML / Edge AI 部署意义

- **联邦学习通信压缩**：DGC 和 1-Bit SGD 的思想直接适用于联邦学习中的**上行带宽节省**。移动设备上传压缩梯度到服务器，压缩比可达 300-600×，将 100MB 梯度压缩到 ~300KB。
- **异构训练**：混合并行的思想启发了边缘-云协同训练——边缘设备做数据并行（推理/轻量微调），云端做更大规模的计算聚合。
- **梯度稀疏化在端侧的启示**：Top-k 梯度选择与模型剪枝的"重要性分数"思想相通——两者都在识别"哪些参数/梯度最重要"。这启发了端侧的**稀疏反向传播**（Sparse Back-Propagation）：只反向传播最重要的梯度，减少端侧训练的计算量。
- **Ring Attention 的边缘应用**：在多个边缘设备（如多台 Raspberry Pi 集群）上联合推理长序列模型时，Ring Attention 提供了通信高效的内存分摊方案。
- **低精度通信**：1-Bit SGD 证明了极端量化在通信中的可行性——这启发了端侧 INT4/INT2 推理中的"极端量化可以工作"的信心。

## 8. 常见误区

1. **"3D 并行就是三种并行的简单叠加"** — 不准确。PTD 的难点在于**并行策略的搜索**——对于给定的模型和硬件，如何选择最佳的 $(N_{DP}, N_{PP}, N_{TP})$ 组合。GPU 拓扑（NVLink 域、节点间带宽）决定了哪些层适合哪种并行。

2. **"梯度压缩会损失模型精度"** — DGC 和 1-Bit SGD 通过残差累积机制**不损失收敛精度**（达到相同的最终 loss），但可能增加训练的迭代轮数（wall-clock time 是否加快取决于压缩比 vs. 额外轮数的 trade-off）。

3. **"Ring Attention 就是环形通信"** — 核心创新不是环形通信模式，而是**分块 softmax 的在线计算**（online softmax with rescaling）。需要精确的 log-sum-exp追踪来保证数值上等同于完整 softmax。

4. **"压缩比越高越好"** — DGC 的 Top-k 选择需要额外的 AllGather 来同步索引（哪些梯度被选中），极端稀疏下索引通信可能超过梯度值通信。0.1% 的稀疏率在实际中往往是最优的。

5. **"跨节点通信总是瓶颈"** — 如果计算/通信能够充分重叠（overlap），通信延迟可以被隐藏。DDP 的异步梯度同步 + 梯度累积可以实现这一点。但对于同步 AllReduce，通信时间 ≥ 最低带宽链路的传输时间，不可完全隐藏。

6. **"梯度压缩只适用于数据并行"** — 也适用于 ZeRO-3/FSDP。FSDP 通信中包含参数 AllGather 和梯度 Reduce-Scatter，梯度压缩可降低 Reduce-Scatter 的通信量，但参数 AllGather 不能压缩（需要精确参数做前向计算）。

## 9. 面试问题

**Q1: 3D 混合并行中，TP、PP、DP 如何协调工作？**
A: 全局 GPU 网格划分为 $(DP, PP, TP)$ 三个维度。TP 组（如 8 GPU）共同处理同一层，通过 AllReduce 同步中间结果——通信最密集，限定在单机内。PP 组按层切分，通过 P2P 发送激活值和梯度——通信量中等，可跨节点。DP 组内每张卡有完整模型流水线，独立处理不同数据——只需梯度同步。三者正交叠加：前向时数据先经过 TP 组内的分布式计算 → PP 组内的流水线传递 → DP 组独立运行，反向时同理。

**Q2: Ring Attention 与标准 Flash Attention 的区别？**
A: Flash Attention 解决的是**单 GPU 内**的 $O(L^2)$ 内存问题（通过 tiling 和 recomputation）。Ring Attention 解决的是**多 GPU 间**的序列并行问题——将 $Q, K, V$ 的序列维度切分到不同 GPU，通过环状通信传递 $K, V$ 块。两者可以组合使用：Ring Attention 负责跨 GPU 分布，每张 GPU 内部用 Flash Attention 计算。

**Q3: 1-Bit SGD 如何保证收敛？**
A: 通过**误差反馈（Error Feedback）**机制。每次量化时将量化误差存入残差缓冲区，下一轮叠加到原始梯度上再量化。这保证了长期来看所有梯度信息最终都被传输——只是时间上有延迟。理论分析证明，在凸优化假设下，1-Bit SGD 的收敛率与普通 SGD 相同（$O(1/\sqrt{T})$）。

**Q4: 为什么 Attention 是长序列训练的瓶颈？**
A: 标准 Attention 的 FLOPs 为 $O(L^2 d)$，内存为 $O(L^2)$。当 $L=128$K、$d=128$、head=32 时，单个注意力矩阵 = $(32 \times 128K \times 128K \times 2) / 1e9 = 1.05$ TB（BF16），远超出任何 GPU 显存。即便用 Flash Attention，$O(L^2)$ 的 FLOPs 也是不可承受的。这就是为何需要序列并行（Ring Attention）和稀疏注意力（如 LongFormer, BigBird）。

**Q5: 在多节点训练中，如何决定用 DDP、FSDP 还是混合并行？**
A: 决策树：(1) 模型能否单卡装下？→ DDP (2) 不能单卡但能装下优化器状态？→ FSDP/ZeRO-3 (3) 单机内部 NVLink 带宽充足？→ 加入 TP (4) 层数很深（如 100+ 层）需要跨节点？→ 加入 PP。最终组合取决于硬件拓扑和模型架构的协同分析。

## 10. 本讲总结

当模型规模和序列长度同时增长，单一并行策略无法同时满足内存和通信约束。本讲介绍了三个进阶技术：

1. **3D 混合并行**（PTD 架构）：在数据、流水线、张量三个维度上同时切分，实现 100B+ 参数模型的训练。关键在于理解每种并行的通信/内存 trade-off，以及 GPU 拓扑对策略选择的影响。

2. **Ring Attention（序列并行）**：将注意力计算的 $O(L^2)$ 内存分摊到多张 GPU，通过环状通信传递 K/V 块。结合 Flash Attention 可实现 128K+ 上下文的高效训练。

3. **梯度压缩**：DGC（稀疏化 + 残差累积）和 1-Bit SGD（极端量化 + 误差反馈）可以在不损失收敛精度的前提下实现 30-600× 的通信压缩，使低带宽网络（1Gbps 以太网）也能支持分布式训练。

贯穿本讲的核心理念是**通信换内存/通信压缩**：当通信成为瓶颈时，压缩梯度是有效手段；当内存成为瓶颈时，增加通信（如 Ring Attention 的 KV 传递）是合理策略。分布式训练的本质就是在这两个约束之间寻找最优平衡。
