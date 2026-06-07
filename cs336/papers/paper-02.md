# Paper-02: FlashAttention — 让 Attention 快 2-4x 的秘密

> Dao et al., 2022. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS 2022.
> Dao, 2023. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning."

---

## 1. 解决什么问题

Self-Attention 的 softmax 操作有一个隐藏但致命的问题：**它对显存带宽（memory bandwidth）的消耗比计算量（FLOPs）严重得多**。

具体来说，标准的 attention 计算过程是：

1. 从 HBM（High Bandwidth Memory，即 GPU 显存）加载 Q、K 到 SRAM
2. 计算 S = QK^T，将 S 写回 HBM
3. 从 HBM 读 S，计算 P = softmax(S)，将 P 写回 HBM
4. 从 HBM 读 P 和 V，计算 O = PV，将 O 写回 HBM

注意，中间结果 S 和 P（都是 n×n 矩阵）被反复从 HBM 写入和读出。每次 HBM 访问的代价约为 1.5× FLOPs 的时钟周期。在 A100 GPU 上，计算 312 TFLOPS 的理论算力 vs 2 TB/s 的显存带宽——这意味着每传输一个 FP16 数值（2 bytes），GPU 可以做大约 312 次浮点运算。

而标准 attention 的 **IO complexity**（数据搬运量）是 O(n²d + nd)，其中 n 是序列长度，d 是 head 维度。对于长序列（n=2048，d=64），attention 矩阵是 2048×2048=4M 个元素，约 8MB per head（FP16）。在实践中，由于 GPU 上的 SRAM（shared memory）只有约 192KB，远远不够放下整个 attention matrix，因此必须不断在 HBM 和 SRAM 之间搬运中间结果——这就是 **memory bound** 的本质。

FlashAttention 要解决的问题是：**如何在 IO-aware 的前提下，精确计算 attention，且不做任何近似**。

---

## 2. 核心创新

FlashAttention 的核心洞察可以概括为一句反直觉的话：**"我们不需要在 HBM 中存储完整的 attention 矩阵"**。

这是通过三个关键技术实现的：

### 2.1 Tiling（分块计算）

将 Q、K、V 分割成小块（tiles），每次只加载一个 tile 到 SRAM，在 SRAM 内完成该 tile 的局部 attention 计算，并逐步聚合结果。这样避免了将整个 n×n attention 矩阵写入 HBM。

对于 softmax 来说，tiling 有一个数学挑战：softmax 需要全局最大值和全局和来做归一化。FlashAttention 的解法是 **online softmax**（也叫做 **lazy softmax**）：

### 2.2 Online Softmax

标准 softmax 需要两次遍历：第一次找 max，第二次计算 exp 并求和。Online softmax 的数学技巧是将 softmax 重写为迭代形式：

设当前已处理的 block 的 softmax 结果为 $f_i$，新的 block 为 $g_i$，当前最大值和当前 scale 分别为 $m_{old}$ 和 $\ell_{old}$，新 block 的最大值为 $m_{new}$：

$$m = \max(m_{old}, m_{new})$$
$$\ell = \ell_{old} \cdot e^{m_{old} - m} + \ell_{new} \cdot e^{m_{new} - m}$$

这个公式的本质是：遇到新的更大值时，重新缩放已有的累积结果，使得增量计算与全局计算完全等价。

### 2.3 Recomputation

在反向传播时，不存储中间的 attention matrix（P = softmax(S)），而是在 backward pass 中重新计算。虽然这增加了 1 次额外的矩阵乘法，但节省了 O(n²) 的显存，使得长序列训练成为可能。

FlashAttention-2 进一步优化了 GPU 利用率：
- 将 Q 的遍历从 inner loop 移到 outer loop，减少了 shared memory 的使用
- 优化了 warp 调度，使得所有 warp 都参与矩阵乘法（FA v1 中有些 warp 在做 rescaling 时空闲）

---

## 3. 为什么有效

从直觉上理解，FlashAttention 有效的原因很简单：**它让 attention 从 memory bound 变成了 compute bound**。

标准 attention 的时间瓶颈不在计算，而在等数据从显存加载。每次计算 S = QK^T 时，Q 和 K 都在那里，但中间结果的搬运占用了绝大部分时间。FlashAttention 让所有的中间计算都在 SRAM 中完成，只把最终结果写回 HBM。

从算术密集度（arithmetic intensity）来看：
- 标准 attention 的 IO: ~ n²d bytes，FLOPs: ~ n²d，AI ≈ 1（非常低，memory bound）
- FlashAttention 的 IO: ~ n² bytes，FLOPs: ~ n²d，AI ≈ d（高了 d 倍）

当 d=64 时，arithmetic intensity 提升了 64 倍，直接跳到了 GPU 的 compute bound 区间，可以充分利用 GPU 的算力。

---

## 4. GPU/硬件角度解释

GPU 的存储层次结构是关键：

| 存储层级 | 大小 | 带宽 | 延迟 |
|----------|------|------|------|
| HBM (显存) | 40-80 GB | ~2 TB/s | ~300-500 cycles |
| L2 Cache | 40 MB | ~4 TB/s | ~200 cycles |
| L1 Cache/SMEM | 192 KB/SM | ~19 TB/s | ~30 cycles |
| Registers | 256 KB/SM | 即时 | 1 cycle |

标准 attention 的问题在于，中间的 attention matrix 必须放在 HBM 中（SRAM 放不下）。HBM 的带宽是 SRAM 的大约 1/10——**数据搬运速度跟不上计算速度**。

FlashAttention 完全改变了数据流：
1. 将 Q 分块，load 到 SRAM
2. 将 K、V 也分块，逐个 load 到 SRAM
3. 在 SRAM 中完成 Q_block × K_block 的乘法和 softmax
4. 对 V_block 做加权求和
5. 结果直接写回 HBM（跳过中间矩阵的存储）

这样做的代价是 Q（分块后的）被反复读取多次。但在 A100 上，80GB HBM 带宽约 2TB/s 的情况下，Q 和 K 各 ~n×d 大小（比 n×n 小得多），反复读的代价远小于存储 n×n 矩阵的代价。

在 FlashAttention-2 中，进一步的优化包括：**causal mask 可以直接在 softmax 之前应用**（将 mask 位置置为 -∞），这避免了额外的 mask 矩阵存储和加载。

---

## 5. 工业意义

FlashAttention 的影响超越论文发表时任何人的预期：

1. **使长上下文成为可能**：GPT-4 的 128k 上下文、Claude 的 200k 上下文，如果不用 FlashAttention，训练时的显存根本无法承受。n=128k 时，attention matrix 是 16B 个元素，即 32GB（FP16），单个 A100 就爆了。

2. **成为 PyTorch 2.0 的标准组件**：`torch.nn.functional.scaled_dot_product_attention` 默认使用 FlashAttention kernel。

3. **催生了 FlashAttention-2 和 FlashAttention-3**：FA-2 优化了并行策略，在 H100 上达到 70% 的理论算力利用率；FA-3 进一步利用 Hopper 架构的 TMA（Tensor Memory Accelerator）和异步数据搬运。

4. **跨领域应用**：FlashAttention 的 IO-aware 设计思想被扩展到其他操作——FlashFFTConvolution、FlashLinearAttention 等，形成了"FLASH"优化范式。

5. **关键开源贡献**：tri dao 的开源实现（cute + CUDA template）成为了 CUDA kernel 开发的教科书级参考。

---

## 6. 如何复现

核心实现要点：

1. **分块大小（tile size）选择**：需要让 Q 的 tile 和 KV 的 tile 同时能放进 SRAM。对于 A100 的 192KB SRAM（实际可用约 160KB），tile size 通常设为：Q: Br=128, K: Bc=128。

2. **Online softmax 的数学正确性**：关键是 scaling factor `e^{m_old - m_new}`。当新的 max 比旧的 max 大时（这是增量计算中的常见情况），旧的累积值需要缩小；反之亦然。实现了这个逻辑后，tiled softmax 与 non-tiled 输出数值完全一致。

3. **Backward pass 中的 recompute**：在 forward 时只存储 logsumexp（每行一个标量）和输出 O。backward 时重新计算 S = QK^T，然后用 forward 时存的 logsumexp 和 softmax 公式推导 dQ、dK、dV。注意需要在 backward 中也用 online softmax（同样用 logsumexp 缩放）。

4. **CUDA 实现要点**：
   - 使用 `__shared__` memory 存储 tile
   - 使用 warp-level 的 `__shfl_down_sync` 做 softmax reduction
   - 使用 `float4` 向量化加载以提高带宽利用率
   - 使用 `#pragma unroll` 展开循环

5. **Causal mask 处理**：在 tiling 中，对每个 (Q_tile, K_tile) 对，判断该 tile 是否需要 mask。如果 Q 位置 < K 位置，则该 tile 的上三角需要 mask。这可以通过判断 block 索引快速处理，不需要实际构造 mask 矩阵。

---

## 7. 面试要点

**必问题**：

1. **为什么标准 attention 慢？**
   答：不是计算慢，是显存带宽跟不上。中间 O(n²) 大小的 attention matrix 需要频繁在 HBM 和 SRAM 之间搬运，导致 memory bound。arithmetic intensity 只有 ~1，远低于 GPU 的机器平衡点。

2. **FlashAttention 的核心 trick 是什么？**
   答：Tiling 分块 + Online Softmax 增量归一化 + 反向传播中重计算。本质是把 attention 从 memory bound 变成 compute bound，arithmetic intensity 提升 d 倍。

3. **Online softmax 的公式是什么，如何保证正确性？**
   答：关键公式是 `ℓ_new = ℓ_old * e^{m_old-m} + (新block的exp和) * e^{m_new-m}`。通过 tracking global max 并在遇到新 max 时重新缩放，保证增量计算等价于全量计算。

4. **FlashAttention 的显存节省来自哪里？**
   答：forward pass 不存储 n×n attention matrix，只存储 O(n) 大小的 logsumexp 和 O。backward pass 通过 recompute 恢复 attention matrix。标准 attention 的显存是 O(n²)，FlashAttention 是 O(n)。

5. **FlashAttention-2 相比 v1 改进了什么？**
   答：将 Q 的遍历从 inner loop 移到 outer loop，减少 shared memory 压力；优化 warp 调度减少空闲；更好地利用 GPU 的并行度，在 H100 上达到了 70%+ 的理论算力利用率。

6. **什么时候 FlashAttention 加速不明显？**
   答：当序列长度很短时（n < 512），attention matrix 本身很小，数据搬运的开销不大，tiling 的 overhead 反而会拖慢速度。此外，当 head 维度很大时（d=128 或 256），FlashAttention 的优势也会减弱。
