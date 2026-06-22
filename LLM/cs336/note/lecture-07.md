# Lecture 07: 分布式训练 Parallelism

## 本讲核心问题

当单个 GPU 的显存（如 NVIDIA H100 的 80GB HBM）无法装下一个 70B+ 参数的大语言模型时，如何通过多卡协作完成训练？本讲回答三个核心问题：(1) 如何把计算和存储分散到多个 GPU 上？(2) GPU 之间如何高效通信？(3) 多种并行策略（数据并行、模型并行、流水线并行）如何组合使用？

---

## 通俗解释

想象一个班级要完成一本超厚的习题集。**数据并行 (Data Parallelism)** 就是给每个学生复印一本完整的习题集，但每个人只做不同的题目——做完后对答案取平均。**模型并行 (Model Parallelism)** 则是一本习题集太厚了，一个人拿不动，必须拆成几部分：张三负责第1-10章，李四负责第11-20章——做一道题可能需要两人先后处理。**流水线并行** 则是工厂流水线思维：王五做完第一章传给赵六做第二章，赵六做完传给下一人，形成接力。

为什么单卡搞不定 70B 模型？以 BF16 精度计算：70B 参数 × 2 bytes = 140GB 的参数。训练时还需要 optimizer state（FP32 参数 + FP32 momentum + FP32 variance = 12 bytes/param），总共 70B × 12 = 840GB。加上梯度（2 bytes/param，140GB）和中间激活值，轻松突破 1TB。一块 H100 只有 80GB HBM，差了 10 倍以上。这就是为什么必须分布式训练。

---

## 数学公式 + 工程意义

### 1. Collective Operations 的数学定义

设有 P 个 GPU，每个 GPU i 持有长度为 N 的向量 x_i。

| 操作 | 数学定义 | 通信量 | 典型用途 |
|------|----------|--------|----------|
| **Broadcast** | 所有 GPU 收到 GPU_0 的 x_0 | O(N) | 分发模型参数 |
| **Scatter** | GPU_i 收到 x 的第 i 段 | O(N/P) | 分发输入数据 |
| **Gather** | GPU_0 收集所有 x_i | O(N) | 汇总结果 |
| **Reduce** | 所有 GPU 求和到 GPU_0 | O(N) | 汇总 loss |
| **All-Gather** | 每个 GPU 拥有全部 x_i 的拼接 | O(N) | 收集所有梯度 |
| **Reduce-Scatter** | GPU_i 拿到第 i 段的求和结果 | O(N) | ZeRO 中分发梯度 |
| **All-Reduce** | 每个 GPU 拿到所有 x_i 的和 | O(2N·(P-1)/P) | DDP 梯度同步 |
| **All-to-All** | GPU_i 发送 scatted 数据到 GPU_j | O(N) | MoE 路由 |

**关键工程洞察**：All-Reduce 的 Ring 算法通信量约为 2α(N·(P-1)/P)，而不随 P 线性增长。这意味着通信开销随节点数增加而**亚线性**增长，是 Data Parallelism 能 scale 的基础。

### 2. ZeRO 三阶段的显存公式

设模型参数量为 Ψ，优化器状态为 KΨ（Adam 时 K=12），GPU 数量为 N_d。

| Stage | 参数分布 | 梯度分布 | 优化器状态分布 | 每卡显存 |
|-------|----------|----------|----------------|----------|
| DDP | 复制 | 复制 | 复制 | (2+2+12)Ψ = 16Ψ |
| ZeRO-1 | 复制 | 复制 | **分片** | (2+2)Ψ + 12Ψ/N_d |
| ZeRO-2 | 复制 | **分片** | **分片** | 2Ψ + (2+12)Ψ/N_d |
| ZeRO-3 | **分片** | **分片** | **分片** | 16Ψ/N_d |

**工程意义**：ZeRO-3 使 16 卡即可训练 70B 模型（70B × 16 bytes ÷ 16 = 70GB < 80GB），而 DDP 需要 70B × 16 = 1.12TB 单卡——物理上不可能。

### 3. Tensor Parallelism（Megatron-LM 风格）

将 Transformer 的 MLP 和 Self-Attention 矩阵在**列维度**或**行维度**上切分。

对于 MLP 块 Y = GeLU(XA)B：

- **列并行**（split A 按列）：X 复制到各 GPU，各 GPU 计算 Y_i = GeLU(XA_i)，然后 Y = [Y_1, Y_2, ..., Y_N]
- **行并行**（split B 按行）：各 GPU 持有 Y_i 的一部分，计算 Z_i = Y_i B_i，最后 All-Reduce 求和

**通信开销**：每次 forward 需要 1 次 All-Reduce（MLP）+ 1 次 All-Reduce（Attention），共 2 次 All-Reduce，每次约 2bsh bytes（b=batch, s=seq_len, h=hidden_dim）。这意味着 Tensor Parallelism 受限于 **GPU 间带宽**——NVLink 900GB/s 下尚可，跨节点 PCIe 则成为严重瓶颈。

### 4. Pipeline Parallelism 与 Bubble

设模型有 L 层，分到 P 个 stage，每个 micro-batch 处理时间为 t_f/t_b。GPipe 方案中，pipeline bubble 占比为：

```
bubble_ratio = (P - 1) / M
```

其中 M 是 micro-batch 数量。**工程意义**：增大 M 可以减少 bubble，但需要更多显存放中间激活。1F1B（one-forward-one-backward）调度可以将激活峰值降低到 O(M)，而 GPipe 需要 O(M·P)。

---

## 工业界真实实现

### Llama 3 (Meta)：FSDP + Tensor Parallel 组合

Llama 3 405B 的训练使用 **FSDP**（基于 ZeRO-3）作为主要并行策略，跨 16,000 张 H100 GPU。FSDP 在通信开销和显存节省之间取得平衡：大部分时间参数是分片的（ZeRO-3 模式），只在 forward/backward 前通过 All-Gather 临时 reconstruct 完整参数。关键优化在于 **通信与计算重叠**：在计算当前层时，prefetch 下一层的参数。

在跨节点层面，Llama 3 使用 Tensor Parallelism 将 Attention 的 QKV 投影和 MLP 的矩阵按列切分到 8 个 GPU，通过 NVSwitch 实现节点内高速通信。

### DeepSeek-V3：Expert Parallelism

DeepSeek-V3 的 MoE 架构有 671B 总参数，但每个 token 只激活 37B 参数。MoE 的训练引入 **Expert Parallelism**：不同的 expert 分布在不同 GPU 上，通过 All-to-All 通信路由 token。关键创新：
- **辅助负载均衡 loss**：防止 token 过度集中到少数 expert
- **Device-level auxiliary loss**：确保 token 在各设备间均匀分布
- **Token-drop strategy**：当 expert 过载时丢弃多余 token（类似网络中的 congestion control）

### Megatron-LM (NVIDIA)：PTD-P 组合并行

Megatron-LM 提出 **PTD-P**（Pipeline + Tensor + Data Parallelism）三维混合并行：

1. **Tensor Parallelism (T)**：节点内，利用 NVLink/NVSwitch 高带宽（900 GB/s），切分单层
2. **Pipeline Parallelism (P)**：跨节点，通信量较小（只传边界激活），适合 InfiniBand
3. **Data Parallelism (D)**：最外层，梯度同步使用 All-Reduce

**典型配置**：训练 175B GPT-3 时，T=8（一个 DGX 节点内的 8 卡），P=8（8 个节点做流水线），D=64（64 份数据并行副本），总计 8×8×64=4096 GPUs。

### DeepSpeed ZeRO：三阶段递进

| 特性 | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|------|--------|--------|--------|
| 分片内容 | Optimizer states | + Gradients | + Parameters |
| 显存节省 | 4x | 8x | N_d x |
| 额外通信 | 无 | 无 | All-Gather（forward/backward）|
| 适用 GPU 数 | ≤ 64 | ≤ 128 | ≤ 1024 |

---

## CUDA/GPU 视角

### Memory Bottleneck 分析

GPU 的内存层次：
- **HBM（High Bandwidth Memory）**：H100 80GB，带宽 3.35 TB/s
- **SRAM（on-chip Shared Memory）**：H100 SM 228KB/SM × 132 SM ≈ 30MB，带宽 ~19 TB/s
- **操作 vs 通信的能耗比**：从 HBM 读取 1 个 FP32 数的能耗约是乘加运算的 200 倍

**通信瓶颈的本质**：All-Reduce 的 ring algorithm 需要每个 GPU 发送和接收各 N·(P-1)/P 个元素。在 InfiniBand NDR 400GB/s 连接下，传输 4GB 数据（相当于 1B 参数 FP32）需要约 10ms——比一层 forward 的 1-5ms 还长。这就是为什么通信需要和计算重叠。

### NVLink、NVSwitch、InfiniBand 对比

| 互联方式 | 带宽（双向） | 拓扑 | 范围 |
|----------|-------------|------|------|
| NVLink 4.0 | 900 GB/s (每个连接) | All-to-All（通过 NVSwitch）| 节点内（8 GPU） |
| NVSwitch | 3.6 TB/s（全双工聚合）| Full bisection | 单节点 |
| InfiniBand NDR400 | 400 GB/s（每端口） | Fat-tree / Dragonfly | 跨节点 |
| PCIe 5.0 x16 | 64 GB/s | 树形 | 单节点（非 NVLink） |

**关键数字**：GPU 内 HBM 带宽 3.35 TB/s，而 NVLink 900 GB/s 是其 1/4，InfiniBand 400 GB/s 是其 1/8。这就是为什么 Tensor Parallelism 的高频通信必须限定在 NVLink 域内（同节点），而 Data Parallelism 可以跨 InfiniBand。

### Fused Operator 在分布式训练中的应用

**Fused Adam**：将 Adam 的参数更新、梯度缩放、weight decay 融合为一个 CUDA kernel，减少 3 次 HBM 读/写。在分布式训练中，这一步在每个 GPU 独立完成（optimizer states 已分片），不需要通信。

**通信融合（Gradient Bucketing）**：PyTorch DDP 不会对每个 gradient tensor 单独做 All-Reduce，而是将多个梯度合并到一个 bucket（默认 25MB），一次通信完成多个算子的参数同步——减少通信启动次数（latency hiding）。

---

## 本讲与整个 LLM 系统的关系

分布式训练是 LLM Scaling Law 的物理实现。没有分布式训练，Scaling Law 只是理论曲线——70B 模型根本无法训练。本讲的技术直接决定了：
- **模型规模上限**：ZeRO-3 + Tensor Parallelism + Pipeline Parallelism 理论上支持万亿参数
- **训练效率**：通信/计算比决定了 GPU 利用率（MFU，Model FLOPs Utilization），Llama 3 405B 训练达到 38% MFU
- **硬件选型**：NVLink 域大小决定 Tensor Parallelism 的上限（通常 ≤ 8），InfiniBand 拓扑决定 Data Parallelism 的 scale-out 能力
- **与推理的关系**：Tensor Parallelism 同样用于推理部署（vLLM 支持），因为 70B 模型推理也需要多卡

---

## 面试问题

1. **DDP 和 FSDP 的核心区别是什么？** 从显存分布、通信模式、参数生命周期三方面回答。

2. **All-Reduce 的 Ring 算法通信量是多少？** 推导 2(N·(P-1)/P) 公式，分析当 P 很大时的极限。

3. **为什么 Tensor Parallelism 不能跨节点大规模使用？** 分析 NVLink vs InfiniBand 的带宽差距，以及通信频率对 TP 的影响。

4. **Pipeline Parallelism 的 bubble 如何计算？** 推导 bubble ratio = (P-1)/M，解释 1F1B 调度如何优化。

5. **ZeRO-3 在 forward 时发生了什么？** 描述 All-Gather 重建参数、计算、丢弃三步过程。

6. **MoE 的 Expert Parallelism 引入什么新的通信模式？** All-to-All 的性质、通信量、对带宽的要求。

7. **FSDP 如何实现通信与计算重叠？** 描述 prefetch 机制和 backward 时的梯度 Reduce-Scatter overlap。
