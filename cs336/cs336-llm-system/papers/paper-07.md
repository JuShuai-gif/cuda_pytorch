# Paper-07: ZeRO / DeepSpeed — 有限显存上训练大模型的终极武器

> Rajbhandari et al., 2019. "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." SC 2020.

---

## 1. 解决什么问题

训练大模型时，显存消耗来自四个部分：

1. **模型参数（Parameters）**：FP16 下每个参数 2 bytes
2. **梯度（Gradients）**：FP16 下每个参数 2 bytes
3. **优化器状态（Optimizer States）**：以 Adam 为例，每个参数需要 momentum (m) 和 variance (v)，FP32 下每个 4 bytes，即共 8 bytes
4. **中间激活（Activations）**：取决于 batch size、序列长度和模型结构

以一个 7.5B 参数的模型为例：
- Parameters: 7.5B × 2 = 15 GB
- Gradients: 15 GB
- Optimizer States: 7.5B × 8 = 60 GB
- Activations: ~15 GB（取决于 batch size）
- **总计约 105 GB——即使最强的单卡（A100-80GB）也放不下**

Megatron-LM 通过 3D parallelism（TP + PP + DP）解决了这个问题，但代价很高：
- 需要复杂的通信拓扑（NVLink + InfiniBand）
- 编码复杂（每个线性层都需要手动插入通信操作）
- 需要修改模型代码（not transparent to the model）

Microsoft DeepSpeed 团队问了一个更根本的问题：**既然训练时所有 GPU 都在同步做相同的计算，那为什么每个 GPU 都要存储完整的优化器状态和梯度？能不能共享？**

这就是 ZeRO（Zero Redundancy Optimizer）的核心思想：**消除数据并行训练中所有 GPU 上的状态冗余**。

---

## 2. 核心创新

ZeRO 分为三个阶段（stages），每个阶段消除一种冗余：

### ZeRO Stage 1: Optimizer State Partitioning (P_os)

在标准数据并行（DP）中，每个 GPU 都有完整的模型副本，执行相同的 forward 和 backward，但处理不同的 mini-batch 数据。所有 GPU 的梯度通过 all-reduce 同步，然后每个 GPU 独立更新优化器。

注意：**所有 GPU 上的优化器状态（m 和 v）完全相同**！因为梯度是同步的，参数也是同步的，所以每个 GPU 经过 Adam 更新后的参数和优化器状态完全一样。

ZeRO-1 的改进：**每个 GPU 只存储 1/N 的优化器状态**（N=DP 度数）。参数更新后，需要进行一次 **all-gather** 来让每个 GPU 都有完整的更新后参数。

显存节省公式（以 Adam 混合精度训练为例）：
- 标准 DP 每卡显存：2Φ + 2Φ + 12Φ + activations（parameters + gradients + optimizer states + activations）
- ZeRO-1 每卡显存：**2Φ + 2Φ + 12Φ/N + activations**（优化器状态分片）

当 N=8 时，节省了 10.5Φ per GPU（约 79% 的优化器状态开销）。

### ZeRO Stage 2: Gradient Partitioning (P_os + P_g)

ZeRO-2 在 Stage 1 基础上进一步分片梯度。标准 DP 在 backward 结束后进行梯度的 all-reduce，然后每个 GPU 有完整梯度。ZeRO-2 改为：

1. Backward 过程中，每个 GPU 计算各参数梯度的不同部分
2. 梯度直接 **reduce-scatter** 到各自的 owner GPU（不需要先完整再分散）
3. 每个 GPU 只保留属于自己的那部分梯度（1/N）
4. 优化器只更新这 1/N 参数

显存节省：
- ZeRO-2：**2Φ + 2Φ/N + 12Φ/N + activations**

当 N=8 时，比 ZeRO-1 再节省 1.75Φ。

### ZeRO Stage 3: Parameter Partitioning (P_os + P_g + P_p)

ZeRO-3 对参数本身也进行分片。在 ZeRO-2 中，forward 和 backward 期间每个 GPU 仍有完整参数。ZeRO-3 中，**参数在大部分时间也是分片的**，只有在以下时刻才通过通信重建：

- **Forward pass**：需要参数时，all-gather 当前层的参数，计算后立刻释放
- **Backward pass**：类似地 all-gather 所需参数
- 参数只在各自 owner GPU 上持久存储

显存节省：
- ZeRO-3：**2Φ/N + 2Φ/N + 12Φ/N + activations**

所有状态都线性可扩展了！N=8 时总状态降低到原来的 1/8，训练 1T 参数模型成为可能（如果 activations 也存在可接受的范围内）。

---

## 3. 为什么有效

ZeRO 有效的原因极为直观——它做了一件在计算机科学历史上反复被证明正确的事：**去重（deduplication）**。

在标准 DP 中，N 个 GPU 存储了 N 份完全相同的优化器状态——这纯粹是浪费。消除冗余不需要任何近似，也不需要改变优化算法。因为 Adam 更新是逐参数的（element-wise operation），将参数按维度切分到不同 GPU 上计算完全不影响结果。

关键在于：**为什么之前没人这么做？** 因为数据并行一直被认为是"每个 GPU 有完整模型的独立副本"。ZeRO 打破了这个思维定式——**数据并行只需要每个 GPU 能在需要时访问到完整参数即可，不需要持久持有**。这种"just-in-time"的数据获取正是 ZeRO-3 的精髓。

此外，ZeRO 还有一个经常被忽略的重要特点：**它对模型代码完全透明**。对比 Megatron-LM 需要手动在每个线性层插入 TP 通信代码，DeepSpeed ZeRO 只需在训练脚本中加几行配置代码：

```python
import deepspeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config_params="ds_config.json",
    optimizer=optimizer
)
```

这是因为 ZeRO 通过 hook PyTorch 的 `nn.Parameter` 和 `torch.optim.Optimizer` 来实现分片和通信，不需要修改模型定义。

---

## 4. GPU/硬件角度解释

ZeRO 的核心权衡是：**用通信换显存**。

**通信量分析**（以 ZeRO Stage 2 为例，N=8，Φ=10B）：

| 操作 | 通信量 | 频率 |
|------|--------|------|
| Forward all-gather (参数) | Φ/N = 1.25B bytes | 每层每次 forward（ZeRO-3 特有） |
| Backward reduce-scatter (梯度) | 2Φ = 20B bytes | 每步 backward 结束时 |
| Optimizer state 无通信 | 0 | - |

总通信量约为 DP 的 1.5 倍。关键在于：**这些通信可以与计算重叠（overlap）**。

**通信隐藏（Communication Hiding）**：

1. **Pre-fetching**：在当前层进行计算时，异步启动下一层的参数 all-gather。到下一层需要用时，参数已经就绪。

2. **Gradient accumulation overlap**：在 backward 计算当前层时，同时进行上一层的梯度 reduce-scatter。CUDA stream 使得计算和通信可以并行。

3. **CPU offloading（Infinity）**：DeepSpeed 后续引入的 ZeRO-Infinity 可以将优化器状态 offload 到 CPU 内存，通过 NVMe SSD 进一步扩展有效显存至数百 GB。关键技巧是使用高效的 CPU→GPU 数据搬运（prefetching + DMA）。

**为什么 ZeRO 的通信量不会成为瓶颈**：在典型的集群中，NVLink 带宽（900 GB/s）远超过模型训练的计算吞吐（即使是最新的 H100，FP16 算力 1000 TFLOPS，每个参数的算术密集度也只能利用约 300 GB/s）。所以通信可以被计算充分覆盖。

---

## 5. 工业意义

DeepSpeed ZeRO 的影响力几乎可以与 Transformer 架构本身媲美：

1. **真正 democratize 了大模型训练**：训练 Llama-7B 只需 8×A100（用 ZeRO-3），而不用 24+ 张卡。这让没有 DGX SuperPOD 的团队也能训练大模型。

2. **ZeRO-Infinity 突破了显存墙**：将训练能力从 GPU 显存扩展到 CPU 内存 + NVMe SSD，使得在单节点（8 卡）上训练 100B+ 参数模型成为可能。

3. **与 Megatron-LM 互补**：DeepSpeed ZeRO + Megatron-LM 的组合（如 Megatron-DeepSpeed）成为训练超大规模模型（如 GPT-3 175B）的事实标准。ZeRO 提供 DP 级的高效扩展，Megatron-LM 提供 TP + PP 的补充。

4. **催生了 FSDP（PyTorch 官方版本）**：PyTorch 1.11 引入的 Fully Sharded Data Parallel (FSDP) 本质上是对 ZeRO-3 的重新实现。FSDP 的 API 设计与 ZeRO 一脉相承。

5. **标准化了训练 infrastructure 的抽象**：DeepSpeed 的配置系统（JSON-based config）定义了模型并行 strategy、offload policy、precision policy、scheduler configuration 等的标准词汇，被后续框架广泛模仿。

---

## 6. 如何复现

关键实现细节：

1. **ZeRO Stage 配置（deepspeed config JSON）**：
   ```json
   {
     "zero_optimization": {
       "stage": 2,                    // 或 1, 3
       "offload_optimizer": {
         "device": "cpu",             // CPU offload
         "pin_memory": true
       },
       "allgather_partitions": true,  // ZeRO-3: all-gather 参数
       "allgather_bucket_size": 5e8,  // 500M 一次通信
       "reduce_scatter": true,
       "reduce_bucket_size": 5e8,
       "overlap_comm": true,          // 通信与计算 overlap
       "contiguous_gradients": true,
       "sub_group_size": 1e9
     }
   }
   ```

2. **Parameter partitioning 实现原理**（ZeRO-3 核心）：
   - 每个参数 `p` 有一个 `ds_id`（属于哪个 GPU）
   - 在非 owner GPU 上，`p.data` 是一个空的 placeholder（0 字节）
   - 访问参数时（forward/backward），`all-gather` 从所有 GPU 收集该参数的各个分片
   - 使用后立即释放非 owner 部分的显存

3. **Gradient accumulation 与 ZeRO 的交互**：当使用 gradient accumulation 时，每个 micro-batch 的 backward 计算完成后的梯度在 reduce-scatter 之前累积在各自 GPU 上。正确的做法是 accum 满后再做一次性 reduce-scatter（减少通信次数）。

4. **Mixed-precision 训练中的 optimizer states 精度**：Adam 的 optimizer states 默认保存在 FP32 中。ZeRO-1 只分片 FP32 states，master weights（FP32 的参数副本）不分片——这保证了参数更新的数值精度。

5. **通信 bucket 大小调优**：bucket 太小 → 通信频繁 → overhead 大。bucket 太大 → buffer 显存占用大 → 通信启动晚 → overlap 效果差。经验值：500M-1B bytes per bucket。

---

## 7. 面试要点

**必问题**：

1. **ZeRO 的三个 Stage 分别分片了哪些状态？显存节省了各自多少？**
   答：Stage 1 分片优化器状态（12Φ → 12Φ/N）；Stage 2 额外分片梯度（2Φ → 2Φ/N）；Stage 3 额外分片参数（2Φ → 2Φ/N）。总计从 16Φ 降到 16Φ/N。

2. **ZeRO-3 相比 Megatron-LM TP 有什么优势？**
   答：ZeRO-3 对模型代码完全透明（不需要修改任何 layer 定义），而 TP 需要手动切分每个线性层和插入通信。ZeRO-3 的通信量更小（只在需要时通信，TP 每层都 all-reduce）。

3. **ZeRO 如何实现通信和计算的重叠（overlap）？**
   答：使用独立的 CUDA stream 进行通信流，与计算流并行。在计算当前层时，pre-fetch 下一层的参数（all-gather）。在 backward 计算当前层梯度时，对上一层的梯度进行 reduce-scatter。

4. **为什么梯度分片使用 reduce-scatter 而不是 all-reduce？**
   答：all-reduce 后所有 GPU 都有完整梯度，每个 GPU 的 optimizer 都要存储完整梯度。reduce-scatter 的结果是每个 GPU 只有一部分梯度——这正是分片所需要的。而且 reduce-scatter 的通信量只有 all-reduce 的一半（N-1/N vs 2(N-1)/N）。

5. **ZeRO-Offload（ZeRO-Infinity）的工作原理？**
   答：将优化器状态和梯度 offload 到 CPU 内存（或进一步到 NVMe）。在 optimizer step 时，CPU 执行 Adam 更新（不需要 GPU），避免了 optimizer states 与 GPU 内存的竞争。高效的 CPU-GPU 数据传输（prefetching + pinned memory）是关键。

6. **ZeRO 的通信开销什么时候会成为瓶颈？**
   答：当 GPU 数量非常大（N>64）且模型较小（<1B）时，通信量相对于计算量占比增大，overlap 困难。此外，如果网络带宽不足（如以太网而不是 InfiniBand），ZeRO-3 的频繁 all-gather 会导致明显的通信瓶颈。
