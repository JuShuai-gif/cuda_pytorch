# Lecture 05: GPU 架构基础

## 本讲核心问题

1. GPU 凭什么比 CPU 快 100 倍做矩阵运算？架构差异的本质是什么？
2. 什么是 Warp？Thread Block？Grid？它们如何映射到硬件上？
3. HBM、L2 Cache、Shared Memory (SRAM)、Registers 各有什么区别？访问速度差多少？
4. 为什么说 memory coalescing 是写 CUDA kernel 的第一准则？
5. B200 的 2.25 PFLOP/s 和 8 TB/s HBM bandwidth 意味着什么？我们离"无限算力"还有多远？

## 通俗解释

### GPU ≈ 一万个小学生同时做简单的加减法

CPU 像一个数学教授——能做微积分、解方程，一个 step 解决复杂问题，但只有 32 个教授同时在干活（32 cores）。

GPU 像一万个小学生——每个人只会做简单的加减乘除，但 **一万个人同时做**。矩阵乘法刚好可以拆成"每个学生算一个位置的值"——每个人独立计算，不需要通信。

这就是 GPU 的力量来源：不是"单个任务做多快"，而是"同时做多少简单任务"（throughput, not latency）。

### SM (Streaming Multiprocessor) ≈ 一个教室

一个 H100 GPU 有 132 个 SM（每个 SM 是一个微型计算单元）。每个 SM 里面有：
- 4 个 Warp Scheduler（负责分配任务）
- 128 个 CUDA Core（小学生的计算台）
- 4 个 Tensor Core（矩阵运算加速器，专门做 matmul）

不同 GPU 的 SM 数量：

| GPU | 架构 | SM 数量 | CUDA Cores/SM | Tensor Cores/SM |
|-----|------|--------|---------------|-----------------|
| V100 | Volta | 80 | 64 | 8 |
| A100 | Ampere | 108 | 64 | 4 (3rd gen) |
| H100 | Hopper | 132 | 128 | 4 (4th gen) |
| B200 | Blackwell | 160 | 128 | 4 (5th gen) |

### Warp ≈ 一个班级的同步上课

Warp 是 GPU 的基本执行单元——32 个线程为一组，**同时执行同一条指令**。就像 32 个学生同时做同一道题（SIMT: Single Instruction, Multiple Threads）。

关键约束：如果一个 warp 里面其中一个线程走了 if 分支，其余 31 个线程也必须等它——这就是 **warp divergence**。所以 GPU 代码要尽量避免分支。

### Memory Hierarchy ≈ 离你越近的抽屉越小但越快

| 层级 | 大小 | 延迟 | 带宽 | 比喻 |
|------|------|------|------|------|
| HBM (显存) | 80 GB | ~800 cycles | 3.35 TB/s | 仓库（很大，但取货要走路） |
| L2 Cache | 50 MB | ~200 cycles | ~7 TB/s | 家门口的储物间 |
| Shared Memory (SRAM) | 256 KB/SM | ~30 cycles | 19.5 TB/s | 课桌抽屉（即拿即用） |
| Registers | 256 KB/SM | ~0 cycles | N/A | 手心里的纸（零延迟） |

从 HBM 读一个数的时间，可以从寄存器读 800 个数。这就是为什么"减少显存访问"是所有 GPU 优化的核心。

### Tensor Core ≈ 专业矩阵计算流水线

Tensor Core 可以在一个时钟周期内完成一次 **4x4 矩阵乘加**：

```
D = A * B + C
```

H100 的 Tensor Core 支持：
- FP64: 67 TFLOPS
- FP32: 134 TFLOPS
- FP16/BF16: 989 TFLOPS（with FP16/BF16 accumulate）
- FP8: 1979 TFLOPS
- INT8: 3958 TOPS

这就是为什么大模型转到 fp8 训练能获得接近 2x 的加速——Tensor Core 的 fp8 吞吐是 bf16 的两倍。

## 数学公式 + 工程意义

### Occupancy 计算

```
Occupancy = active_warps_per_SM / max_warps_per_SM
```

H100 每个 SM 最多有 64 个 warp（2048 线程）。但实际能同时跑多少个 warp 取决于：

1. **Register 限制**：每个线程用 255 个 register？那 `2048*255 = 522K registers > 256KB`，warp 数必须减少
2. **Shared Memory 限制**：每个 block 用 64KB shared memory？那最多 `228KB/64KB = 3` 个 block（但 SM 最多同时跑 32 个 block）
3. **Block 数量限制**：每个 SM 最多同时执行 32 个 thread block

Occupancy 不高意味着 SM 里的计算单元在"空转"等数据——这就是隐式的性能损失。

### Memory Coalescing

GPU 至少一次从 HBM 读取 32 bytes（一个 warp 的 32 线程每个读 4 bytes）。但如果这 32 个线程读的不是**连续地址**，就要多次读取：

```cuda
// Good: coalesced access (one transaction)
float val = data[threadIdx.x];  // thread 0 reads data[0], thread 1 reads data[1], ...

// Bad: strided access (32 transactions)
float val = data[threadIdx.x * stride];  // threads read data[0], data[N], data[2*N], ...

// Terrible: random access (32 transactions + cache misses)
float val = data[indices[threadIdx.x]];
```

Memory coalescing 的效果：连续访问 = 1 次 transaction = ~200 个 cycle。完全随机的访问 = 32 次，每一次都可能 cache miss 去 HBM 读 = 32 x 800 = 25600 cycles。

### Bank Conflicts

Shared memory 分为 32 个 bank，每个 bank 每个 cycle 可以服务一个地址。如果多个线程同时访问同一个 bank 的不同地址，就会产生 bank conflict——请求被串行化：

```cuda
// No bank conflict: each bank accessed once
__shared__ float cache[256];
float val = cache[threadIdx.x];  // contiguous access

// 2-way bank conflict: two threads access same bank
float val = cache[2 * threadIdx.x];  // stride 2

// 32-way bank conflict: all 32 threads access same bank
float val = cache[32 * threadIdx.x];  // stride 32
```

Bank conflict 最多可以让 shared memory 访问慢 32 倍。这是写 high-performance kernel 必须考虑的问题。

## 工业界真实实现

### NVIDIA GPU 演进：从 V100 到 B200

| GPU | 年份 | 制程 | MEM | BW | BF16 TFLOPS | SM | 关键创新 |
|-----|------|------|-----|-----|-------------|-----|----------|
| V100 | 2017 | 12nm | 16/32GB HBM2 | 900 GB/s | 125 | 80 | 第一代 Tensor Core |
| A100 | 2020 | 7nm | 40/80GB HBM2e | 2.0 TB/s | 312 | 108 | Sparsity 2x, MIG |
| H100 | 2022 | 4nm | 80GB HBM3 | 3.35 TB/s | 989 | 132 | Transformer Engine, FP8 |
| B200 | 2024 | 4nm | 192GB HBM3e | 8.0 TB/s | 2250 | 160 | FP4, NVLink 5, dual-die |

关键趋势：

1. **算力增速 > 带宽增速**：H100 BF16 (989 TFLOPS) / HBM BW (3.35 TB/s) = 295 FLOP/byte。越来越偏向 compute bound——未来更需要 arithmetic intensity 更高的 kernel
2. **低精度算力倍增**：每次精度降低 1 倍（bf16->fp8, fp8->fp4），算力翻倍
3. **显存增速相对滞后**：算力 1.8x/代，带宽 1.5x/代，显存 1x-2x/代
4. **MIG (Multi-Instance GPU)**：A100/H100 可以分割为独立的 GPU instance——这对推理部署（多个小模型共享一张卡）非常关键

### B200 规格解读

B200 的一颗芯片（die）中：
- 2.25 PFLOP/s (BF16 with sparsity: 4.5 PFLOP/s)
- 192 GB HBM3e (8 层堆叠)
- 8 TB/s HBM bandwidth
- 30 TB/s NVLink 5 bandwidth (18 个 NVLink 通道)
- 1.8 TB/s 跨芯片带宽 (NV-HBI)

这个配置意味着：

```python
# 8 x B200 node
total_memory = 8 * 192 = 1536 GB
total_compute = 8 * 2.25 = 18 PFLOP/s (BF16)

# Can train Llama 2-70B with full precision?
# model + grads (bf16): 140 GB
# optimizer (fp32): 560 GB
# With ZeRO-3: ~20 GB per GPU -- easily fits!

# 1M context training?
# KV cache: 80 layers * 2 * 1M tokens * 8192 dim * 2 bytes = 2.6 TB
# Even B200 can't hold this! Need RingAttention or quantization
```

### vLLM 的 GPU 视角

vLLM 的 PagedAttention 利用 GPU 的 virtual memory 管理 KV cache。传统做法给每个 request 预分配连续的 KV cache 块（像 malloc），但大部分块是空的（padding）。PagedAttention 像操作系统一样使用"页表"——KV cache 不连续存储，页表映射到物理地址。

从 GPU 角度，这避免了 HBM 的碎片化和浪费。vLLM 的测试显示相比 HuggingFace Transformers，吞吐提升 24x——不是因为 compute 更快，而是 **HBM 利用率** 更高。

## CUDA/GPU 视角

### NVIDIA GPU 架构：从 A100 到 H100 的关键变化

**H100 (Hopper) 的 TMA (Tensor Memory Accelerator)**：

TMA 是一个独立的硬件单元，专门负责异步地把数据从 HBM 搬进 shared memory。传统做法要写 load kernel（占用 CUDA core 计算资源），TMA 解放了 CUDA core，让数据搬运和计算完全 overlap。

```cuda
// H100 TMA: hardware-managed async copy
// CPU launches TMA descriptor, GPU continues computing
cp.async.bulk.shared::cluster.global  // One instruction, hardware handles alignment + coalescing
```

这对 LLM 训练的意义：flash attention 的 tiling 可以从 TMA 获得加速，数据预取不再消耗 CUDA core。

**B200 (Blackwell) 的 FP4 支持**：

```
FP4 (E2M1): 1 sign bit, 2 exponent bits, 1 mantissa bit
Range: [-6, 6], precision: ~0.5

BF16: 16 bits -> 2.25 PFLOP/s
FP4:  4 bits  -> 9.0 PFLOP/s (with sparsity: 18 PFLOP/s)
```

FP4 对推理的意义巨大——一个 405B 的密集模型在 FP4 下只需要 405B * 0.5 bytes = 202 GB，可以在 **单张 B200** 上放下。

### 实际 kernel 的 performance breakdown

以 GeLU activation kernel 为例（elementwise, memory-bound）：

```python
# Each element: read x, compute 0.5*x*(1+tanh(...)), write y
# Arithmetic Intensity = 5 FLOPs / 8 bytes (2 reads + 1 write) = 0.625 FLOP/byte

# H100 ridge point: 295 FLOP/byte
# 0.625 << 295 -> severely memory bound

# So optimization goal: FUSE with preceding operation
```

### LLM 的 compute bound vs memory bound 分析

对于 decoding（自回归推理，batch=1）：

```
Forward pass per token:
- matmul: 2 * d_model^2 * 2 (QKV + output proj) FLOPs  -->  compute bound (AI ~ d_model)
- attention: 2 * S * d_head FLOPs                        -->  memory bound (reading KV cache)
- MLP: 16 * d_model^2 FLOPs                               -->  compute bound

# Total AI for decoding is dominated by reading KV cache
# AI(decoding, S=4096) approx = (24*d_model^2) / (2*d_model*(1+S))
# With d_model=8192: AI = 24*8192^2 / (2*8192*4097) = 24 * 8192 / 8194 ≈ 24
# 24 << 295 (H100 ridge) -> severely memory bound
```

这就是为什么 **decoding 阶段 GPU 利用率只有 1-3%**——几乎所有时间都在等 HBM。

## 本讲与整个 LLM 系统的关系

```
Tokenizer -> Embedding -> Attention -> MLP -> Loss -> Optimizer -> Distributed -> Inference
               [_____________________________________________________________]
                                   所有阶段都运行在 GPU 上
                                   本讲 = 理解硬件基础
```

GPU 架构是整个 LLM 系统的**物理载体**。不理解 GPU 的 memory hierarchy、warp 调度、tensor core 特性，就不可能做真正的性能优化。LS 系统的瓶颈分析（memory bound vs compute bound）、kernel 融合策略、精度选择——所有这些决策的"根系"都在 GPU 架构中。

## 面试问题

**Q1: A100 和 H100 的关键架构差异是什么？为什么 H100 训练大模型能快 2-3x？**

A: (1) 算力：H100 BF16 = 989 TFLOPS vs A100 = 312 TFLOPS (3.2x)；(2) 带宽：H100 HBM = 3.35 TB/s vs A100 = 2.0 TB/s (1.7x)；(3) H100 的 TMA 硬件异步数据搬运，让计算和数据传输 overlap；(4) H100 Transformer Engine 有硬件 fp8 支持，算力再翻倍到 1979 TFLOPS；(5) SM 数量 132 vs 108。综合这些因素，在 fp8 训练场景下 H100 可以比 A100 快 3-6x。

**Q2: 解释 Bank Conflict 并从 CUDA/LLM 角度说明如何避免？**

A: Shared memory 分 32 个 bank，同一 cycle 内如果 2+ 个线程访问同一个 bank 的不同地址，产生冲突。避免方法：(1) padding——在 shared memory 数组每行末尾加 1 个元素，破坏对齐；(2) 确保访问步长和 32 互质；(3) 对于 matmul tiling，让 shared memory 的列数等于 warp 大小（32），每个线程读一个 bank。在实际 LLM kernel 中（如 FlashAttention），shared memory 加载 Q 和 K 的 tile 时使用 swizzle pattern 来避免 bank conflict。

**Q3: 分析 Llama 2-70B decoding 阶段的 GPU 利用率为什么只有 1-3%。**

A: Decoding 时 batch=1, seq_len 逐步增长。每个 step 的算术强度很低：(1) attention 需要读完整的 KV cache（内存密集，无重用），对 H100 来说每次读的 FLOP/byte 只有 ~24；(2) QKV 和 MLP 的矩阵乘法虽然是 compute-bound，但因为 batch=1，矩阵的维度是 [1, d_model] x [d_model, d_model]，activation 无法填充 GPU 的算力——本质上是一个"窄矩阵乘法"。H100 有 132 个 SM，每个有 128 个 CUDA core = 16896 个 core，但 batch=1 时很多 core 空闲。解决方法：增加 batch（batching）、speculative decoding（多个 draft token 并行验证）、continuous batching（vLLM 将多个请求合并成一个 batch）。

**Q4: 如果设计一个在 H100 上运行的推理引擎，你会如何安排数据布局来最大化 Memory Coalescing？**

A: KV cache 的数据布局至关重要：(1) KV cache 存储在 HBM 中，按 head 维度连续存储（而不是按 layer），确保同一个 head 的 K,V 在连续内存上；(2) 推理时，每个 warp 读一个 head 的一个 token 的 KV：`cache[head_idx * S * d_head + pos * d_head + lane_id]`——这保证 warp 内 32 个线程访问连续地址；(3) 使用fp8/fp4 存储 KV cache 减少传输量；(4) 用 prefetch 指令提前将下一层的 KV cache 从 HBM 搬到 L2。vLLM 和 TensorRT-LLM 都采用了类似的布局策略。
