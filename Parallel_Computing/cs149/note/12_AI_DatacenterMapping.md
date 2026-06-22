# CS149 第 12 讲：把 AI 应用映射到 AI 数据中心

**PDF**：Lecture 12 - Mapping AI Applications to the AI Datacenter

**课程**：Stanford CS149，2025 年秋季

---

## 本讲核心问题

1. AI 数据中心为什么首先是"内存与互连系统"问题，而不只是算力问题？
2. HBM、DRAM 银行、行缓冲、内存控制器调度如何影响 AI 性能？
3. 为什么分布式训练需要多种并行模式同时协作？
4. 如何理解集体通信与计算-通信重叠？

---

## 1. AI 数据中心的瓶颈并不只在算子本身

当模型规模和集群规模上来后，系统性能越来越依赖：

- 单卡本地内存带宽
- 片上 / 片外存储层次
- 设备间互连带宽与延迟
- 通信原语实现
- 能否把通信和计算重叠

所以"AI 数据中心映射"的本质是：在多级存储和多级互连上组织模型、数据与通信。

---

## 2. CPU vs GPU 内存总线宽度对比

| 维度 | CPU | GPU |
|---|---|---|
| 内存类型 | DRAM (DDR/DIMM) | HBM (High-Bandwidth Memory) |
| 总线宽度 | 64-bit | 1024-bit / stack |
| 设计取向 | 延迟优化 | 带宽优化 |

---

## 3. HBM：为带宽而生的内存系统

### 3.1 为什么 GPU / AI 加速器更偏爱 HBM

与传统 CPU 使用的 DDR / DIMM 相比，HBM 通过超宽总线、3D 堆叠和更短物理路径提供远高得多的总带宽。

### 3.2 HBM 3D 堆叠技术细节

- DRAM 通过 **Through-Silicon-Vias (TSVs)** 垂直连接
- TSVs 提供 logic layer 和 DRAM 之间的高并行连接
- 堆叠底层 "logic layer" 就是内存控制器
- Silicon "interposer" 作为高带宽互连

三大优势：更高带宽、更高能效、更小封装。

### 3.3 GPU HBM 采用历史

| GPU | 接口宽度 | HBM 规格 | 峰值带宽 | 容量 |
|---|---|---|---|---|
| AMD Radeon Fury (2015) | 4096-bit | 4×HBM1 | 512 GB/s | 4 GB |
| NVIDIA P100 (2016) | 4096-bit | 4×HBM2 | 720 GB/s | 16 GB |
| NVIDIA H100 (2022) | 6144-bit | 6×HBM3 | 3.2 TB/s | 80 GB |

---

## 4. DRAM 内部结构：不是一块平面大黑箱

### 4.1 DRAM 的基本物理组成

- 1 transistor + 1 capacitor per bit
- **Row buffer**：每行 2 Kbits
- **Data pins**：8 bits per chip
- 操作流程（以 DDR3-1600 为参考）：

```
Precharge (~10ns) → Row Activation (~10ns) → Column Selection (~10ns) → Data Transfer
```

### 4.2 两种访问延迟

- **最优情况（Row Hit）**：从已激活的 row 读取 — 仅需 CAS
- **最坏情况（Row Miss）**：bit lines 未就绪，需读新 row — PRE + RAS + CAS
- **关键问题**："何时执行 precharge？"Precharge 将 bit lines 就绪并**将 row buffer 内容写回 DRAM 数组**（读操作是破坏性的）

### 4.3 DRAM Burst Mode

摊销延迟到更大的传输上。每个 DRAM 命令描述批量传输，bits 在连续时钟周期内放置到输出引脚上。数据引脚如果只做单字节传输，利用率极低——这是最稀缺的资源！

### 4.4 多 Bank 流水线

- 所有 bank 共享相同的 data pins（一次只能传输一个）
- Banks 允许流水化：在一个 bank 传输数据时，另一个 bank 做 precharge/activate
- 目标：最大化数据引脚利用率

### 4.5 DIMM 组织与 Cache Line 读取

- 8 个 DRAM 芯片组成 64-bit DIMM
- **正确做法**：物理地址以字节粒度跨芯片交错分布 → 8 个芯片并行传输一个 cache line
- **错误做法**：连续物理地址映射到同一芯片的同一行 → 性能极差

---

## 5. 内存控制器调度

### 5.1 FR-FCFS 策略

控制器要在多个目标之间权衡：提高总吞吐、降低单请求延迟、保持公平性、降低能耗。

**FR-FCFS（First-Ready, First-Come-First-Serve）**：
1. 优先服务当前 open row 的请求（最大化行局部性）
2. 其他行请求按 FIFO 顺序处理
3. 可能将多个小请求合并为连续的大请求（burst mode）

控制器为每个 bank 维护独立的请求队列。

### 5.2 DDR4 具体参数

| 参数 | 数值 |
|---|---|
| DDR4 2400 | 64-bit × 1.2GHz × 2 (DDR) = 19.2 GB/s per channel |
| 2 channels | 38.4 GB/s |
| CAS 延迟 | ~13 ns |
| 参考处理器 | Intel Core i7-7700K |

---

## 6. 数据移动能耗再次成为主角

### 6.1 精确能耗数据

| 操作 | 能耗 |
|---|---|
| 32-bit FP math op | ~0.9 pJ |
| Local SRAM on-chip access | ~5 pJ |
| Load 32 bits from LPDDR | ~640 pJ |

来源：Han, ICLR 2016 (45nm CMOS)，Bill Dally (NVIDIA), Tom Olson (ARM)

### 6.2 推论

- **重新计算值往往比存储后重读更节能**
- 以 10 GB/s 从内存读取 ≈ 1.6W（移动 GPU 整个功率预算仅 ~1W）
- iPhone 16 电池 ≈ 14 瓦时，Macbook Pro 电池 ≈ 99 瓦时
- 模型越大，参数和激活搬运越可怕；训练时 optimizer state、gradient、activation checkpointing 都是内存系统问题；推理时 KV cache 是关键内存瓶颈

---

## 7. 集体通信原语是分布式 AI 的基础积木

### 7.1 常见原语

| 原语 | 描述 |
|---|---|
| **AllReduce** | 跨所有 rank 求和，结果广播到所有 rank |
| **ReduceScatter** | 求和后分散结果块到各 rank |
| **AllGather** | 从各 rank 收集块，聚合成完整张量 |
| **All-to-All** | rank i 将其数据的第 j 块发送给 rank j |

AllReduce = ReduceScatter + AllGather（ring 算法）

### 7.2 六种并行方式与通信原语映射

| 并行方式 | 切片维度 | 通信原语 |
|---|---|---|
| **Data Parallel (DP)** | Batch dim | Reduce-Scatter + All-Gather |
| **Tensor Parallel (TP)** | Hidden dim (weights) | Reduce-Scatter + All-Gather 或 All-Reduce |
| **Pipeline Parallel (PP)** | Layer dim | Send-Recv (P2P) |
| **Expert Parallel (EP)** | Expert (MoE) | All-to-All |
| **Sequence Parallel (SP)** | Sequence dim | All-Gather / Reduce-Scatter |
| **Context Parallel (CP)** | Context | All-Reduce 变体 |

---

## 8. 分布式矩阵乘法与局部-全局协作

### 8.1 一个典型模式

inputA [M×K] × inputB [K×N] = out [M×N]

将 K 维分布到 S 个 rank 上：
- 每 rank 计算 [M×K/S] × [K/S×N] = [M×N] 的部分结果
- Reduce-Scatter 合并 S 个部分结果

### 8.2 实例值

BS=16, M=24576, K=131072, N=8192。32 个 RDU 上做此分布。

---

## 9. 计算-通信重叠：系统能否扩展的关键

### 9.1 有无重叠的量化对比

| Sockets | 无重叠 | 有重叠 (RDU) |
|---|---|---|
| 8 | 理论峰值 88.5%, 实测 72% | — |
| 32 | 理论 52% | 持续 **70-79%** |

关键洞察：32 socket 持续 70+% 利用率归功于计算-通信重叠。AllReduce 完全与权重加载和计算重叠，不消耗 HBM 容量或带宽。

### 9.2 为什么单纯"更快网络"不够

如果通信总在计算后串行发生：设备规模一大，通信会吃掉越来越多时间，总 FLOPS 利用率快速下降。要实现重叠需要：算子切分粒度合适、通信可异步发起、缓冲区管理合理、编译器/runtime 了解依赖边界。

---

## 10. 流水并行中的气泡问题

### 10.1 基本机制

- Mini-batch 分割为多个 micro-batches
- Forward 和 Backward 计算跨 micro-batches 流水化
- **气泡** = 计算资源闲置（pipeline 的 ramp-up 和 drain 阶段）
- 更多 micro-batches → 更小气泡 → 但更多通信量

### 10.2 影响性能的因素

并行度、Global batch size、流水化调度策略、Microbatch size — 每一项都影响通信量、气泡大小、内存占用。

---

## 11. 大规模模型并行配置

从 1.7B 到 1T 参数规模的训练使用 TP + PP + DP 组合：

| 模型规模 | 峰值 FLOPS 利用率 |
|---|---|
| 1.7B | 44% |
| 530B | 49% |
| 1T | 49% |

**更大模型 → 更高利用率**（每 flop 的通信占比更低）

---

## 12. HBM4 演进新方向

未来 HBM4 在 logic die 层集成：
- SRAM cache
- KV cache compression
- I/O interfaces (Ethernet, PCIe)
- 存内计算（near-memory compute）

展示了存内计算的最新趋势。

---

## 13. 内存瓶颈的三条通用原则

1. **将数据存储靠近处理器**（Locate data storage near processor）
2. **将计算移向数据存储**（Move computation to data storage）
3. **数据压缩**（用额外计算换取更少的数据传输，trade-off extra computation for less data transfer）

---

## 常见误区

1. **误区：AI 数据中心问题就是"多几张卡"。**
   真正难点是存储层次、互连与通信组织。
2. **误区：HBM 只是更快的 DRAM。**
   它代表一种为吞吐而设计的系统取向。
3. **误区：通信优化只属于分布式系统，不属于并行计算课程。**
   大规模并行的核心就是计算与通信协同。
4. **误区：并行模式可以单独看。**
   现实大模型几乎总是多种并行方式混合使用。
5. **误区：只有 MPI 那种显式 `send/recv` 才算通信。**
   DRAM 内部的行切换、缓存一致性流量、跨 socket 内存访问同样是通信。

---

## 对应源码

| 文件 | 主题 | 重点 |
|---|---|---|
| `lecture12_part1.cpp` | 分布式 GEMM 与集体通信 | Reduce-Scatter / AllReduce、局部结果合并 |
| `lecture12_part2.cpp` | DRAM 与控制器 | bank、row buffer、burst、调度策略 |

---

## 学完本讲应做到

- 能解释 AI 数据中心为什么首先是内存与通信问题。
- 能理解 HBM、DRAM 行缓冲与内存控制器调度的性能含义。
- 能区分 DP、TP、PP、EP、SP、CP 等并行方式的通信特征。
- 能说明计算-通信重叠为何决定大规模扩展效率。
- 能用自己的话解释 DRAM 内部 precharge/activate/CAS 的完整操作流程。
