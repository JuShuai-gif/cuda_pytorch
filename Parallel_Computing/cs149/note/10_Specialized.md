# CS149 第 10 讲：硬件专用化

**PDF**：Lecture 10 - Hardware Specialization

---

## 本讲核心问题

1. 为什么通用 CPU 在能效上很难和专用加速器竞争？
2. GPU、DSP、FPGA、ASIC 分别牺牲了什么，又换来了什么？
3. 为什么 AI 推动硬件越来越“为矩阵乘法而生”？
4. 什么是 systolic array、dataflow 架构与 hardware lottery？

---

## 1. 为什么要专用化：能效是第一推动力

### 1.0.1 能耗公式

$$ Power = (Ops/second) \times (Joules/Op) $$

核心思想：Better energy efficiency ⇒ Specialization（固定功能更省能）

### 1.0.2 H.264 视频编码能耗分解案例

| 阶段 | 能耗占比 |
|---|---|
| 整像素运动估计 | 较大 |
| 亚像素运动估计 | 较大 |
| 帧内预测/DCT/量化 | 中等 |
| 算术编码 | 较小 |
| 功能单元（FU）计算 | **占比很小** |

关键发现：即使在 SIMD 优化实现后，大部分能量消耗在控制流上，而非实际的计算单元。来源：Hameed et al. ISCA 2010。

### 1.0.3 FFT 专用化的面积与功耗数据

- ASIC 以 ~1/1000 的芯片面积达到与一个 CPU 核相同性能
- GPU 核：面积效率比 CPU 核高约 5-7 倍
- ASIC 以仅 ~1/100 的功耗达到与一个 CPU 核相同性能
来源：Chung et al. MICRO 2010

### 1.0.4 现代处理器执行一条指令的 13 步开销

读指令 → 译码 → uop cache 查找 → icache 查找 → 地址翻译 → 检查依赖/流水线 hazard → 识别可用执行资源 → 控制寄存器文件 SRAM → 移动数据至执行单元 → 执行运算 → 写回 → ...

SIMD 执行的价值：将这些开销摊销到多条数据操作上。

### 1.1 通用处理器的隐性成本

CPU 能执行几乎任意程序，是因为它要付出大量控制成本：

- 取指与译码
- 分支与乱序控制
- 通用寄存器与通用数据通路
- 面向复杂控制流的灵活性设计

这些都很有价值，但对于某些高重复、规则的计算而言，它们也是巨大的能耗负担。

### 1.2 为什么专用化能更省电

专用硬件通过缩小“问题空间”，换来：

- 更短的数据路径
- 更少的控制逻辑
- 更高的数据复用
- 更高比例的芯片面积用于真正的算术单元

### 1.3 本讲的核心认识

专用化不是单纯“做得更快”，而是：

- 在相同功耗预算下做更多事
- 或在相同吞吐下消耗更少能量

---

## 2. 可编程性与能效的连续谱

可以把常见计算平台看成一条连续谱：

```text
CPU -> GPU -> DSP -> 领域专用加速器 -> FPGA -> ASIC
```

### 2.0.1 各平台能效具体倍数

| 平台 | 相对 CPU 的能效 |
|---|---|
| DSP (Qualcomm Hexagon) | ~20× |
| GPU | <10× |
| Domain-Specific Accelerator (如 Google TPU) | ~20× |
| FPGA | ~50× (争议中) |
| ASIC | 100-1000×（假定 compute-bound 且非浮点运算） |

### 2.0.2 Qualcomm Hexagon DSP 具体参数

- VLIW（Very Long Instruction Word）架构
- "单条指令指定多个不同操作同时执行"（与 SIMD 不同）
- 每周期执行 29 个 RISC 操作
- 用于 modem、audio、日益增加的 image processing
- 搭载于 Qualcomm Snapdragon SoC 和 Google Pixel 手机

### 2.0.3 Anton 超级计算机（分子动力学专用 ASIC）

DE Shaw Research 开发的蛋白质分子动力学超算：
- Anton 1 (2008)：512 个专用 ASIC 计算粒子间相互作用
- 定制低延迟通信网络优化 N 体模拟
- Anton 3 (2025)：比同时代 GPU 快约 20 倍

### 2.0.4 FPGA 详细结构

- "介于 ASIC 和处理器之间的中间地带"
- 可编程 LUT（Xilinx Virtex-7：6 输入 1 输出 LUT，相当于 64 元素表）
- 40 输入 AND = 串联 8 个 LUT6（delay = 3）
- 现代 FPGA 含：DSP blocks（乘法器）+ CPUs（ARM, RISC-V）
- 编程语言：Verilog/硬件描述语言
- Amazon EC2 F1/F2 提供云端 FPGA

ASIC 设计成本：10-1 亿美元（设计/验证/制造）

### 2.1 越往左

- 可编程性强
- 通用性高
- 开发门槛低
- 单位能效相对较差

### 2.2 越往右

- 对目标任务更极致优化
- 控制开销更低
- 可获得更高 perf/watt
- 但灵活性下降、开发成本上升

### 2.3 这不是非黑即白

现实系统常常是混合的：

- CPU 负责控制
- GPU 负责大规模张量计算
- NPU / TPU 负责矩阵乘法主干
- 固定功能单元负责编解码、显示、I/O 加速

---

## 3. 为什么 GPU 仍然“不够专”

### 3.1 GPU 已经比 CPU 更偏吞吐

它大量晶体管用于：

- SIMD / SIMT 算术单元
- Tensor Core
- 高带宽显存接口

### 3.2 但 GPU 仍然是通用处理器

它仍然有：

- 指令流控制
- 线程调度
- 通用地址生成
- 通用寄存器与通用访存层次

### 3.3 对 AI 而言的局限

对于高度结构化的大矩阵运算，GPU 仍存在：

- 指令与控制开销
- 通用存储与通用调度路径带来的额外能耗

于是推动了更激进的专用化方向。

---

## 4. Systolic Array：用数据流替代指令流驱动

### 4.1 基本思想

systolic array 的核心是：

- 让数据在处理单元阵列中有节奏地流动
- 每个处理单元执行固定、简单、重复的局部计算
- 部分和和输入在邻近单元间传递

### 4.2 为什么它很适合矩阵乘法

矩阵乘法本身具有：

- 规则的二维并行结构
- 高度重复的乘加模式
- 清晰的局部数据复用机会

### 4.3 与 SIMD 的关键差异

| 维度 | SIMD | Systolic Array |
|---|---|---|
| 驱动方式 | 指令驱动 | 数据流驱动 |
| 数据流 | 常需经寄存器 / 存储层次 | 更多是局部传递 |
| 控制结构 | 相对集中 | 更分布式 |
| 能效 | 中等到高 | 更高 |

### 4.4 为什么它高效

- 降低全局数据搬运
- 降低取指译码开销
- 利用邻近 PE 间局部通信
- 高比例面积用于乘加

> 对应源码：`lecture10_part1.cpp`
> 内容：权重驻留型 systolic array 的矩阵乘法模拟。

---

## 5. TPU：为矩阵乘法高度定制的加速器

### 5.0.1 NVIDIA A100 Ampere SM 具体规格

| 参数 | 数值 |
|---|---|
| 每 SM fp32 ALUs (mul-add) | 64 |
| 每 SM int32 ALUs | 32 |
| 每 SM Tensor Cores | 4 |
| Tensor Core 操作 | 8×4 × 4×8 MMA (fp16 输入, fp32 累加) |
| GA100 总 SM | 108 |
| 总 fp32 mul-add ALUs | 6,912 |
| 总 Tensor Cores | 432 |
| 峰值 fp32 | 19.5 TFLOPS (@1.4 GHz) |
| Tensor Core 峰值 | 312 TFLOPS (fp16/32 mixed) |

### 5.0.2 NVIDIA H100 基本规格

- 第四代 Tensor Core
- Tensor Memory Accelerator (TMA) 单元
- CUDA Cluster Capability
- HBM3: 最高 80 GB
- TSMC 4nm 工艺
- 800 亿晶体管
- 每 SM：256 KB shared memory / L1，64 KB registers per sub-core / 256 KB total
- 每 SM 4 个 warp selector，独立 Fetch/Decode
- 最高 64 warps/ SM
- Tensor Cores: 16×16×16 [fp16, fp16, fp32] 格式

### 5.0.3 H100 整体性能

- 144 SMs
- Tensor Cores (systolic array MMA): 989 TFLOPS (fp16)
- SIMD: 134 TFLOPS (fp16), 67 TFLOPS (fp32)

### 5.0.4 Tensor Core 占总算力比例的趋势

| GPU | Tensor Core 占比 |
|---|---|
| V100 | 89% |
| A100 | ~50% (?) |
| H100 | 94% |
| B100 | 96% |
| Next Gen | 98% |

GPU 越来越"专用化"于矩阵乘法。

### 5.0.5 NVIDIA 芯片代际演进

| 代际 | 关键新特性 |
|---|---|
| V100 | Tensor Core |
| A100 | Tensor Core 3rd gen, Sparsity, Async Copy, L2 Cache Residency |
| H100 | Tensor Core 4th gen, Sparsity, FP8, Transformer Engine, Async Exec, Distributed SHMEM, DPX |
| B100 | Tensor Core Next gen, Sparsity, Transformer Engine 2nd gen, FP4, Decompression Engine |

### 5.0.6 B100 的 Tensor Core 编程模型

- Register bandwidth limits for tensor cores
- Tensor data in SMEM and TMEM
- Single threads execute MMA ⇒ No more warps!
- 新指令族：`tcgen05.alloc`, `cp.async.bulk.tensor` (配合 mbarrier), `tcgen05.mma batch`, `tcgen05.commit`, `tcgen05.fence`
- "Not your father's CUDA"

### 5.0.7 TPU v1 细节

- 算术单元约占芯片 30% 面积（注意控制逻辑的极低面积占比）
- 5 条关键指令：Read Host Memory, Write Host Memory, Read Weights, Matrix_Multiply/Convolve, Activate
来源：Jouppi et al. 2017

### 5.0.8 Hardware Lottery：TPU ↔ Transformer 的正反馈

TPU → 密集矩阵乘法（OI ∝ n）→ 设计 Transformer 模型 → Transformer 模型主导 → 硬件更加专用化于矩阵乘法。Sara Hooker 提出的概念。

### 5.0.9 Plasticine 可重构数据流架构

- S (Switch)、PMU (Pattern Memory Unit)、PCU (Pattern Compute Unit)
- AI Models ⇒ Dataflow Architecture (Prabhakar, Zhang et al. ISCA 2017)
- 数据流图：GEMM + Parallel Patterns (map/filter/reduce) + GEMM
- "No instructions ⇒ No instruction fetch/decode overhead"（无指令 = 无取指译码开销）

### 5.0.10 AI 芯片厂商全景图

Google TPU3、Apple Neural Engine、AWS Trainium 2、Intel DL Inference Accelerator、Cerebras WSE、SambaNova Cardinal SN10

### 5.1 TPU 的关键观念

- 面向机器学习的关键热点：矩阵乘法
- 以大规模 systolic array 为算力中心
- 让大量晶体管真正用于“乘加”，而不是复杂控制

### 5.2 为什么 TPU 能有更高 perf/watt

因为它把很多 CPU / GPU 中通用但昂贵的能力削减掉了：

- 更少的通用指令控制
- 更强的数据复用路径
- 更匹配主流 AI 负载的数值格式与存储组织

### 5.3 但它也有边界

- 对极不规则控制流不友好
- 过于特殊的算子可能仍需要回退到通用处理器或其他单元

---

## 6. Roofline 与专用化

专用化不是只提升峰值算力，还可能同时改变：

- 有效带宽
- 片上缓存 / SRAM 组织
- 数据复用路径
- 指令开销摊销方式

### 6.0.1 数据移动能量的精确数值

| 操作 | 能量 |
|---|---|
| Integer op | ~1 pJ |
| Floating point op | ~20 pJ |
| 读 64 bits from small local SRAM (1mm) | ~26 pJ |
| 读 64 bits from LPDDR | ~1200 pJ |

来源：Bill Dally (NVIDIA), Tom Olson (ARM)

### 6.0.2 指令流控制开销摊销

| 指令类型 | 控制开销占比 |
|---|---|
| Half-precision FMA | 2000% |
| Half-precision DP4 (vec4 dot product) | 500% |
| Half-precision 4×4 MMA | 27% |

核心原理：**摊销指令流处理的开销到单条复杂指令的多个操作上**。这就是为什么 MMA/Tensor 指令比标量 FMA 更高效。

### 6.0.3 数值格式的具体参数

| 格式 | 位宽 | Sign | Exponent | Mantissa | Range |
|---|---|---|---|---|---|
| FP32 | 32 | 1 | 8 | 23 | ±3.4×10³⁸ |
| BF16 | 16 | 1 | 8 | 7 | 同 FP32，精度更低 |
| BF8 E4M3 | 8 | 1 | 4 | 3 | 0-448 |
| BF8 E5M2 | 8 | 1 | 5 | 2 | 0-57344 |

FP32 公式：$$-1^S \times (1 + M \times 2^{-23}) \times 2^{E-127}$$

### 6.1 一条重要线索

随着硬件越来越专用，系统会努力让：

- 单次指令或单次调度所触发的有效计算量更大

于是 instruction overhead 被摊得更薄。

### 6.2 典型例子

- 标量 FMA：一次触发计算少
- 向量 dot product：一次触发计算更多
- MMA / tensor instruction：一次触发一个小矩阵乘加

这本质上就是在提高“每单位控制成本对应的有效算术量”。

> 对应源码：`lecture10_part2.cpp`
> 内容：roofline 分析、算术强度、不同平台性能上限和能效比较。

---

## 7. 数值格式也是专用化的重要组成部分

### 7.1 为什么 AI 可以接受更低精度

许多神经网络对数值噪声有容忍空间，因此可使用：

- BF16
- FP16
- FP8
- 甚至更激进的低精度表示

### 7.2 降低精度带来的系统收益

- 存储更省
- 带宽压力更低
- 单位面积可放更多算术单元
- 数据搬运能耗更低

### 7.3 这也是专用化的一部分

硬件不是只专门化在“算什么”，还专门化在：

- 数据怎么表示
- 中间结果如何累加
- 不同精度怎样混合使用

---

## 8. Dataflow 架构：进一步削弱指令中心化

### 8.1 核心思想

- 把计算表示为数据流图
- 节点准备好输入令牌后就自动触发计算
- 更强调空间映射与流式执行

### 8.2 为什么它对 AI 有吸引力

- AI 图中有大量规则算子与可流水化阶段
- 数据流模型容易做算子融合与跨层重叠
- 控制路径可以极度简化

### 8.3 与 systolic array 的关系

- systolic array 是一种非常规则的数据流阵列
- 更广义 dataflow 架构则允许更复杂图结构与流水组织

---

## 9. Hardware Lottery：算法成功与硬件生态的耦合

### 9.1 概念

有些研究路线之所以大获成功，不只是因为算法本身最好，也因为：

- 它恰好特别适合当下最易获得、最强大的硬件平台

### 9.2 为什么这对系统研究很重要

- 硬件会塑造软件与算法的演进方向。
- 能高效映射到主流硬件的算法，往往更容易形成生态优势。

### 9.3 对 AI 的启发

- Transformer、GEMM-heavy 模型与张量硬件之间存在强正反馈
- 系统与硬件不是被动执行算法，而是在反过来塑造算法主流

---

## 常见误区

1. **误区：专用化只是为了更快。**
   更准确地说，是为了更高能效与更低数据移动成本。
2. **误区：GPU 已经够专用了，所以不需要 TPU/NPU。**
   对矩阵主导负载，进一步削减控制开销仍能带来巨大收益。
3. **误区：ASIC 一定最好。**
   若负载变化快或规模不足，开发与灵活性成本可能压过收益。
4. **误区：数值精度只是算法问题。**
   它同时决定存储、带宽、能耗与硬件设计空间。

---

## 对应源码

| 文件 | 主题 | 重点 |
|---|---|---|
| `lecture10_part1.cpp` | systolic array | 权重驻留、局部通信、矩阵乘法映射 |
| `lecture10_part2.cpp` | roofline 与专用化收益 | 算术强度、平台上限、能效权衡 |

---

## 学完本讲应做到

- 能解释硬件专用化的根本驱动力是能效。
- 能比较 CPU、GPU、FPGA、ASIC 在可编程性与效率上的权衡。
- 能理解 systolic array 为什么适合矩阵乘法。
- 能意识到硬件生态会反向塑造主流算法。

