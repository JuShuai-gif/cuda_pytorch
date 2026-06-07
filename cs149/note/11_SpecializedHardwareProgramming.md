# CS149 第 11 讲：如何为 AI 专用硬件编程

**PDF**：Lecture 11 - Programming Specialized Hardware for AI

---

## 本讲核心问题

1. 专用硬件不是“有了就自动快”，那程序应该怎样映射上去？
2. 同步、异步、流水化之间的边界在哪里？
3. 为什么现代 AI 硬件编程越来越强调数据搬运与计算重叠？
4. 如何理解 TPU、H100/B100、SambaNova 这类平台背后的统一模式？

---

## 1. 同步执行与异步执行

### 1.1 同步执行

同步模型下，程序往往呈现：

- 先加载
- 再计算
- 再存回
- 然后处理下一块

优点：

- 逻辑简单
- 易于推理正确性

缺点：

- 很多硬件资源会在等待中空转
- 无法充分重叠数据移动与计算

### 1.2 异步执行

异步模型下，程序会把：

- 下一块数据加载
- 当前块计算
- 上一块结果回写

尽量重叠起来。

### 1.3 为什么 AI 系统必须走向异步

因为现代 AI kernel 常有明显的数据搬运阶段：

- HBM -> shared memory / SRAM
- SRAM -> register / tensor memory
- 计算完成后再写回

若这些阶段串行排队，吞吐会大幅下降。

> 对应源码：`lecture11_part1.cpp`
> 内容：异步流水线、生产者-消费者、重叠加载/计算/存储时间线模拟。

---

## 2. 三类平台，不同外观，同一主线

课程里涉及的典型平台有：

- Google TPU：systolic array
- NVIDIA H100/B100：Tensor Core + 异步拷贝 / TMA
- SambaNova：可重构数据流架构

虽然表面接口不同，但核心目标一致：

1. 最大化片上数据复用
2. 最小化片外流量
3. 让数据搬运与计算重叠
4. 用更高层抽象组织规则矩阵与张量计算

---

## 3. Systolic Array 的数据流选择

### 3.1 三种常见驻留策略

- **Weight-Stationary**：权重留在 PE，中间值和输入流动
- **Output-Stationary**：部分和留在 PE
- **Input-Stationary**：输入激活留在 PE

### 3.2 它们本质上在优化什么

都在回答同一个问题：

- 哪一种数据最值得“少移动”？

因为不同工作负载中最贵、最常复用的对象不同。

### 3.3 为什么 TPU 倾向权重驻留

- 在很多推理场景中，权重会被大量输入复用
- 把权重固定在 PE，可减少重复加载成本

---

## 4. 现代 GPU 编程为什么越来越不像“老 CUDA”

### 4.1 传统 CUDA 的视角

过去常强调：

- warp
- thread block
- shared memory
- 手写同步

### 4.2 新一代张量硬件的变化

在 H100 / B100 一类平台上，性能关键越来越集中在：

- Tensor Core MMA 指令
- 异步拷贝
- Tensor Memory Accelerator（TMA）
- 更高级的 pipeline 与 barrier 机制

### 4.3 这意味着什么

高性能内核的核心工作，逐渐从“写很多线程逻辑”转向：

- 组织数据块如何搬运
- 组织多阶段流水如何衔接
- 组织片上 tile 如何在共享内存和寄存器间流动

---

## 5. TMA / 异步数据搬运：把地址生成和搬运也专用化

### 5.1 为什么普通加载指令不够

如果让大量线程自己去做：

- 地址计算
- 小粒度 load/store
- 重复边界处理

那么会浪费很多指令与寄存器资源。

### 5.2 专用搬运单元的价值

像 TMA 这样的机制允许：

- 用一个描述符描述一个张量区域
- 由硬件异步完成批量搬运
- 完成后通过 barrier 通知消费者

### 5.3 它解决了什么

- 显著减少指令条数
- 降低地址生成开销
- 让数据搬运与计算更好解耦
- 减轻普通线程在搬运上的负担

---

## 6. DSL 的作用：把“流水 + tile + 同步”抽象出来


### 6.0.1 ThunderKittens 的完整设计原则

三条核心原则：
1. **tile of 16×16 as primitive data type** — 以块为基本数据单元
2. **Asynchrony everywhere** — 异步搬运无处不在
3. **High-level GPU coordination patterns (producer-consumer)** — 以生产者-消费者模式组织

ThunderKittens 三步 matmul 实现：
- Step 1: 定义 layout (`gl<bf16,1,1,...>`, `st_bf<64,64>`, `rt_fl<16,256>` 等)
- Step 2: 定义 pipeline 和 producer (`NUM_CONSUMER_WARPS=8`, `INPUT_PIPE_STAGES=4`, `decrease_registers<40>`)
- Step 3: compute (`mma_AB`, `mma_async_wait` 等)

### 6.1 为什么需要 DSL

专用硬件编程越来越复杂，直接写低级接口会遇到：

- 模板多
- 约束多
- 同步细节繁琐
- 稍有不慎就浪费大量性能

### 6.2 ThunderKittens 之类 DSL 的意义

它们通常提供：

- tile 作为基本数据类型
- 明确的 producer / consumer pipeline 模式
- 高层块级协调抽象

### 6.3 本质上在做什么

- 把底层片上流水模板“结构化”
- 让程序员关注块级算法，而不是每条微指令

这与 ISPC、Halide、Triton 的思想是一脉相承的：

- 提升抽象层，释放编译器 / 运行时优化空间

---

## 7. Metapipelining：流水的流水

### 7.1 定义

metapipelining 可以理解为：

- 不只是让单个算子内部形成流水
- 还让多个 tile、多个阶段、多个循环层级之间形成分层流水

### 7.2 为什么它强

因为它能让：

- tile 的加载
- tile 的计算
- tile 的写回
- 外层块的切换

同时在不同阶段重叠推进。

### 7.3 典型收益

- 减少阶段性空泡
- 提高设备利用率
- 支持更激进的跨层融合

> 对应源码：`lecture11_part2.cpp`
> 内容：metapipeline、分层粗粒度流水、数据流 tile 在多阶段中的推进。

---

## 8. 数据流执行与 token 控制

### 8.1 数据流模型的关键差异

传统程序强调：

- 指令顺序
- 控制流跳转

数据流系统更强调：

- 节点何时拿到足够输入
- 数据到位即可触发计算
- token 决定阶段是否前进

### 8.2 为什么适合 AI

很多 AI 图本身就是：

- 大量规则算子
- 边界清晰
- 可以 tile 化
- 能形成稳定流式管线

### 8.3 对程序员思维的改变

你需要从“我写了哪些线程”转向“哪些 tile 在什么时候准备好，谁消费谁”。

---

## 9. GPU 与 RDU / Dataflow 平台的关键差异

### 9.0.1 SambaNova SN40L RDU 完整硬件规格

| 参数 | 数值 |
|---|---|
| PCU 数量 | 1,040 |
| PMU 数量 | 1,040 |
| 峰值性能 | 638 TFLOPS (bf16) |
| 片上 SRAM | 520 MB |
| HBM | 64 GB |
| DDR | 1.5 TB |

- PCU: systolic and SIMD compute (16 × 8 bf16)
- PMU: 每个 0.5 MB，高地址生成灵活性和带宽
- S: Mesh switches（互连交换机）
- AGCU: 片外内存和 IO 的门户

### 9.0.2 为什么专用的重要性：NVIDIA 市场数据

- NVIDIA 2025 季度收入 > $470 亿
- GPU AI kernel 经常在价值数亿美元的 GPU 集群上运行数月
- FlashAttention-2 从 A100 上 ~70% 利用率退化到 H100 上 ~35%
- 花了 2 年才通过 FlashAttention-3 恢复到 ~65%
- "劣质 kernel 浪费了数十亿美元的计算资源"

### 9.0.3 TMA vs LDGSTS 对比

- A100: LDGSTS（通过 L1 cache 加载到 shared memory）
- H100: TMA（Tensor Memory Accelerator，可**绕过 L1**）
- TMA 消除数千条指令和内存寻址开销
- TMA 消除不必要的 L1 和寄存器数据搬运

### 9.0.4 Warpgroup 与 PTX 定义

- **Warpgroup**: 128 个连续线程（4 个 warp）
- **PTX**: Parallel Thread Execution — NVIDIA 虚拟指令集架构

### 9.0.5 Metapipelining 的完整定义

"Hierarchical coarse-grained pipeline: A 'pipeline of pipelines'"

9 个关键特性：
1. 利用嵌套循环并行性
2. 将并行模式（循环）转换为流式流水
3. 在循环体中插入流水阶段
4. 流水阶段并行执行，重叠多个循环迭代
5. 阶段间中间数据存储在双缓冲区中
6. 处理具有不同执行时间的不平衡阶段
7. 缓冲区可用于改变访问模式（如转置数据）
8. 在 fusion 无法工作的情况下也能使用

### 9.0.6 数据流执行与 Token 控制

"Dataflow execution with token control ⇒ no lock-based synchronization"

传统程序强调指令顺序和控制流跳转。数据流系统强调节点何时拿到足够输入——数据到位即可触发计算，token 决定阶段是否前进。

### 9.0.7 Llama3.1 8B 完整模型图

Embedding → Decoder 0-31 → Classifier → Sampling

每个 Decoder 内部：
RMS Norm → Q/K/V GEMM → QK matmul → Scale/Maskfill/Softmax → PV matmul → O GEMM → RMS Norm → Gate/Up GEMM → SilU/Mul → Down GEMM → Add（含 AllReduce 通信点）

### 9.0.8 GPU vs RDU 的 Kernel 融合程度

| 维度 | GPU (TensorRT-LLM) | RDU (SambaNova) |
|---|---|---|
| Kernel 融合 | Limited (~10 个 kernel 边界) | 整个 Decoder 融合为 1 个 Kernel |
| Launch 开销 | 高 | 零 |
| 数据局部性 | 低 | 5× SRAM 优势 (520MB vs 100MB) |
| 片外中间流量 | 大量 GB | 数据流融合消除 |

### 9.0.9 Kernel 调用次数对比

每 token 生成：
- GPU: ~800 次 kernel 调用
- RDU: ~3 次 kernel 调用
- **100 倍更少的 kernel 调用**

### 9.0.10 AllReduce 与计算完全重叠

在数据流平台上："Fully overlap AllReduce with weight load and compute. AllReduce does not consume HBM capacity or bandwidth."

### 9.0.1 SambaNova SN40L RDU 完整硬件规格

| 参数 | 数值 |
|---|---|
| PCU 数量 | 1,040 |
| PMU 数量 | 1,040 |
| 峰值性能 | 638 TFLOPS (bf16) |
| 片上 SRAM | 520 MB |
| HBM | 64 GB |
| DDR | 1.5 TB |

- PCU: systolic and SIMD compute (16 × 8 bf16)
- PMU: 每个 0.5 MB，高地址生成灵活性和带宽
- S: Mesh switches（互连交换机）
- AGCU: 片外内存和 IO 的门户

### 9.0.2 为什么专用的重要性：NVIDIA 市场数据

- NVIDIA 2025 季度收入 > $470 亿
- GPU AI kernel 经常在价值数亿美元的 GPU 集群上运行数月
- FlashAttention-2 从 A100 上 ~70% 利用率退化到 H100 上 ~35%
- 花了 2 年才通过 FlashAttention-3 恢复到 ~65%
- "劣质 kernel 浪费了数十亿美元的计算资源"

### 9.0.3 TMA vs LDGSTS 对比

- A100: LDGSTS（通过 L1 cache 加载到 shared memory）
- H100: TMA（Tensor Memory Accelerator，可**绕过 L1**）
- TMA 消除数千条指令和内存寻址开销
- TMA 消除不必要的 L1 和寄存器数据搬运

### 9.0.4 Warpgroup 与 PTX 定义

- **Warpgroup**: 128 个连续线程（4 个 warp）
- **PTX**: Parallel Thread Execution — NVIDIA 虚拟指令集架构

### 9.0.5 Metapipelining 的完整定义

"Hierarchical coarse-grained pipeline: A pipeline

### 9.1 GPU 常见问题

- kernel 边界多
- launch 开销和阶段同步多
- 中间结果跨 kernel 往往要回到较慢存储层

### 9.2 数据流平台的优势

- 更激进的算子融合
- 更大的片上存储
- 更少的 kernel 边界
- 更容易把整段推理流程编成单个持续流式执行图

### 9.3 本讲的系统结论

现代 AI 加速器优化不只是让单个 GEMM 更快，而是：

- 让整条数据流路径上的所有阶段更紧密地黏合在一起

---

## 常见误区

1. **误区：专用硬件编程只是换一组更快的指令。**
   真正关键是数据流、异步流水和片上存储组织。
2. **误区：异步只是为了隐藏一点延迟。**
   在 AI 硬件里，异步往往决定吞吐是否能逼近峰值。
3. **误区：DSL 只是语法糖。**
   DSL 的价值在于把可优化的模式结构化表达出来。
4. **误区：高性能 GPU 编程仍主要是线程层面的技巧。**
   现在更核心的是 tile 级搬运与 pipeline 组织。

---

## 对应源码

| 文件 | 主题 | 重点 |
|---|---|---|
| `lecture11_part1.cpp` | 异步流水 | 加载、计算、存储重叠 |
| `lecture11_part2.cpp` | metapipelining | 分层流水、tile 级数据流推进 |

---

## 学完本讲应做到

- 能解释为什么现代 AI 硬件编程必须重视异步流水。
- 能理解 TMA / 专用搬运单元为何重要。
- 能用数据流与 tile 流动的视角看待高性能 kernel。
- 能认识到 DSL 在专用硬件时代的真正价值。

