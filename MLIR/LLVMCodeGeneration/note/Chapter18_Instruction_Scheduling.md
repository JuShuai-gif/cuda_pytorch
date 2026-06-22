# Chapter 18: Instruction Scheduling

> **From the perspective of a production AI compiler engineer who needs to understand LLVM deeply to work on MLIR/Triton/AI compiler stacks.**

## 核心概念（详细展开）

### 指令调度的本质

指令调度（Instruction Scheduling）是一种低级优化，它**不改变计算语义**，但改变指令的顺序以提高以下某个或某几个指标：

- **指令级并行度（ILP）**：最大化可并行执行的指令数
- **寄存器压力（Register Pressure）**：最小化同时活跃的寄存器数
- **延迟隐藏（Latency Hiding）**：将高延迟指令（如 load）与无关计算交错，隐藏等待时间
- **吞吐量（Throughput）**：最大化每个周期的指令发射数（issue width 利用）

**为什么指令调度对 AI 编译器至关重要：**

1. **GPU Warp 调度**：GPU 依赖 warp-level 的指令调度来隐藏内存延迟。Triton 的 `tl.load` 通常在循环中交错计算，手动实现类似调度的效果
2. **AI 加速器（TPU/NPU）**：通常有高度流水线化的 systolic array，指令级别的精确调度直接影响利用率
3. **MLIR 的调度框架**：IREE 等项目正在 MLIR 中构建调度抽象，LLVM 的调度经验是重要参考
4. **Triton 的特殊性**：Triton **完全不使用** LLVM 的调度——它生成 PTX 后依赖 `ptxas`（NVIDIA 的 PTX 汇编器）来调度

### 调度的三类信息

```
┌─────────────────────────────────────────────────┐
│              LLVM Scheduling Stack               │
│                                                  │
│  1. ScheduleDAG (DDG)                           │
│     - 数据依赖图：producer/consumer 关系          │
│     - 内存依赖约束                                │
│     - Edges = use-def chains                    │
│                                                  │
│  2. MachineSchedStrategy                        │
│     - 调度算法/启发式                             │
│     - 控制 pickNode/tryCandidate 等决策           │
│     - Top-down / Bottom-up / Bidirectional      │
│                                                  │
│  3. TargetSchedModel / MCSchedModel             │
│     - 目标硬件的资源信息                           │
│     - 指令延迟、处理单元、issue width             │
│     - 调度事件的 bindings                         │
└─────────────────────────────────────────────────┘
```

## LLVM / MLIR 流程（深入）

### MachineScheduler 在 Pipeline 中的位置

```
Machine IR (post-instruction-selection)
    │
    ▼
┌──────────────────────────────┐
│  PHI Elimination             │
├──────────────────────────────┤
│  Register Coalescing         │
├──────────────────────────────┤
│  Pre-RA MachineScheduler ←──  ← 本章焦点   │
│  (enableMachineScheduler)    │
├──────────────────────────────┤
│  Register Allocation         │
├──────────────────────────────┤
│  Post-RA MachineScheduler ←──  ← 同样可定制的   │
│  (enablePostRAMachineScheduler)│
├──────────────────────────────┤
│  Prologue/Epilogue Insertion │
│  Frame Lowering              │
└──────────────────────────────┘
```

**启用调度器：**

```cpp
class H2BLBSubtarget : public H2BLBGenSubtargetInfo {
public:
  bool enableMachineScheduler() const override { return true; }
  bool enablePostRAMachineScheduler() const override { return true; }
};
```

**调度器的默认行为：**
- 未启用时：SDISel 使用简单的线性化启发式
- 启用时：SDISel 退化为简单线性化，MachineScheduler 承担重调度
- 不覆盖调度策略时：All `mayLoad` 指令假定延迟为 `DefaultLoadLatency` (4 cycles)

### ScheduleDAGInstrs 类与 DDG

```
指令序列:
  a = instrA
  b = instrB a
  c = instrC a
  d = instrD b, c

对应的 DDG (Data Dependency Graph):
        instrA
       /      \
      ▼        ▼
    instrB   instrC
       \      /
        ▼    ▼
        instrD

调度方向说明:
  - Top-down:    从 instrA 开始，向下调度
  - Bottom-up:   从 instrD 开始，向上调度
  - Bidirectional: 两头同时开始（默认）
```

**Edges 的方向语义：**
如果 DDG 中 edge 从 A → B，则在最终基本块中 B 必须出现在 A 之前。换言之，edges 表示 use-def chains（看起来可能反直觉）。

#### DDG Mutations（图突变）

Mutations 在 DDG 完全构建后修改其边，用于：

1. **添加额外的调度约束**（如某些指令必须在一起）
2. **Group memory operations**（load/store clustering 以减少 cache miss）
3. **添加延迟约束**（保证高延迟操作之间的间距）

```cpp
// 创建自定义 mutation
class MyDAGMutation : public ScheduleDAGMutation {
public:
  void apply(ScheduleDAGInstrs *DAG) override {
    // 遍历 DDG 并修改依赖关系
    for (SUnit &SU : DAG->SUnits) {
      // 例如：强制 WIDENING_SMUL 在其输入之后立即调度
      if (SU.getInstr()->getOpcode() == MyTarget::WIDENING_SMUL) {
        for (SDep &Pred : SU.Preds) {
          if (Pred.getSUnit()->getInstr()->mayLoad())
            Pred.setLatency(1);  // 减少延迟 → 优先调度
        }
      }
    }
  }
};

// 注册 mutation
DAG->addMutation(std::make_unique<MyDAGMutation>());
```

**内置的 mutation（直接可用）：**
- `createLoadClusterDAGMutation()` — 将同类型的 load 聚类
- `createStoreClusterDAGMutation()` — 将 store 聚类

⚠️ **关键警告：** Mutation 不能创建循环依赖，否则调度器无法找到有效解。

### MachineSchedStrategy 类（调度策略）

```cpp
// 自定义 pre-RA 调度策略
class H2BLBPreRASchedStrategy : public GenericScheduler {
protected:
  bool tryCandidate(SchedCandidate &Cand, SchedCandidate &TryCand,
                    SchedBoundary *Zone) const override {
    // 1. 先运行父类启发式（GenericScheduler 的默认逻辑）
    bool BetterCand = GenericScheduler::tryCandidate(Cand, TryCand, Zone);

    // 2. 如果父类已经判定 TryCand 更好且有实际原因 → 保留
    if (BetterCand && TryCand.Reason != NodeOrder &&
        TryCand.Reason != NoCand)
      return true;

    // 3. 自定义启发式
    if (Zone != nullptr) {
      // 优先调度 loads（隐藏内存延迟）
      if (TryCand.SU->getInstr()->mayLoad()) {
        TryCand.Reason = Stall;
        return true;
      }
      // 优先调度高延迟指令
      if (TryCand.SU->getInstr()->getOpcode() == MyTarget::WIDENING_SMUL) {
        TryCand.Reason = Stall;
        return true;
      }
    }
    return TryCand.Reason != NoCand;
  }
};
```

**Reason 字段的含义：**
- `NoCand`：没有其他候选指令（无需理由）
- `NodeOrder`：原始基本块顺序（平局打破机制）
- `Stall`：因为避免 stall（你手动设置此值表示你优先调度该指令）
- 其他 system 提供的 reason：`RegPressure`、`ResourceReduce` 等

#### 调度策略的连接

```cpp
ScheduleDAGInstrs *
H2BLBPassConfig::createMachineScheduler(MachineSchedContext *C) const {
  ScheduleDAGMILive *DAG = new ScheduleDAGMILive(
      C, std::make_unique<H2BLBPreRASchedStrategy>(C));
  // 注册 mutations
  DAG->addMutation(createLoadClusterDAGMutation());
  DAG->addMutation(createStoreClusterDAGMutation());
  return DAG;
}
```

#### 调度方向策略

```cpp
void H2BLBSubtarget::overrideSchedPolicy(MachineSchedPolicy &Policy,
                                          unsigned NumRegionInstrs) const {
  Policy.OnlyTopDown = true;   // 仅做 top-down 调度
  Policy.OnlyBottomUp = false;
  // 也可设置 Policy.OnlyBottomUp = true 或都不设（默认 bidirectional）
}
```

**调度方向对比：**
- **Top-down**：ready queue = 已调度所有后继的节点
- **Bottom-up**：ready queue = 已调度所有前驱的节点（常产生更好的寄存器压力）
- **Bidirectional**：LLVM 默认，兼顾两者

### 调度模型（Scheduling Model）

这是调度框架中最复杂的部分。它描述**目标硬件的能力**。

#### 模型的四大组件

```
┌──────────────────────────────────────────────┐
│            SchedMachineModel                 │
│  - IssueWidth: 每周期最大发射数                  │
│  - LoadLatency: Load 指令的默认延迟              │
│  - CompleteModel: 是否所有指令都覆盖             │
├──────────────────────────────────────────────┤
│            ProcResourceUnits                 │
│  - 处理单元（ALU、Load/Store Unit、FPU 等）      │
│  - BufferSize: 缓冲区大小（影响 in/out-of-order）│
│  - ProcResGroup: 资源分组                      │
├──────────────────────────────────────────────┤
│            Scheduling Events                 │
│  - SchedRead: 每个操作数的读取事件               │
│  - SchedWrite: 每个定义的写入事件                │
│  - 每指令最多有 [#explicit defs, #operands] 个事件│
├──────────────────────────────────────────────┤
│            Scheduling Bindings               │
│  - WriteRes: 绑定 Write 事件到处理单元           │
│  - ReadAdvance: 绑定 Read 事件（可选转发路径）    │
│  - InstRW: 一体化的绑定方案（推荐）                │
└──────────────────────────────────────────────┘
```

#### 调度事件描述

```tablegen
// 方法 1: WriteRes/ReadAdvance（事件在指令中定义）
let SchedRW = [WriteWUMUL, ReadWUMULArg0, ReadWUMULArg1] in
def WIDENING_UMUL : H2BLBWidenIMul<"wumul", /*isSign=*/0>;

// 方法 2: Sched 类继承
def WIDENING_SMUL : H2BLBWidenIMul<"wsmul", /*isSign=*/1>,
      Sched<[WriteWSMUL, ReadWSMULArg0, ReadWSMULArg1]>;
```

#### 处理单元定义

```tablegen
// 基本单元
def ALURes : ProcResource<1>;    // 1 个 ALU 单元
def MemRes : ProcResource<2>;    // 2 个内存单元
def FMAUnit : ProcResource<4>;   // 4 个 FMA 单元（如 Tensor Core）

// 资源分组（用于需要多个资源的指令）
def MemAndALURes : ProcResGroup<[MemRes, ALURes]>;
```

#### 调度 Bindings

**方法 1: WriteRes/ReadAdvance（传统方法）**

```tablegen
let SchedModel = H2BLBDefaultModel in {
  let Latency = 2 in
  def : WriteRes<WriteWSMUL, [ALURes]>;       // 写事件消耗 ALU，延迟 2
  def : ReadAdvance<ReadWSMULArg0, 0>;         // 操作数 0 无转发路径
  def : ReadAdvance<ReadWSMULArg1, 1>;         // 操作数 1 有 1-cycle 转发
}
```

**方法 2: InstRW（推荐方法）**

```tablegen
let SchedModel = H2BLBDefaultModel in {
  // 定义事件和 binding 的一体化方案
  let Latency = 3 in
  def DefaultWriteLoad : SchedWriteRes<[MemRes]>;

  // 使用正则表达式匹配指令
  def : InstRW<[DefaultWriteLoad], (instregex "^LD[^i]*$")>;
}
```

**为什么推荐 InstRW：**
- 事件和 binding **在同一位置**（scheduling model 内）
- 避免事件的**全局命名污染**（SchedWrite 是全局的，新处理器可能使旧事件定义失效）
- 更好的可维护性（不需要在多处修改）

#### 构建调度模型的推荐步骤

```
1. 创建 SchedMachineModel 实例（设置 IssueWidth, LoadLatency）
2. 描述处理单元（至少覆盖第一轮迭代需要的）
3. 为一组指令创建 SchedWriteRes/SchedReadAdvance
4. 用 InstRW 装饰指令
5. 写 .mir 测试验证调度效果
6. 重复步骤 2-5 覆盖全部指令
7. 设置 CompleteModel = true
```

## 调度模型深度解析：LLVM 微架构建模实战

> 本节整合自 Zhihu 高赞回答（原文翻译自 Min Hsu 的 LLVM Scheduling Model 系列文章），补充了本书 Chapter 18 中未详细展开的实战细节，是理解 LLVM 调度模型从"会用"到"精通"的关键内容。

### 问题背景：编译器指令调度 vs CPU 乱序执行

**核心问题**：LLVM 实现了指令调度，CPU 也实现了乱序执行，二者功能部分重叠——区别和联系是什么？如何协同？

**TL;DR**：
- **编译器调度**（LLVM）：静态地重排 **直线代码** 中的指令顺序，目的是最小化延迟、最大化吞吐量。它对硬件是透明的，通过调度模型"预判"哪些指令可以并行、哪些需要等待。
- **CPU 乱序执行**（OOO）：硬件在运行时动态调度指令，有真实的保留站（reservation station）和重排序缓冲（ROB）。它看到的是编译器排好的"静态顺序"，但在运行时可能重新排序。
- **协同关系**：编译器尽力为硬件"铺好路"——把高延迟指令提前、把有依赖的指令拉远，减少硬件调度器的压力。编译器做**粗粒度**调度，硬件做**细粒度**动态调度。二者互补而非重复。

**AI 编译器启示**：
- GPU（NVIDIA）是 in-order 的（warp 级别），编译器调度至关重要——这也是为什么 Triton 依赖 `ptxas` 做最终调度
- AI 加速器（TPU/NPU）多数是 in-order 或 VLIW，编译器调度质量直接决定性能天花板
- 理解 LLVM 调度模型是理解"AI 编译器如何将运算映射到硬件"的基础

### 1. 基本概念：SchedWrite / SchedRead / WriteRes

#### 为什么需要调度模型

最简单的方式是 "opcode=ADD 的延迟是 X"，但有些架构 opcode 多达**几万条**（如 RISC-V 的 RVV 伪指令），逐条写延迟表不可扩展。而且很多指令调度特征相同（各种 ALU 操作延迟一样）。

LLVM 的解决方案：**基于操作数读写 token** 的描述方式。

#### 三步构建调度信息

**Step 1**：在指令定义中为操作数分配 token：

```tablegen
// From llvm/lib/Target/RISCV/RISCVInstrInfoM.td
def DIV : ALU_rr<0b0000001, 0b100, "div">,
               Sched<[WriteIDiv, ReadIDiv, ReadIDiv]>;
```

- 第一个操作数是 **写操作数**（目标寄存器，也叫 definition），用 `SchedWrite` token
- 后面是两个 **读操作数**（源寄存器，也叫 use），用 `SchedRead` token
- **默认假设**：`SchedRead` 是瞬时完成的，不消耗延迟也不占用资源
- **结论**：写操作数实际上决定了整条指令的调度属性

**Step 2**：在调度模型中定义这些 token 使用的硬件资源：

```tablegen
// From llvm/lib/Target/RISCV/RISCVSchedSiFive7.td
// Integer division
def : WriteRes<WriteIDiv, [SiFive7PipeB, SiFive7IDiv]> {
    let Latency = 66;
    let ReleaseAtCycles = [1, 65];
}
```

**WriteRes** 为一个 `SchedWrite` 指定了三类信息：

| 属性 | 含义 | 本例 |
|------|------|------|
| **Latency** | 从 issue 到结果对后续 use 可见的周期数 | 66 cycles |
| **使用的硬件资源** | 指令占用的处理器资源列表 | `SiFive7PipeB`, `SiFive7IDiv` |
| **ReleaseAtCycles** | 在每个资源上"占坑"的周期数 | PipeB 占 1, IDiv 占 65 |

**Step 3**：所有 SchedRead 默认不占资源，无需额外声明（除非用 `ReadAdvance` 处理前向通路 bypass）。

#### 关键概念区分：Latency vs ReleaseAtCycles

这是最容易混淆的两个概念：

| | Latency（延迟） | ReleaseAtCycles（资源占用） |
|---|---|---|
| **含义** | 数据依赖延迟（RAW hazard） | 结构依赖/资源占用 |
| **影响** | 依赖此结果的后续指令需等待 | 同资源的后续指令派发受限 |
| **计算** | 定义 → 使用的最小安全间隔 | 此指令在该资源上"占坑"几拍 |
| **例** | 除法结果 66 cycles 后才可用 | PipeB 只占 1 cycle，IDiv 占 65 cycles |

**四种典型场景**：

1. **完全不流水化的单元**（经典除法）：`Latency ≈ sum(ReleaseAtCycles)`。吞吐量 = 1/Latency，结果也在计算完成时 ready
2. **完全流水化的单元**（乘法/FMA）：`ReleaseAtCycles` 很小（1 cycle），`Latency` 很大（3-4 cycles）。资源可以"连轴转"，但结果还在后面几级流水线里"跑"
3. **In-order 资源**（BufferSize = 0）：年轻 uop 等老 uop 的 `ReleaseAtCycle` 结束才能占用。数据依赖看 `Latency`，调度约束看 `ReleaseAtCycles`
4. **Latency device**（BufferSize = 1）：后一条 uop 在前一条 **issue 后**再等 `Latency` 个周期才能 issue（不是等到 `ReleaseAtCycle` 完全结束）。调度器强制把它们视为生产者/消费者对

### 2. 建模微架构：处理器资源缓冲区

指令调度的核心目标依赖对"指令在流水线中如何流动"的精确建模。我们需要关注的包括：
- **RAW（Read After Write）数据相关**：可能需要额外等待
- **结构相关（Structural Hazard）**：某些单元被大量请求压垮

关键问题：**我们有没有足够的资源来执行一条指令？**

#### 超标量架构的基本模型

```
┌─────────────────────────────────────┐
│  issue width = 3（每周期最多发送3条）│
│  ┌──────┐ ┌────────┐ ┌───────────┐  │
│  │ Int   │ │ Float  │ │ Load/Store│  │
│  │ ALU   │ │ Unit   │ │ Unit      │  │
│  └──────┘ └────────┘ └───────────┘  │
│     全流水化    全流水化   全流水化   │
└─────────────────────────────────────┘
```

在 TableGen 中，用 `ProcResource` 表示可执行 uops 的"单元"（≈ 一条 pipeline）：

```tablegen
// SiFive P670: 4条整数流水线
def SiFiveP600IEXQ0 : ProcResource<1>;
def SiFiveP600IEXQ1 : ProcResource<1>;
def SiFiveP600IEXQ2 : ProcResource<1>;
def SiFiveP600IEXQ3 : ProcResource<1>;

// 把它们打包成一个资源组：任意一条有空就能执行
def SiFiveP600IntArith : ProcResGroup<[SiFiveP600IEXQ0, SiFiveP600IEXQ1,
                                        SiFiveP600IEXQ2, SiFiveP600IEXQ3]>;
```

**注**：超标量（superscalar）和乱序执行（out-of-order）是彼此独立的概念。超标量关注"执行单元有多宽"，乱序关注"指令按什么顺序执行"。

#### 四种缓冲区模型（BufferSize）

| 类型 | BufferSize | 行为 | 典型用途 |
|------|-----------|------|---------|
| **解耦保留站** (Decoupled) | >0 | 每个资源独立的调度队列 | PowerPC POWER9 Branch unit (16) |
| **统一保留站** (Unified) | -1（默认） | 全局共享 MicroOpBufferSize | 大多数现代 OOO CPU |
| **顺序执行** (In-order) | 0 | dispatch=issue，占资源到 ReleaseAtCycle | 嵌入式CPU、AI加速器 |
| **延迟设备** (Latency device) | 1 | 1个buffer，强制生产者/消费者关系 | 非流水化除法单元 |

**工业实践**：

**AMD Zen2**（解耦保留站，实际使用 ProcResGroup 简化写法）：
```tablegen
// Zen2 有 4 个整数 ALU，各自独立调度器，每个 Buffer = 16
// 但 LLVM 模型中打包为 ProcResGroup，设总 Buffer = 64
def Zn2ALU : ProcResGroup<[Zn2ALU0, Zn2ALU1, Zn2ALU2, Zn2ALU3]> {
    let BufferSize = 64;  // 16×4
}
```

大多数指令关联 `Zn2ALU` 资源组，不关心具体跑在 Zn2ALU[0-3] 的哪一条——这让模型简洁且正确。

**统一保留站**（常用于 SiFive 系列）：
```tablegen
def SiFiveP600Model : SchedMachineModel {
    let IssueWidth = 4;
    let MicroOpBufferSize = 160;  // 全局统一缓冲区
}
```

`MicroOpBufferSize` 等价于 `min(ROB size, 寄存器重命名池容量, 统一保留站真实容量)`。

**延迟设备**（模拟顺序单元在乱序核中的行为）：
```tablegen
// Samsung Exynos M5: 串行整数除法=latency device
let Super = M5UnitC, BufferSize = 1 in
def M5UnitD : ProcResource<1>; // Integer division (serialized)

// RISC-V Rocket: 同样是 latency device
let BufferSize = 1 in {
  def RocketUnitIDiv    : ProcResource<1>; // Int Division
  def RocketUnitFPDivSqrt : ProcResource<1>; // FP Divide/Sqrt
}
```

### 3. ProcResource 单元数量与吞吐量计算

`ProcResource<N>` 的 `N` 表示内部有多少个**单元（unit）**，直接决定吞吐量。

LLVM 计算 reciprocal throughput（吞吐率倒数，即每条指令平均周期数）的核心逻辑：

```cpp
// llvm/lib/MC/MCSchedule.cpp
double MCSchedModel::getReciprocalThroughput(...) {
    // 遍历每条指令使用的写资源
    for (auto &Entry : WriteProcResEntries) {
        unsigned NumUnits = SM.getProcResource(Entry.ProcResourceIdx)->NumUnits;
        double Temp = NumUnits * 1.0 / Entry.ReleaseAtCycle;
        Throughput = Throughput ? std::min(*Throughput, Temp) : Temp;
    }
    return 1.0 / *Throughput;  // 瓶颈资源决定吞吐量
}
```

公式：`throughput = NumUnits / ReleaseAtCycle`，取所有资源中的**最小值**（瓶颈资源）。

**示例**：`ProcResource<3>` 表示三条等效流水线，可以并行派发最多 3 条使用该资源的指令：

```asm
# 三条乘法无 RAW 依赖 → 可并行派发
mul a1, a1, a2
mul t4, t4, t5
mul t0, t0, t1
```

| instruction | Consumed IEX units |
|-------------|-------------------|
| mul a1, a1, a2 | 1/3 |
| mul t4, t4, t5 | 2/3 |
| mul t0, t0, t1 | 3/3 |

### 4. ProcResGroup vs Super Resource：工业实践中的坑

**核心问题**：真实硬件中，执行单元往往**异构**——不是所有流水线都能执行所有指令。例如 3 条 pipe：2 条能做 MUL/ALU，1 条能做 DIV，但只有 1 条能同时做 MUL 和 DIV。

#### ProcResGroup 方案（推荐）

```tablegen
def IEX0 : ProcResource<1>;
def IEX1 : ProcResource<1>;
def IEX2 : ProcResource<1>;

def IntegerArith : ProcResGroup<[IEX0, IEX1]>;
def IntegerMul   : ProcResGroup<[IEX1, IEX2]>;

def : WriteRes<WriteIALU, [IntegerArith]>;
def : WriteRes<WriteIMul, [IntegerMul]>;
def : WriteRes<WriteIDiv, [IEX1]>;        // 直接引用单条 pipe

// LLVM 内部自动展开重叠资源：
// WriteIDiv 实际效果 = WriteRes<WriteIDiv, [IEX1, IntegerArith, IntegerMul]>
```

LLVM 对重叠的 `ProcResGroup` **自动展开**：WriteIDiv 不仅消耗 IEX1，还隐含消耗了 `IntegerArith` 和 `IntegerMul`（因为 IEX1 属于这两个组）。

**验证**：
```asm
mul a1, a1, a2      # 占用 IntegerMul 1/2
mul t4, t4, t5      # 占用 IntegerMul 2/2（满）
div s0, s0, t0      # 尝试占用 IntegerMul → FAIL（满）→ dispatch hazard ✅
```

DIV 被正确阻塞：两条 `mul` 已占满了 `IntegerMul`（2/2），DIV 需要的 IEX1 恰好也在 `IntegerMul` 中。

#### Super Resource 方案（有陷阱！）

```tablegen
// 错误示范！
def IEX : ProcResource<3>;
let Super = IEX in {
    def IntegerArith : ProcResource<2>;
    def IntegerMul   : ProcResource<2>;
    def IntegerDiv   : ProcResource<1>;
}
```

**陷阱**：Super Resource 要求**树形层次**（每个 ProcResource 最多一个 super），无法表达 DAG 结构。当两条 MUL 已占满 IntegerMul(2/2)，整数计数上看 DIV 可能"成功派发"，但实际找不到可用 pipe——**模型允许了一个实际上不可能发生的派发**。这叫"虚假的成功"（false positive）。

#### 选择指南

| 场景 | 推荐方案 |
|------|---------|
| 均质执行单元（所有 pipe 能力相同） | `ProcResource<N>` |
| LSU/AGU 等天然树形层次 | Super Resource（更简洁）|
| pipe 能力部分重叠、非树形 | **ProcResGroup**（唯一正确方案）|
| 12条 pipe，只需关心数量 | Super Resource（远更简洁）|

**历史注记**：`ProcResGroup` 是在 `Super Resource` 之后被发明的（commit 4e67cba8），正是因为 Super Resource 无法处理非树形资源关系。

### 5. 工业案例：AMD Zen3 LSU 建模

AMD Zen3 的 Load/Store Unit（LSU）是一个经典的 Super Resource 应用：

```
结构：3条L/S管线 → Load可用3条，Store可用2条，互斥使用
```

```tablegen
def Zn3LSU : ProcResource<3>;

let Super = Zn3LSU in
def Zn3Load : ProcResource<3> { ... }

let Super = Zn3LSU in
def Zn3Store : ProcResource<2> { ... }

// Load 指令
defm : Zn3WriteResInt<WriteLoad,  [Zn3AGU012, Zn3Load], ...>;
// Store 指令
defm : Zn3WriteResInt<WriteStore, [Zn3AGU012, Zn3Store], ...>;
```

LLVM 自动展开：
```tablegen
// 展开后等价于
defm : Zn3WriteResInt<WriteLoad,  [Zn3AGU012, Zn3Load, Zn3LSU], ...>;
defm : Zn3WriteResInt<WriteStore, [Zn3AGU012, Zn3Store, Zn3LSU], ...>;
```

**Z3Load/Z3Store** 分别限制 load/store 的可用 pipe 数，**Zn3LSU** 全局限制总 L/S uop 数。三者配合精确建模了 "3条 pipe，最多2条做store" 的硬件约束。

## 关键机制解析（工业视角）

### 调度对寄存器分配的影响

Pre-RA 和 Post-RA 调度服务于不同目的：

| 调度阶段 | 目的 | 对寄存器分配的影响 |
|---------|------|------------------|
| **Pre-RA** | 最大化 ILP / 隐藏延迟 | 增加寄存器压力（更多同时活跃的值） |
| **Post-RA** | 微调已分配寄存器的时序 | 不影响寄存器分配（已固定） |

```
经典权衡：
  高 ILP 调度 → 高寄存器压力 → 更多 spill → 性能下降
  低寄存器压力调度 → 少 spill → 但可能有更多 stall → 性能下降
  
最优解取决于目标硬件：
  - 有大量寄存器的 GPU（如 A100 有 65536 个寄存器/SM）→ 偏向 ILP
  - 寄存器稀缺的嵌入式 CPU → 偏向低寄存器压力
```

### 调度区域的划分

基本块被切分为多个调度区域（regions），每个区域独立调度。由 `TargetInstrInfo::isSchedulingBoundary` 控制边界：

```cpp
bool H2BLBInstrInfo::isSchedulingBoundary(const MachineInstr *MI,
                                           const MachineBasicBlock *MBB,
                                           const MachineFunction &MF) const {
  // 自定义边界条件：call 指令、barrier 指令等
  return MI->isCall() || MI->isBarrier();
}
```

**注意：** LLVM 中调度区域不能跨基本块（无 super block / hyper block 支持）。

### 调度模型的完全性

```tablegen
def H2BLBDefaultModel : SchedMachineModel {
  let IssueWidth = 2;       // 每周期 2 条指令
  let LoadLatency = 4;      // load 默认 4 cycles
  let CompleteModel = 1;    // ← 标记为 complete
  // ... 所有指令都必须覆盖
};
```

当 `CompleteModel = 1` 时，TableGen 会检查是否有指令没有被调度模型覆盖。这类似于 Rust 的 exhaustive match——防止新增指令时忘记更新调度模型。

## AI 编译器关联

### GPU Warp 调度 vs LLVM 指令调度

GPU 的调度发生在两个层面：

```
┌──────────────────────────────────────────────┐
│  Level 1: Warp Scheduler (硬件)               │
│  - NVIDIA GPU 在每个 SM 上有多个 warp scheduler │
│  - 每个 cycle 选择一个就绪的 warp 执行指令      │
│  - 通过 warp 切换隐藏内存延迟                   │
│  - 这是硬件调度，LLVM 不可见                    │
├──────────────────────────────────────────────┤
│  Level 2: Instruction Scheduling (编译器)      │
│  - LLVM MachineScheduler 可为 GPU kernel 做    │
│    指令级重排序                                 │
│  - 目标：避免寄存器 bank conflict               │
│  - 目标：最大化双发射利用率                      │
│  - AMDGPU 和 NVPTX 后端都使用 MachineScheduler  │
└──────────────────────────────────────────────┘
```

**GPU 特有的调度考量：**

1. **Register Bank Conflicts**：
   - NVIDIA GPU 的寄存器文件被分为多个 bank
   - 同时访问同一个 bank 的不同寄存器会导致 bank conflict
   - 调度可以重排指令来减少 conflict

2. **Dual Issue**：
   - 现代 GPU 支持每周期同时发射一定类型的指令对（如 FP32 + INT32）
   - 调度可以配对兼容的指令

3. **Scoreboard Stalls**：
   - 长延迟操作（如 global memory load，~300-800 cycles）需要在发射后等待
   - 调度可以提前发射 load 并在等待期间执行独立计算

### AI 加速器指令调度（TPU、NPU）

AI 加速器的调度通常比 CPU/GPU 更复杂：

```
┌──────────────────────────────────────────────────────┐
│  TPU-like Accelerator Scheduling                     │
│                                                      │
│  Systolic Array (MXU):                                │
│    - 流水线深度 8-16 stages                           │
│    - 每个 cycle 读入一行/列数据                         │
│    - 需要精确控制数据到达的时序                          │
│                                                      │
│  Vector Unit:                                        │
│    - 支持多 bank 的 vector registers                  │
│    - 需要管理 bank conflicts                          │
│                                                      │
│  Scalar Unit:                                        │
│    - 处理控制流和地址计算                              │
│    - 需要与 vector/MXU 同步                            │
│                                                      │
│  DMA Controller:                                     │
│    - 异步数据搬移（HBM → VMEM → registers）             │
│    - 需要 double buffering + pipeline scheduling      │
└──────────────────────────────────────────────────────┘
```

**AI 加速器调度模型的设计思路（参考 LLVM SchedMachineModel）：**

```tablegen
def TPUModel : SchedMachineModel {
  let IssueWidth = 8;        // VLIW-like: 每周期 8 个 slot
  let LoadLatency = 100;     // HBM access ~100 cycles
  let CompleteModel = 1;

  // 定义处理单元
  def MXUUnit  : ProcResource<2> { let BufferSize = 8; }  // 2 个 MXU，各 8 级流水
  def VPUUnit  : ProcResource<1> { let BufferSize = 4; }  // 向量单元
  def SPUUnit  : ProcResource<1>;                          // 标量单元
  def DMAUnit  : ProcResource<1> { let BufferSize = -1; } // 异步 DMA
}
```

### MLIR 调度在 IREE 中的实践

IREE 的调度范式与 LLVM 不同：

| 维度 | LLVM Scheduling | IREE Scheduling |
|------|----------------|-----------------|
| **调度粒度** | 指令级别（MachineInstr） | 操作级别（dispatch region/tile） |
| **调度范围** | 基本块内 | 全程序（loop nests + fusion） |
| **目标** | ILP / 延迟隐藏 | Data locality / tile size / 并行度 |
| **实现** | MachineScheduler pass + TableGen model | MLIR pass pipeline + custom heuristics |
| **硬件信息** | SchedMachineModel（延迟、资源） | Target-specific cost model + tile size config |

**IREE 的调度决策示例：**

```mlir
// 调度前：未 tiled 的 matmul
%result = linalg.matmul ins(%lhs, %rhs : tensor<1024x1024xf32>)
                        outs(%init : tensor<1024x1024xf32>)

// 调度后：tiled + vectorized + parallel
// Step 1: Tile to 64x64x64 → 4-level loop nest
// Step 2: Vectorize innermost tile → 8x8 vectors
// Step 3: Parallelize outer loops → workgroup dispatch
// Step 4: Assign to hardware compute units
```

### Triton 的调度策略：依赖 ptxas

**Triton 完全不在编译器中做指令级调度。** 其理由和后果：

```
Triton → MLIR → LLVM IR (nvvm dialect) → PTX → ptxas → cubin
                                                    ↑
                                            所有调度在此发生
```

**为什么 Triton 不做调度：**

1. **PTX 是 virtual ISA**：PTX 指令不直接映射为 SASS 指令，因此调度 PTX 是无意义的
2. **ptxas 最了解硬件**：NVIDIA 的 ptxas 拥有精确的微架构知识（pipeline depth、register bank、dual issue 规则）
3. **避免重复劳动**：任何编译器层面的调度都可能被 ptxas 推翻

**Triton 的隐含调度策略（通过代码结构实现）：**

```python
# Triton kernel: 通过循环结构实现"软件流水线"
@triton.jit
def matmul_kernel(A, B, C, ...):
    # Prefetch: 提前加载数据（隐藏延迟）
    a = tl.load(A_ptr + offsets)     # load 1
    b = tl.load(B_ptr + offsets)     # load 2

    for k in range(0, K, BLOCK_K):
        # 计算上一个 tile + 预取下一个 tile（double buffering）
        acc += tl.dot(a, b)           # 使用上次加载的数据
        a = tl.load(A_ptr + offsets)  # 预取下一个 tile
        b = tl.load(B_ptr + offsets)

    tl.store(C_ptr + offsets, acc)
```

这种手动构造的软件流水线本质上是**在 source 级别做调度**。

## 示例说明

### 示例 1：DDG Mutation 效果

```
原始指令序列：
  %0 = LOAD [addr1]       // latency: 4 cycles
  %1 = ADD %0, 1          // 依赖 %0
  %2 = LOAD [addr2]       // 独立
  %3 = ADD %2, 2          // 依赖 %2

原始 DDG（无 mutation）：
  LOAD1 → ADD1（4-cycle edge）
  LOAD2 → ADD2（4-cycle edge）
  调度器可能：LOAD1 → LOAD2 → ADD1 → ADD2（交错隐藏延迟）

应用 LoadCluster mutation 后：
  LOAD1 → LOAD2 → ADD1 → ADD2（强制 grouping）
  调度器必须：LOAD1 → LOAD2 → ADD1 → ADD2
  好处：更好的 cache locality（连续访问）
  代价：ADD1 等待时间更长
```

### 示例 2：Scheduling Model 的延迟建模

```tablegen
// 指令：FMA Rdest, Rsrc1, Rsrc2, Racc
// 行为：Rdest = Rsrc1 * Rsrc2 + Racc

let SchedModel = TPUModel in {
  // FMA 指令的调度描述
  let Latency = 4 in                    // 总延迟 4 cycles
  def WriteFMA : SchedWriteRes<[MXUUnit]> {
    let NumMicroOps = 1;                // 1 个微操作
    let ResourceCycles = [4];           // 占用 MXU 4 个 cycle（流水化）
  }

  // 操作数读取：前一个 FMA 的结果有 2-cycle bypass
  def ReadFMAArg0 : SchedReadAdvance<0>;  // Rsrc1: 无特殊转发
  def ReadFMAArg1 : SchedReadAdvance<1>;  // Rsrc2: 1-cycle 转发
  def ReadFMAAcc  : SchedReadAdvance<2>;  // Racc: 2-cycle 转发

  // 绑定到指令
  def : InstRW<[WriteFMA, ReadFMAArg0, ReadFMAArg1, ReadFMAAcc],
               (instregex "^FMA")>;
}

// 效果：
// FMA1: write at cycle 4
// FMA2 that uses FMA1's result as Racc: can start at cycle 2 (bypass)
// 而不是 cycle 4 (full latency) — 节省 2 cycles
```

### 示例 3：GPU Kernel 调度分析

```
CUDA kernel 的调度考量：

原始 PTX:
  ld.global.f32 %r1, [%rd1];      // latency ~300 cycles
  ld.global.f32 %r2, [%rd2];      // latency ~300 cycles
  add.f32 %r3, %r1, %r2;          // stall until r1, r2 ready
  st.global.f32 [%rd3], %r3;

最佳调度（LLVM MachineScheduler 可以做到）：
  ld.global.f32 %r1, [%rd1];      // 发射指令 1
  ld.global.f32 %r2, [%rd2];      // 发射指令 2 (不等待指令 1)
  // ... 插入 ~300 cycles 的独立指令（其他 warp 的计算）...
  add.f32 %r3, %r1, %r2;          // 此时 r1, r2 已就绪
  st.global.f32 [%rd3], %r3;

实际效果：
  不调度：ADD 等待 300 cycles + 调度器切换 warp（由硬件完成）
  调度后：编译器层面最大化 ILP，让硬件调度器有更多就绪 warp
```

## 总结

### 核心要点

1. **调度是可选但重要的优化**：对于顺序处理器（如某些 DSP/AI 加速器）至关重要，对于乱序处理器也有价值
2. **三个定制轴**：DDG Mutation（约束修改）、SchedStrategy（算法定制）、SchedModel（硬件描述）
3. **InstRW 优于 WriteRes/ReadAdvance**：更好的局部性和可维护性
4. **GPU 调度是分层的**：编译器做指令级调度（LLVM），硬件做 warp 级调度（NVIDIA GPU），Triton 依赖 ptxas 做最终调度

### AI 编译器工程师的关键理解

| 概念 | LLVM 实践 | AI 编译器实践 |
|------|----------|-------------|
| 调度粒度 | MachineInstr（指令级） | Operation/tile（操作级） in MLIR |
| 硬件描述 | SchedMachineModel (TableGen) | Target cost model + tile config |
| 调度算法 | MachineSchedStrategy (GenericScheduler) | Custom MLIR pass heuristics |
| 约束修改 | DDG Mutations | MLIR pass interleaving |
| 延迟隐藏 | Scheduler + ReadAdvance | Double buffering + SW pipelining |
| GPU 调度 | MachineScheduler (AMDGPU/NVPTX) | ptxas (Triton) / custom scheduler |

### 进阶话题

- **VLIW 处理器和 itineraries**：VLIW 处理器（如一些 DSP）需要 itineraries-based 调度模型，比 scheduling events 更精确但也更复杂
- **llvm-exegesis 工具**：可以自动测量指令延迟和微架构特性，辅助构建调度模型
- **调度与寄存器分配的迭代**：理论上可以迭代 pre-RA 调度和寄存器分配（LLVM 目前不支持），这在寄存器稀缺的架构上可能显著提升性能
- **MLIR 的调度 IR**：IREE 等正在探索在 MLIR 中表达调度决策（通过 `scf.for` + `affine` dialect），这是 AI 编译器领域的活跃研究方向
