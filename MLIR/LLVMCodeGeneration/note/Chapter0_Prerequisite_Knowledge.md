# Chapter 0: 前置知识 —— 计算机体系结构与编译基础

> 在学习 LLVM 后端开发和 AI 编译器之前，你需要掌握这些基础知识。本章不是要替代任何教科书，而是提供一个 **以编译器开发者视角** 组织的知识地图——帮你快速定位哪些知识是必须的、哪些可以边学边补。

---

## 核心概念（知识地图）

```
学习 LLVM Code Generation 的预备知识体系：

┌─────────────────────────────────────────────────────┐
│                    Level 3: AI 编译器                  │
│  MLIR / Triton / IREE / XLA / TVM                    │
├─────────────────────────────────────────────────────┤
│      Level 2: LLVM 后端开发（本书内容）               │
│  调度 / 寄存器分配 / 指令选择 / Stack Lowering       │
├─────────────────────────────────────────────────────┤
│   Level 1: 前置知识（本章内容）                       │
│  ┌─────────────────────────────────────────────┐    │
│  │ 计算机体系结构 │ 汇编语言 │ 编译原理基础    │    │
│  │ C/C++ 基础     │ 数据结构 │ 算法基础        │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

---

## 一、计算机体系结构（必须掌握）

### 1.1 ISA（指令集架构）基础

**ISA** 是软硬件接口——编译器输出的是针对特定 ISA 的机器码。

| ISA 类型       | 代表                     | 对编译器的挑战                        |
| -------------- | ------------------------ | ----------------------------------- |
| **RISC**       | ARM AArch64, RISC-V, MIPS | 规整、易调度                          |
| **CISC**       | x86, x86-64               | 复杂指令（带副作用），需要 uop 分解    |
| **VLIW**       | Hexagon, 一些 DSP         | 编译器负责并行调度，硬件不做重排      |
| **GPU ISA**    | PTX (NVIDIA), GCN (AMD)  | SIMT 模型，大量寄存器，显式内存层次    |
| **AI 加速器** | TPU/NPU 自定义 ISA        | 矩阵乘专用指令，数据流编程模型         |

**编译器开发者视角的核心概念**：

- **寻址模式（Addressing Mode）**：`[base + offset]`、`[base + index * scale + offset]`。LLVM 通过 `getelementptr`（GEP）统一处理，后端负责映射到具体寻址模式。
- **调用约定（Calling Convention）**：参数如何传递（寄存器/栈）、返回值放哪里、谁清理栈。LLVM 中通过 `CCState`、`CCValAssign` 建模。**这对 AI 编译器尤其重要**——GPU kernel 的 ABI 与 CPU 函数完全不同。
- **编码格式（Instruction Encoding）**：每条指令如何编码为 bit 序列。LLVM 后端通过 TableGen 描述，最终交给 MC（Machine Code）层处理。固定长度（RISC）vs 变长（CISC）编码对指令选择有根本影响。

### 1.2 CPU 微架构

理解微架构才能写出有效的**调度模型（Scheduling Model）**——这是 LLVM 后端开发的核心任务之一。

#### 流水线（Pipeline）

```
┌──────┐  ┌───────┐  ┌───────┐  ┌────────┐  ┌──────────┐
│ IF   │→│ ID/RF  │→│ Issue │→│  EXE   │→│ Writeback│
│取指  │  │译码/取 │  │发射   │  │ 执行   │  │  写回    │
│      │  │寄存器  │  │       │  │        │  │          │
└──────┘  └───────┘  └───────┘  └────────┘  └──────────┘
```

**编译器需要关心的流水线问题**：

| 问题                          | 名称         | 编译器如何应对                                  |
| ----------------------------- | ------------ | ---------------------------------------------- |
| 一条指令的结果下条就要用      | RAW hazard   | 调度器拉远 producer/consumer 距离 + ReadAdvance |
| 连续多条指令用同一个功能单元  | Structural   | ProcResource 计数 + 调度器分散负载              |
| 分支跳转后不知道该取哪条指令  | Branch stall | 分支预测提示 + 基本块重排（block placement）     |
| 除法/memory load 需要很多周期 | 长延迟操作   | 调度器将无关指令插入延迟槽                     |

#### 乱序执行（Out-of-Order Execution）

乱序 CPU 的核心组件（注：这些概念直接影响 LLVM 调度模型的设计）：

```
┌──────────────────────────────────────────────┐
│              乱序执行核心（OOO Core）           │
│                                               │
│  ┌──────────┐   ┌───────────────┐            │
│  │ 保留站   │   │ Reorder Buffer│             │
│  │(Reservation│  │    (ROB)      │            │
│  │ Station)  │   │  重排序缓冲区   │            │
│  └──────────┘   └───────────────┘            │
│  存放等待操作数的指令  保证指令按序提交       │
│                                               │
│  关键参数：                                    │
│  - Issue Width：每周期最多发射几条指令         │
│  - ROB Size：最多同时"飞行中"的指令数         │
│  - 物理寄存器数量：SchedModel 的 MicroOpBufferSize │
│  实际 = min(ROB, 重命名寄存器池, 保留站容量)    │
└──────────────────────────────────────────────┘
```

**与编译器调度的关系**（重要！）：

- 编译器的调度模型（`ProcResource`, `BufferSize`）本质上是对乱序硬件的一种**粗粒度抽象**
- `BufferSize > 0` → 告诉调度器这个资源有缓冲，可以接收更多指令
- `BufferSize = -1`（默认）→ 统一保留站，用 `MicroOpBufferSize` 全局限制
- 编译器做**静态粗调度**（pre-RA/post-RA scheduling），硬件做**动态细调度**
- 二者协同：编译器把高延迟指令提前、有依赖的指令拉远 → 减少硬件调度器压力 → 减少 stall

#### 超标量（Superscalar） vs 乱序（OOO）

**它们是正交概念**：
- 超标量：每周期发射多条指令（取决于 issue width）
- 乱序：指令可以不按程序顺序执行（取决于 ROB/保留站）
- 可以有"顺序+超标量"（很多嵌入式 CPU/GPU）
- 理论上也可以"乱序+非超标量"（没人这么做）

### 1.3 GPU 架构基础（AI 编译器必须了解）

**GPU 与传统 CPU 的关键区别**：

| 特性               | CPU                           | GPU (NVIDIA)                          |
| ------------------ | ----------------------------- | ------------------------------------ |
| **执行模型**       | 多核线程并行（MIMD）          | SIMT：同一指令多个线程同时执行       |
| **Warp/Wavefront** | 无                            | 32 线程一组（NVIDIA warp），lockstep |
| **寄存器文件**     | ~32 通用寄存器/核             | 65536 寄存器/SM (A100)               |
| **调度策略**       | 乱序执行                      | Warp 级 in-order，warp 间乱序切换    |
| **延迟隐藏**       | OOO + 分支预测 + cache prefetch | Warp 切换（零开销上下文切换）         |
| **内存层次**       | L1/L2/L3 cache                | Shared Memory + L1 + L2 + Global     |

**为什么这对 LLVM 调度模型很重要**：

- GPU 每个 SM 的指令调度器是 in-order 的（warp 级别），但 warp 之间是乱序切换的
- Triton 生成的 PTX → 最终由 `ptxas`（NVIDIA 的 PTX 汇编器）调度
- GPU 寄存器分配极其关键：spill 到 local memory 的代价是 **300-800 cycles**（vs 1-4 cycles 的寄存器访问）
- Shared memory（L1 可编程缓存）可以被编译器显式管理——MLIR 的 `gpu.shared` 和 `memref` dialect 对此建模

---

## 二、汇编语言基础（需要熟练阅读）

你不需要手写汇编（虽然偶尔会手工写 .mir 测试），但你**必须能读懂**以下内容：

### 2.1 基础汇编结构

```asm
# x86-64 AT&T 语法
movq    %rsp, %rbp          # 源,目标 顺序（AT&T）
addq    $8, %rsp            # $ 表示立即数

# AArch64
add     x0, x1, x2          # x0 = x1 + x2
ldr     x0, [x1, #8]        # x0 = *(x1 + 8)
str     x0, [sp, #-16]!     # sp -= 16; *(sp) = x0 (pre-index)

# RISC-V
add     a0, a1, a2          # a0 = a1 + a2
lw      a0, 0(a1)           # a0 = *(a1 + 0)
```

### 2.2 与 LLVM 后端的映射

| 汇编概念           | LLVM 后端对应                            |
| ------------------ | ---------------------------------------- |
| **指令操作码**     | `MachineInstr::getOpcode()`              |
| **寄存器**         | `MCRegister` → 通过 `MCRegisterInfo` 查询|
| **立即数**         | `MachineOperand::isImm()`                |
| **内存寻址**       | `MachineOperand` + `TargetInstrInfo` 解析|
| **Label/符号**     | `MCSymbol` + `MCExpr`                    |
| **directives**     | `MCStreamer` 控制（`.text`, `.data` 等） |

### 2.3 PTX 汇编（AI 编译器特需）

Triton 生成 PTX（NVIDIA 的中间汇编），你至少要能读懂：

```ptx
// PTX: NVIDIA 的虚拟 ISA（JIT 编译到 SASS）
ld.global.ca.f32   %f1, [%rd1];    // 从 global memory load（cache at all levels）
st.shared.f32      [%rd2], %f2;    // store 到 shared memory
bar.sync           0;               // 同步 barrier
// Tensor Core 指令
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    {%f1, %f2, %f3, %f4}, {%f5, %f6}, {%f7, %f8}, {%f1, %f2, %f3, %f4};
```

**LLVM 如何生成 PTX**：NVPTX 后端将 MachineInstr 序列化为 PTX 文本 → 通过 `MCStreamer` 输出 → NVIDIA 驱动 JIT 编译为 SASS。

---

## 三、编译原理基础（核心概念速查）

### 3.1 编译流程概览

```
源代码 (.c/.cpp)
    │
    ▼
┌──────────┐
│ Frontend │  词法分析 → 语法分析 → 语义分析 → 生成 IR
│ (Clang)  │  你可能需要了解但不需要深入实现
└──────────┘
    │ LLVM IR (.ll / .bc)
    ▼
┌──────────┐
│ Middle-  │  优化：instcombine, mem2reg, inlining, loop opts...
│ End      │  目标无关的 pass pipeline
└──────────┘
    │ LLVM IR (优化后)
    ▼
┌──────────┐
│ Backend  │  指令选择 → 调度 → 寄存器分配 → 汇编生成
│ (本书重点)│  目标相关的 lowering
└──────────┘
    │ 汇编/目标文件 (.s / .o)
    ▼
  可执行文件
```

### 3.2 核心概念定义

整理书中反复出现的关键术语（建议熟记）：

| 术语                     | 定义                                                                   | AI 编译器对应                   |
| ------------------------ | ---------------------------------------------------------------------- | ------------------------------ |
| **IR（中间表示）**       | 编译器内部表示程序的数据结构，介于源码和目标代码之间                   | MLIR 各种 Dialect              |
| **Lowering（降级）**     | 将高层抽象的 IR 逐步转换为更接近目标硬件的低层 IR                     | MLIR Dialect Conversion        |
| **Pass**                 | 对 IR 执行一次遍历/变换的独立单元                                      | MLIR Pass                      |
| **Pass Pipeline**        | 按特定顺序执行的 Pass 序列                                             | IREE/Triton 的编译 Pipeline    |
| **SSA（静态单赋值）**    | 每个变量在 IR 中只被赋值一次，通过 phi 节点合并不同路径的值           | MLIR 使用 block-argument 风格  |
| **Canonical Form**       | IR 的标准表达形式（如常量放右边），便于优化器识别模式                  | MLIR Canonicalization          |
| **Dominance**            | 节点 d 支配节点 n：所有到达 n 的路径必经 d                             | SSA 的 use-def 合法性基础      |
| **Legalization**         | 将 IR 中目标硬件不直接支持的操作展开为支持的等价序列                   | MLIR 的 Dialect Conversion     |
| **ABI**                  | 函数间调用的二进制接口约定（参数传递、返回值、栈布局）                 | GPU kernel ABI、Triton ABI    |

### 3.3 数据结构备忘

| 数据结构                   | 用途                         | LLVM 类                              |
| -------------------------- | ---------------------------- | ------------------------------------ |
| **DAG（有向无环图）**      | 指令选择中的模式匹配、数据流 | `SelectionDAG`、`ScheduleDAG`        |
| **CFG（控制流图）**        | 函数内部控制流分析           | `MachineBasicBlock` 的 succ/pred     |
| **Dominator Tree**         | SSA 属性验证、代码移动       | `DominatorTree`、`MachineDominatorTree` |
| **Call Graph**             | 跨函数优化（IPO）            | `CallGraph`                          |
| **Live Interval**          | 寄存器分配                   | `LiveInterval`、`LiveRange`          |

---

## 四、C/C++ 基础（与 LLVM 开发相关）

### 4.1 你需要熟悉的 C++ 特性

LLVM 代码库大量使用现代 C++（但不激进），以下是高频出现的特性：

| 特性                       | LLVM 中的用途                                               |
| -------------------------- | ----------------------------------------------------------- |
| **继承与虚函数（CRTP）**   | Pass 框架（`PassInfoMixin`）、`TargetLowering` 等基类       |
| **模板元编程**             | `AnalysisInfoMixin<Derived>`、`PassInfoMixin<Derived>`      |
| **智能指针**               | `std::unique_ptr<Module>`、`std::shared_ptr<>`              |
| **LLVM RTTI**              | `isa<>`、`cast<>`、`dyn_cast<>`（替代标准 RTTI，更高效）    |
| **LLVM 容器类**            | `SmallVector<>`、`SmallPtrSet<>`、`DenseMap<>`、`StringRef` |
| **Lambda + std::function** | Pass 注册回调、pipeline 构造                                |
| **迭代器模式**             | 遍历 Module/Function/BasicBlock/Instruction                 |

### 4.2 CMake 基础

LLVM 使用 CMake，你需要能：

- 编写基础 `CMakeLists.txt`：`add_executable`、`target_link_libraries`
- 配置 LLVM 构建：`-DLLVM_ENABLE_PROJECTS="clang;mlir"`、`-DLLVM_TARGETS_TO_BUILD="X86;NVPTX"`
- 理解 LLVM 的 `llvm_map_components_to_libnames` 宏
- `find_package(LLVM)` 模式

### 4.3 开发工具链

| 工具      | 用途                                      |
| --------- | ----------------------------------------- |
| **git**   | 版本控制，理解 GitHub PR 流程             |
| **CMake** | 构建系统                                  |
| **Ninja** | 快速增量构建（LLVM 标准构建驱动）         |
| **gdb/lldb** | 调试 LLVM pass、分析 crash              |
| **lit**   | LLVM 集成测试框架（本书 Ch1 详细讲解）    |
| **FileCheck** | 测试输出模式匹配（与 lit 配合使用）   |
| **TableGen** | LLVM DSL，描述目标信息（本书 Ch6）     |

---

## 五、学习路线建议

### 5.1 如果你完全零基础

```
第 1 周: 计算机体系结构基础（流水线、ISA、寄存器）
    → 推荐: "Computer Organization and Design" (Patterson & Hennessy)
    → 重点章节: Chapter 4 (The Processor), Chapter 5 (Memory Hierarchy)

第 2 周: 汇编基础（x86-64 或 AArch64，顺手学 PTX）
    → 不需要精通，能读懂即可
    → 推荐: 在线 Compiler Explorer (godbolt.org) 交互学习

第 3 周: 编译原理基础（IR、Pass、SSA、支配树）
    → 推荐: "Engineering a Compiler" (Cooper & Torczon)
    → 重点章节: Chapters 5 (IR), 8-10 (Optimization)

第 4+ 周: 开始本书 LLVM Code Generation
    → 从 Chapter 3 (Compiler Basics & LLVM APIs) 开始
    → 配合 LLVM 官方 Kaleidoscope 教程 (https://llvm.org/docs/tutorial/)
```

### 5.2 如果你已有编译基础（但无 LLVM 经验）

```
第 1 周: LLVM 项目构建 + 熟悉代码库结构（本书 Ch1-2）
    → 拉取 llvm-project，构建带 clang 的开发版本
    → 用 lit 跑测试，熟悉 FileCheck

第 2 周: LLVM IR 和 Pass Manager（本书 Ch3-5, Ch7-8）
    → 写第一个 FunctionPass
    → 理解 Legacy PM vs New PM 的区别

第 3 周: Machine IR 和 TableGen（本书 Ch6, Ch11-12）
    → 理解 .mir 文件格式
    → 写寄存器和指令的 TableGen 描述

第 4+ 周: 指令选择、调度、寄存器分配（本书 Ch14-21）
    → 这是 LLVM 后端的核心，章节多、最耗时
```

### 5.3 如果你主攻 AI 编译器

```
关键技能栈：
1. LLVM 后端基础（本书全部）      → AI 编译器底层都是 LLVM
2. MLIR Dialect 设计               → 在前者基础上扩展
3. GPU 架构与 PTX                  → 本书 Ch18-21 的调度/寄存器分配知识直接适用
4. Triton 编译器源码阅读           → Triton 本质上是一个 LLVM IR 生成器 + 自定义 pass
5. XLA/IREE 编译流程               → 理解 HLO → LHLO → Linalg → LLVM 的全链路

推荐路线:
    LLVM 基础 (本书 Ch1-8)
        → Machine IR (本书 Ch11-13)
            → 指令选择与调度 (本书 Ch14-18)
                → 寄存器分配与汇编 (本书 Ch19-21)
                    → MLIR 教程 (mlir.llvm.org)
                        → Triton 源码
                            → IREE/XLA 源码
```

---

## 六、推荐资源

### 必读书目

| 书名                                           | 用途                               |
| ---------------------------------------------- | ---------------------------------- |
| **Computer Organization and Design** (P&H)     | 体系结构入门圣经，重点读 Chapter 4 |
| **Engineering a Compiler** (Cooper & Torczon)  | 编译原理最工程化的教材             |
| **LLVM Code Generation** (本书)               | LLVM 后端开发的权威参考            |
| **Getting Started with LLVM Core Libraries**   | LLVM 快速入门                     |

### 在线工具

| 工具                                  | 用途                                          |
| ------------------------------------- | --------------------------------------------- |
| **Compiler Explorer** (godbolt.org)   | 对照 C/C++ 源码与各架构汇编                   |
| **LLVM Doxygen** (llvm.org/doxygen)   | LLVM C++ API 在线文档                         |
| **LLVM Command Guide**               | opt/llc/llvm-mc 等工具用法                     |
| **MLIR 官网** (mlir.llvm.org)         | MLIR Tutorial + 各 Dialect 文档               |

### 必读代码

| 代码路径                                    | 学习目的                         |
| ------------------------------------------- | -------------------------------- |
| `llvm/lib/IR/`                              | LLVM IR 核心实现                 |
| `llvm/lib/CodeGen/`                         | Machine IR 和后端通用实现        |
| `llvm/lib/Target/X86/X86ScheduleZnver4.td`  | 学习工业级调度模型（AMD Zen4）   |
| `llvm/lib/Target/AArch64/AArch64RegisterInfo.td` | 学习寄存器描述               |
| `third_party/triton/lib/Conversion/`        | Triton 的 MLIR 转换 pass        |
| `third_party/iree/compiler/src/iree/compiler/` | IREE 的编译 pipeline          |

---

## 总结

### 核心要点

1. **体系结构是编译器开发的"上下文"**：不理解 ISA 和微架构，就无法评估优化效果、无法写出正确的调度模型
2. **汇编语言是编译器的"输出格式"**：不需要手写，但要能读懂，能调试后端生成结果
3. **编译原理中的 SSA/支配/CFG 是 LLVM 的"通用语言"**：MLIR/Triton/IREE 都在用这些概念
4. **C++/CMake 是 LLVM 开发的"交流工具"**：CRTP、LLVM RTTI、SmallVector 等是高频使用的 idiom
5. **GPU 架构知识是 AI 编译器的"独特要求"**：SIMT、warp、shared memory、PTX 是绕不开的

### 学习建议

- **不要试图一次学完所有前置知识再开始本书**——边学边补效率最高
- **Compiler Explorer (godbolt.org) 是最好的学习工具**——写一小段 C，看不同优化级别的汇编输出
- **LLVM 源码是最好的文档**——Doxygen + 源代码搜索 = 最准确的 API 使用方式
- **写代码比读代码重要 10 倍**——读完每章后立即动手写一个对应的 pass
- **对于 AI 编译器方向**：本书学完后，MLIR 的 `gpu`/`nvvm`/`llvm` dialect 是你需要重点研究的下一站

> **下一步**：从 Chapter 1 开始，学习如何构建 LLVM 和理解其目录结构。
