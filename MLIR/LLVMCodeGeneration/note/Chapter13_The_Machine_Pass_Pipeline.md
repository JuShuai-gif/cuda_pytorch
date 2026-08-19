# Chapter 13: The Machine Pass Pipeline

## 核心概念（详细展开）

### Machine Pass Pipeline 的架构设计

Machine pass pipeline 是 LLVM 后端中 Machine IR 经历的顺序处理流程。它的设计围绕着
**SSA → 非 SSA** 这一核心转换，分为明确的阶段：

```
┌────────────────────────────────────────────────┐
│ Phase 1: INSTRUCTION SELECTION (第14-17章)      │
│   LLVM IR → SelectionDAG / G_MIR               │
│   Legalization → Target-specific MachineInstr   │
│   完成时: Machine IR 处于 SSA 形式              │
├────────────────────────────────────────────────┤
│ Phase 2: SSA-BASED MACHINE OPTIMIZATIONS        │
│   - PHI elimination (SSA → 非 SSA 边界)        │
│   - MachineCSE, MachineLICM, MachineSink        │
│   - PeepholeOptimizer, MachineCombiner          │
│   完成时: 非 SSA Machine IR (COPY 替代 PHI)     │
├────────────────────────────────────────────────┤
│ Phase 3: REGISTER ALLOCATION                    │
│   - Live Interval Analysis                      │
│   - Coalescing (消除不必要的 COPY)              │
│   - Spill/Reload 插入                           │
│   完成时: 只有物理寄存器，没有虚拟寄存器         │
├────────────────────────────────────────────────┤
│ Phase 4: POST-RA OPTIMIZATIONS                  │
│   - Post-RA Machine LICM / CSE                  │
│   - Branch folding, tail duplication            │
│   - If-conversion                               │
├────────────────────────────────────────────────┤
│ Phase 5: CODE EMISSION                          │
│   - Prologue/Epilogue Insertion                 │
│   - Frame lowering (栈帧最终布局)               │
│   - Branch relaxation (长跳转展开)              │
│   - MCInstLower → MC 层面输出                   │
└────────────────────────────────────────────────┘
```

**IR 在三个阶段的实际变迁**：

| 阶段 | 示例 IR | 关键特性 |
|------|---------|---------|
| Early (ISel 后) | `%1:gpr64sp = PHI ...` `%8:gpr64 = ADDXri %1, 1` | SSA, 虚拟寄存器, PHI 指令存在 |
| Pre-RA (PHI消除后) | `%1:gpr64sp = COPY ...` `%8:gpr64 = ADDXri killed %1, 1` | 非 SSA, COPY 替代 PHI, kill flags |
| Post-RA | `renamable $x8 = ADDXri killed renamable $x8, 1` | 纯物理寄存器, renamable flags |

### TargetPassConfig 的核心作用

`TargetPassConfig` 是控制整个 Machine pass pipeline 的中心类：

```cpp
class MyTargetPassConfig : public TargetPassConfig {
public:
    MyTargetPassConfig(MyTargetTargetMachine &TM, PassManagerBase &PM);

    // === 注入点 (Injection Points) ===

    // ISel 之前：目标特定的 LLVM IR 预处理
    bool addPreISel() override;

    // 指令选择器
    bool addInstSelector() override;

    // 代码生成中的 IR-level passes
    bool addIRPasses() override;

    // 寄存器分配之前：SSA 优化
    bool addPreRegAlloc() override;

    // 寄存器分配器选择
    bool addOptimizedRegAlloc() override;

    // 寄存器分配之后
    bool addPostRegAlloc() override;

    // 代码发射之前：第一轮
    bool addPreEmitPass() override;

    // 代码发射之前：第二轮（最后机会）
    bool addPreEmitPass2() override;
};
```

**标准注入模式**：

```cpp
// 在 SSA 阶段注入自定义优化
bool MyTargetPassConfig::addPreRegAlloc() {
    // 先调用基类的标准实现！
    if (!TargetPassConfig::addPreRegAlloc())
        return false;

    // 在标准 SSA passes 之后注入自定义 pass
    addPass(createMyTargetSpecificOptimization());
    return true;
}

// 在 Post-RA 阶段注入 cleanup passes
bool MyTargetPassConfig::addPreEmitPass() {
    addPass(createMyTargetExpandPseudoInsts());
    addPass(createMyTargetBranchRelaxation());
    return true;
}
```

**关键原则**：
1. **永远调用基类方法**：`addPreRegAlloc()` 的第一行必须是 `TargetPassConfig::addPreRegAlloc()`
2. **理解 SSA 属性的转换点**：PHI elimination 是不可逆的 SSA→非SSA 转换
3. **Post-RA 不允许创建虚拟寄存器**：所有的 `createVirtualRegister` 调用必须在 RA 前

## LLVM / MLIR 流程（深入）

### MLIR Pass Pipeline vs Machine Pass Pipeline

```
MLIR Pipeline (IREE example):
┌──────────────────────────────────────────────┐
│ MLIR Optimization Passes                      │
│  - canonicalize, cse, inlining, linalg-fuse   │
│  - gpu-kernel-outlining                       │
│    ↓ lowering                                 │
│  - convert-linalg-to-loops                    │
│  - convert-scf-to-cf                          │
│  - convert-arith-to-llvm                      │
│  - convert-func-to-llvm                       │
│  - gpu-to-llvm / gpu-to-nvvm                  │
│    ↓ translate (边界)                         │
├──────────────────────────────────────────────┤
│ LLVM IR                                       │
│    ↓ LLVM Pass Pipeline (中端)                 │
│ Optimized LLVM IR                             │
│    ↓ LLVM CodeGen Pipeline (后端) ← 本书焦点  │
│    ↓ Machine Pass Pipeline                    │
│ Assembly / Object File                        │
└──────────────────────────────────────────────┘
```

**对比分析**：

| 特性 | MLIR Pass Pipeline | LLVM Machine Pass Pipeline |
|------|-------------------|--------------------------|
| IR 类型 | MLIR operations (多 dialect) | MachineInstr (单层) |
| Pass 基类 | `OperationPass<>` | `MachineFunctionPass` |
| 分析管理 | `AnalysisManager` | `MachineFunctionAnalysisManager` |
| SSA 保证 | 核心 dialects 都是 SSA | 灵活：SSA → 非 SSA 转换 |
| 定制方式 | 完全可自定义 pipeline | TargetPassConfig hooks |
| Pass 间通信 | Operation 属性 + analyses | analyses + MachineFunctionInfo |

**关键差异**：MLIR 的 pass pipeline 完全由用户编写（显式 pass 列表），而 LLVM Machine pipeline
通过 TargetPassConfig 的虚方法 + 默认实现提供 "模板方法模式" 的定制。

### Triton Backend Optimization Order（实际案例分析）

Triton 编译器在生成 LLVM IR 后调用 LLVM 的 CodeGen pipeline。
Triton 的优化顺序（从高层到低层）：

```
Triton IR 优化 (Triton 自有 pass pipeline):
  1. loop-unroll      (展开 triton loop)
  2. pipeline-pass    (软件流水线化 shared memory 加载)
  3. prefetch-pass    (预取 shared memory)
  4. coalesce-pass    (合并全局内存访问)
  5. combine-pass     (算术指令合并)
     ↓ (Triton IR → LLVM IR translation)
LLVM IR 优化 (标准 LLVM 中端):
  6. -O3 pipeline     (instcombine, gvn, licm, slp-vectorizer...)
     ↓
LLVM Machine Pass Pipeline (Triton 使用 NVPTX 后端):
  7. SDISel: LLVM IR → NVPTX MachineInstr
  8. MachineCSE      (消除冗余 PTX 操作)
  9. PeepholeOptimizer (本地指令折叠)
  10. PrologueEpilogueInserter (NVPTX 无栈帧，此 pass 几乎空转)
  11. NVPTX AsmPrinter → PTX text
```

**Triton 的优化策略启示**：
- 由于 PTX 是虚拟 ISA（最终由 ptxas 编译），大量传统后端优化（寄存器分配、指令调度）
  在 PTX 层面没有意义
- 因此 NVPTX 的 Machine pass pipeline 比 AArch64/x86 简单得多
- Triton 把更多的优化放在了 Triton IR 层面（coalescing、pipelining、combine）
- 这体现了 **"在正确的 IR 层面做正确的优化"** 的原则

### IREE Flow for GPU Optimization

```
IREE 的 GPU 优化全流程:
┌──────────────────────────────────────────────┐
│ IREE Input (MLIR)                             │
│  1. IREE Flow pipeline (fusion, dispatch)     │
│  2. IREE Stream pipeline (async scheduling)   │
│  3. IREE HAL pipeline (device assignment)     │
│    ↓                                          │
│ IREE → LLVM Backend boundary                  │
│  4. convert-to-llvm (mlir → llvm dialect)     │
│  5. translate-to-llvmir (llvm dialect → IR)   │
│  6. LLVM optimization pipeline                │
│  7. LLVM Machine pipeline (NVPTX/AMDGPU)      │
│    ↓                                          │
│ PTX / AMDGPU Assembly                         │
└──────────────────────────────────────────────┘
```

**IREE 的 Machine Pass 使用特点**：
- IREE 生成的是优化的 LLVM IR，所以 Machine pipeline 主要做 "最后的物理化"
- 对于 GPU 后端，Machine pipeline 承担：
  - Shared memory lowering（将 LLVM IR 的 addrspace(3) 映射到 Machine IR 的共享内存操作）
  - GPU-specific 指令选择（如 wmma/mfma 矩阵乘法指令的匹配）
  - 寄存器 bank 分配（GlobalISel 的 RegisterBankSelect pass）

## 关键机制解析（工业视角）

### 调试 Machine Pass Pipeline

```bash
# 查看完整的 pass pipeline 结构
llc -mtriple=aarch64 -debug-pass=Structure input.ll -o /dev/null
# 输出类似：
# Machine InstCombiner
# Machine Early If-Conversion
# Machine Block Placement
# ...

# 在特定 pass 前停止（生成 .mir 文件检查 IR）
llc -stop-before=peephole-opt input.ll -o before_peep.mir

# 运行单个 pass
llc -run-pass=peephole-opt input.mir -o after_peep.mir

# 从特定 pass 开始运行
llc -start-before=peephole-opt input.mir -o out.s

# 启用特定 pass 的 debug log
llc -debug-only=peephole-opt input.ll -o /dev/null
# 寄存器分配的 debug log 组 (regalloc) 包括多个 pass
llc -debug-only=regalloc input.ll -o /dev/null
```

### CodeGenPrepare - LLVM IR 层面的准备

CodeGenPrepare 是一个特殊的 pass：它在 LLVM IR 层面运行，但服务于 Machine IR 的生成。
它的核心任务是**让 LLVM IR 对后端更友好**：

```cpp
// CodeGenPrepare 的主要转换:
// 1. 拆分复杂的 GEP 为单步地址计算
// GEP: %ptr = getelementptr %base, i64 %idx1, i64 %idx2, i32 %idx3
//   → 拆分为多个独立的地址计算

// 2. 下沉地址计算更靠近使用处（减少寄存器压力）
// Before:  %addr = gep %base, %idx  (在入口块定义)
// After:  移动到使用 %addr 的基本块

// 3. 扩展 switch 为 jump table
// switch %val, label %default [
//   i32 0, label %case0
//   i32 1, label %case1
// ] → jump table + indirect branch

// 4. 优化 overflow intrinsic 为后端友好的形式
// @llvm.sadd.with.overflow.i32(%a, %b) → 后端可直接匹配的 add + flags 形式
```

**工业重要性**：CodeGenPrepare 是连接 LLVM IR 优化和 Machine IR 生成的桥梁。对于 AI 编译器：
- 如果后端对某些 pattern 有更好的支持（如专用地址生成硬件），CodeGenPrepare 的 hook 可以注入
  目标特定的 IR 变换
- `TargetLowering` 类提供了一系列虚方法让后端控制 CodeGenPrepare 的行为

### PeepholeOptimizer - 本地窥孔优化

PeepholeOptimizer 在 Machine IR（SSA 和非 SSA 皆可）上操作，扫描小窗口的指令：

```cpp
// 主要变换类型:

// 1. COPY 折叠
// Before:  %a = COPY %b
//          ... = use %a
// After:   ... = use %b    (直接使用源寄存器，消除中间 COPY)

// 2. 符号扩展/零扩展消除
// Before:  %a = SXTB %b    (符号扩展 byte)
//          %c = ADD %a, %d
// After:   %c = ADD_SXTb %b, %d  (融合扩展和加法)

// 3. 比较链优化
// Before:  %a = SUBS %x, %y
//          %b = CSEL ..., implicit $nzcv
// After:   优化为更优的条件选择形式

// 4. Load-Store 前向传播
// Before:  STORE %val, [%addr]
//          %v = LOAD [%addr]
// After:   %v = COPY %val  (load 直接从之前的 store 获取值)
```

**目标 hook**（通过 TargetInstrInfo）：
```cpp
bool optimizeLoadInstr(MachineInstr &MI, ...);  // 优化 load 指令
bool foldImmediate(MachineInstr &UseMI, ...);   // 折叠立即数
bool isCopyInstr(const MachineInstr &MI, ...);  // 是否为 COPY 指令
```

### MachineCombiner - 指令重组

MachineCombiner 在一个基本块内的指令迹（trace）上工作：

```cpp
// 1. 指令重组 (Reassociation) 以利用寻址模式
// Before:  %tmp = MUL %a, 4
//          %addr = ADD %base, %tmp
//          %val = LOAD %addr
// After:   %val = LOAD [%base, %a, lsl 2]  (利用基址+索引+左移寻址)

// 2. FMA (Fused Multiply-Add) 模式识别
// Before:  %tmp = FMUL %a, %b
//          %result = FADD %tmp, %c
// After:   %result = FMA %a, %b, %c  (融合乘加)

// 3. 常量传播和折叠
// Before:  %1 = MOV 3
//          %2 = MUL %a, %1
// After:   %2 = LSL %a, 2  (乘3被优化为加法链或移位)
```

**目标 hook**：
```cpp
// 定义可被 combiner 识别的模式
bool getMachineCombinerPatterns(
    MachineInstr &Root,
    SmallVectorImpl<MachineCombinerPattern> &Patterns) const;

// 生成替代指令序列
void genAlternativeCodeSequence(
    MachineInstr &Root, MachineCombinerPattern Pattern,
    SmallVectorImpl<MachineInstr *> &InsInstrs, ...) const;
```

### MachineCSE - 机器级公共子表达式消除

```cpp
// Before MachineCSE:
// bb.0:
//   %1 = ADD %a, %b
//   %2 = MUL %1, %c
//   ...
//   %3 = ADD %a, %b     // 与 %1 相同！
//   %4 = SUB %3, %d
//
// After MachineCSE:
//   %2 = MUL %1, %c
//   ...
//   %4 = SUB %1, %d     // 重用 %1
```

MachineCSE 需要考虑 LLVM IR CSE 不需要考虑的约束：
- **寄存器 flags 依赖**：如 `implicit $eflags` 的 flags 定义/使用关系
- **内存依赖**：load 指令的别名关系
- **物理寄存器冲突**：如果某个虚拟寄存器被分配了特定物理寄存器

### MachineLICM - 循环不变代码外提

```cpp
// Before MachineLICM:
// loop:
//   %const = MOV 42          // 循环不变！
//   %addr = LOAD [%ptr]
//   %val = ADD %addr, %const
//   ...
//   br loop
//
// After MachineLICM:
// preheader:
//   %const = MOV 42          // 外提到循环前
// loop:
//   %addr = LOAD [%ptr]
//   %val = ADD %addr, %const
//   ...
//   br loop
```

MachineLICM 比 LLVM IR 的 LICM 更保守，因为它必须考虑寄存器压力：
外提太多指令会增加循环外的寄存器需求。它使用 `MachineRegisterInfo` 和
register pressure sets 来做启发式决策。

### Machine Pass 的分析框架

| 分析 | 用途 |
|------|------|
| `MachineLoopInfo` | Machine IR 层级循环检测 |
| `MachineDominatorTree` | 支配关系 |
| `MachinePostDominatorTree` | 后支配关系 |
| `MachineBlockFrequencyInfo` | 基本块执行频率估计 |
| `MachineBranchProbabilityInfo` | 分支概率估计 |
| `LiveIntervals` | 寄存器的活跃区间 |
| `LiveVariables` | 基本块边界活跃信息 |
| `SlotIndexes` | 指令的程序点编号 |
| `LiveRegUnits` | 物理寄存器单位的活跃状态（基本块级别） |
| `MachineRegisterInfo` | 虚拟/物理寄存器跟踪 |

**分析失效管理**：Machine pass 必须通过 `getAnalysisUsage()` 声明它们的分析需求
和哪些分析会失效：

```cpp
void MyMachinePass::getAnalysisUsage(MachineFunctionAnalysisManager &AM) const {
    AM.addRequired<MachineLoopInfo>();           // 需要循环信息
    AM.addRequired<MachineDominatorTree>();       // 需要支配树
    AM.addPreserved<MachineRegisterInfo>();       // 不修改寄存器信息
    // 默认：所有其他分析被标记为失效
}
```

## AI 编译器关联

### MLIR Pass Pipeline 设计 vs Machine Pass Pipeline

MLIR 的 pass pipeline 设计在很多方面借鉴了 LLVM Machine pipeline 的经验教训：

| 设计决策 | LLVM Machine Pipeline | MLIR 采纳的改进 |
|---------|----------------------|----------------|
| Pipeline 构建 | TargetPassConfig 虚方法 | 显式 pass 列表（更可预测） |
| Pass 间通信 | 隐式：通过 analyses | 显式：Operation 属性 + analyses |
| 测试方式 | .mir 文件 + -run-pass | .mlir 文件 + -pass-pipeline |
| 多级 IR | 无（单 Machine IR） | 多 dialect 共存，逐步 lowering |
| SSA 保证 | 灵活（允许非 SSA） | 核心 dialects 强制 SSA |

**MLIR 的改进为 AI 编译器带来的好处**：
1. **Pipeline 可组合性**：IREE 可以为不同的硬件后端组合不同的 pass pipeline
2. **渐进 lowering**：每个 pass 只降低一个抽象维度，更容易推理和调试
3. **模式重写系统**：MLIR 的 `PatternRewriter` + dialect conversion 比
   Machine pass 的手动模式匹配更声明式

### Triton Backend 中 Machine Pass Pipeline 的实际作用

Triton 编译器的 NVPTX 后端使用简单的 Machine pass pipeline：

```python
# Triton 内部使用的 NVPTX pass pipeline (概念性):
passes = [
    # ISel: LLVM IR → NVPTX MachineInstr
    "nvptx-isel",

    # 基础清理
    "machine-cse",
    "peephole-opt",

    # NVPTX 特定的 pseudo 展开
    "nvptx-expand-pseudos",

    # 无寄存器分配！(PTX 有无限虚拟寄存器)
    # 无指令调度！(留给 ptxas)

    # 直接输出
    "nvptx-asm-printer",
]
```

**为什么 Triton 的 Machine pipeline 如此简单**：
- PTX 是虚拟 ISA，有无限虚拟寄存器 → 不需要寄存器分配
- PTX 指令调度由 ptxas (NVIDIA 专有) 负责 → 不需要 LLVM 调度
- 只需要做基本的 CSE 和 peephole 优化即可
- Triton 层面的优化（coalescing、pipelining）已经消除了大量冗余

### IREE 的 GPU Pass 优化流程

IREE 的 GPU 优化更复杂，因为它支持多种 GPU 后端（CUDA、ROCm、Intel GPU）：

```
IREE GPU Flow (简化):
┌─────────────────────────────────────┐
│ MLIR Optimization (IREE 层面)        │
│  - dispatch region formation         │
│  - workgroup tiling                  │
│  - vectorization                     │
│  - bufferization                     │
├─────────────────────────────────────┤
│ Lowering to LLVM (MLIR)              │
│  - gpu → llvm dialect                │
│  - 插入 gpu.barrier                 │
│  - shared memory promotion           │
├─────────────────────────────────────┤
│ Translate to LLVM IR                 │
│  - mlir-translate → LLVM IR          │
├─────────────────────────────────────┤
│ LLVM Pass Pipeline                   │
│  - -O2 优化                          │
│  - SLPVectorizer (可选)              │
│  - LoopVectorizer (可选)             │
├─────────────────────────────────────┤
│ LLVM Machine Pass Pipeline           │
│  - GlobalISel (AMDGPU) / SDISel      │
│  - MachineCSE, MachineSink           │
│  - Register Allocation               │
│  - Post-RA Scheduler                 │
│  - PrologueEpilogue                  │
└─────────────────────────────────────┘
```

### 自定义 AI 加速器的 Pass Pipeline 设计原则

对于自定义 AI 加速器，Machine pass pipeline 的设计需要考虑：

1. **指令选择器的选择**：
   - 简单 ISA → SDISel（成熟、bug 少）
   - 需要特殊 pattern matching → GlobalISel（可以插入自定义 passes）
   - 超大规模并行单元 → 可能需要 TableGen + 自定义 pattern match code

2. **必须的 Machine passes**：
   ```
   最小可行 pipeline:
   ISel → PHIElimination → RegisterAllocation → PrologEpilogInserter → Output
   ```

3. **可选的优化 passes**：
   - `MachineCSE`：消除冗余指令（几乎总是值得加入）
   - `PeepholeOptimizer`：本地改进（开销低）
   - `MachineLICM`：循环优化（在 MLIR 层面已做的话可以跳过）
   - `MachineCombiner`：访问模式优化（如融合乘加）

4. **AI 特定 passes**：
   - TensorCore/Matrix Unit 指令匹配（类似 `AMDGPU` 的 MFMA pattern）
   - SIMD/SIMT 向量化（将标量操作合并为向量操作）
   - Memory coalescing optimization（合并分散的 memory access）

## 示例说明

### 示例 1：自定义 pass 注入

```cpp
// MyTargetPassConfig.cpp
bool MyTargetPassConfig::addPreRegAlloc() {
    if (!TargetPassConfig::addPreRegAlloc())
        return false;

    // O1+ 添加激进优化
    if (getOptLevel() != CodeGenOptLevel::None)
        addPass(createMyTargetVectorCombinePass());

    // 特定子目标才需要
    if (TM->getSubtargetImpl()->hasAIAccelerator())
        addPass(createMyTargetTensorCoreISelPass());

    return true;
}

bool MyTargetPassConfig::addPreEmitPass() {
    // 展开 pseudo 指令为实际硬件指令
    addPass(createMyTargetExpandPseudoInsts());
    // 可能需要扩展的长跳转
    addPass(createMyTargetBranchRelaxation());
    return true;
}

bool MyTargetPassConfig::addPreEmitPass2() {
    // 最后的机会：插入 NOP 填补延迟槽
    if (TM->getSubtargetImpl()->hasDelaySlots())
        addPass(createMyTargetDelaySlotFiller());
    return true;
}
```

### 示例 2：分析一个完整的 Machine pass 运行序列

```bash
$ llc -mtriple=h2blb -debug-pass=Structure input.ll -o /dev/null

Machine Pass Pipeline Structure:
  FunctionPass Manager
    Machine InstCombiner
    Machine Early If-Conversion
    Machine Block Placement
    PHIElimination                # ← SSA → 非 SSA 转换点
    Two-Address Instruction Pass
    Machine Late Instructions Cleanup Pass
    Machine Copy Propagation Pass
    Post-RA Pseudo Instruction Expansion Pass
    Post-RA Machine LICM
    Post-RA Machine CSE
    Optimize PHIs
    Stack Frame Finalization
    Control Flow Optimizer
    # ... more passes ...
    Machine Verifier               # ← 验证 Machine IR 合法性
    Machine Outliner
    Function Pass Manager
      # ... passes ...
```

### 示例 3：为 AI 加速器定制 pass pipeline

```cpp
// AITargetPassConfig.cpp - 自定义 AI 加速器 pass pipeline
class AITargetPassConfig : public TargetPassConfig {
    bool addInstSelector() override {
        // 使用 GlobalISel 以支持自定义 AI 指令
        addPass(createAITargetIRTranslator());
        addPass(createAITargetLegalizer());
        addPass(createAITargetRegBankSelect());
        addPass(createAITargetInstructionSelector());
        return false; // false = 已完全处理 ISel
    }

    bool addPreRegAlloc() override {
        // AI 专用的 SSA 级优化
        addPass(createAITargetMatrixCombinePass()); // 融合矩阵操作
        addPass(createAITargetSIMDCoalescePass());  // 合并 SIMD 访问
        addPass(createAITargetMemoryLayoutPass());  // 优化 SRAM/DRAM 布局
        return true;
    }

    bool addPostRegAlloc() override {
        // Post-RA: 利用物理寄存器分配结果
        addPass(createAITargetRegisterPackingPass()); // 寄存器打包
        return true;
    }
};
```

## 工业落地：修改 Machine pipeline 的验收门禁

新增、删除或移动一个 machine pass 时，至少回答：

- 它要求 SSA、NoVRegs、TracksLiveness 等哪些 MachineFunction properties？
- 它运行在 regalloc 前还是后？操作虚拟寄存器还是物理寄存器？
- 它修改 CFG、live intervals、slot indexes 或 register pressure 后，哪些分析仍然有效？
- 对所有 subtarget 都启用，还是受 feature/cpu/opt-level 控制？
- 编译时间与代码质量收益是否覆盖新增复杂度？

开发阶段可启用 machine verifier 并在 pass 前后保存 MIR；CI 除 lit 回归外还要比较：

```text
编译时间 / 峰值内存
汇编或对象文件大小
spill/reload 数量与栈帧大小
关键指令数、分支数、寄存器压力
目标机 benchmark 的均值、方差与尾延迟
```

不要仅凭单个 kernel 的汇编更短就调整全局 pipeline。pass 顺序变化可能改善一个 subtarget，
同时破坏另一个 subtarget 的 pattern、调度或寄存器压力。

## 总结

Machine pass pipeline 是 LLVM 后端的核心执行框架，管理 Machine IR 从 ISel 输出到最终代码的全过程：

- **分阶段设计**：ISel → SSA优化 → PHI消除 → 寄存器分配 → Post-RA优化 → Prologue/Epilogue → 输出
- **TargetPassConfig** 提供模板方法模式，通过虚方法在关键注入点插入 passes
- **SSA ↔ 非SSA 转换**是不可逆的，决定了每个阶段的可用优化类型
- **CodeGenPrepare** 在 LLVM IR 层面为后端准备代码（拆分 GEP、下沉地址计算等）
- **PeepholeOptimizer** 做本地窗口优化（COPY 折叠、扩展消除）
- **MachineCombiner** 做指令重组优化（FMA 融合、寻址模式优化）
- **MachineCSE / MachineLICM** 做全局优化（公共子表达式消除、循环外提）
- **Post-RA 优化** 利用物理寄存器信息做最后的改进
- **分析框架** 提供循环分析、活跃分析、分支概率等辅助信息

**与 AI 编译器的关系**：
- MLIR 的 pass pipeline 设计借鉴了 LLVM 的经验，使用更显式、更可预测的声明式 pipeline
- Triton 的 Machine pipeline 非常简单，因为 PTX 是虚拟 ISA——无需寄存器分配和指令调度
- IREE 的完整 GPU 优化跨越 MLIR → LLVM IR → Machine IR 三个层次
- 自定义 AI 加速器应选择最小但有效的 Machine pass 集合，根据 ISA 复杂度和
  MLIR 层面已完成的优化程度来决定
- `TargetPassConfig` 的注入点机制允许灵活地为 AI 加速器添加专用 passes
