# Chapter 14: Getting Started with Instruction Selection

## 核心概念（详细展开）

### 指令选择问题的本质

Instruction Selection (ISel) 将 LLVM IR 的 "无限可能" 空间映射到目标架构的 "有限指令集"。
可以把它看作一个漏斗（funnel）：

```
LLVM IR (无限可能空间):
  add, sub, mul, sext, zext, load, store, call, phi, select...
  i1, i8, i32, i64, float, double, <4 x float>, ptr...
  ↓ ISel (漏斗)
Machine IR (目标特定指令):
  ADDWrr, SUBWrr, MADDWrrr (什么组合存在由 ISA 决定)
  GPR32, GPR64, FPR32, FPR64 (什么寄存器类存在由 ISA 决定)
```

LLVM 提供 **两个半** ISel 框架：
1. **SDISel (SelectionDAG ISel)**：传统的、单体化的指令选择器
2. **GlobalISel (GISel)**：现代的、模块化的指令选择器
3. **FastISel**：SDISel 的子选择器，追求极致编译速度

### 三个框架的对比

| 维度 | FastISel | SDISel | GlobalISel |
|------|----------|--------|------------|
| 编译时间 | 最快 (但常回退) | ~20% 的后端时间 | 平衡，~2x 快于 SDISel |
| 模块化 | 单体 | 单体 (一个 MachineFunctionPass) | Pipeline of passes |
| 优化范围 | 单基本块（受限） | 单基本块（DAG 限制） | 函数范围 |
| 可测试性 | 困难（只能从 LLVM IR 开始） | 困难（同上） | 容易（.mir per pass） |
| 灵活性 | 最低 | 中等（通过 TargetLowering hooks） | 最高（完整 pass pipeline 控制） |
| 成熟度 | 稳定 | 成熟（数十年打磨） | 较新（仍在完善） |
| 回退支持 | → SDISel (per BB) | — | → SDISel (per function) |

**框架选择决策树**：
```
开始新的后端？
├─ 想要最快的开发速度？
│   └─ 用 SDISel（成熟、bug 少、TableGen 支持完善）
├─ 想要长期可维护性？
│   └─ 用 GlobalISel（模块化、可测试、未来方向）
├─ 目标架构的 ISA 很规则（如 RISC-V）？
│   └─ GlobalISel 表现优秀
├─ 目标架构的 ISA 很复杂（如 x86）？
│   └─ SDISel 目前支持更好
└─ 只需要 O0 编译？
    └─ FastISel 作为补充
```

### 两种 IR 表示的对比

#### SelectionDAG (SDISel)

```text
SelectionDAG has 9 nodes:
  t0: ch,glue = EntryToken
      t2: i16,ch = CopyFromReg t0, Register:i16 %0
      t4: i16,ch = CopyFromReg t0, Register:i16 %1
    t5: i16 = add t2, t4
  t7: ch,glue = CopyToReg t0, Register:i16 $r1, t5
  t8: ch = H2BLBISD::RETURN_GLUE t7, Register:i16 $r1, t7:1
```

**关键概念**：
- **SDNode**：DAG 中的节点，有 opcode (`ISD::ADD`、`ISD::LOAD`... 或自定义 `XXXISD::FOO`)
- **SDValue**：SDNode 的单个结果值，是 `(SDNode*, index)` 的轻量包装
- **EVT / MVT**：Extended Value Type / Machine Value Type，表示类型
- **三种依赖边**：
  - **Data dependency**：数据流（use-def chain，方向与 def-use 相反！）
  - **Chain dependency**：调度顺序（如 load 必须在 aliasing store 之后）
  - **Glue dependency**：强制相邻（如 ABI lowering 中物理寄存器的紧凑序列）

**EntryToken** 是所有 DAG 的起始节点，提供 chain 和 glue。

#### Generic Machine IR (GlobalISel)

```text
# 通用虚拟寄存器：有类型、可选寄存器 bank
%0:_(s32) = G_CONSTANT i32 0           # 通用常量
%1:gprb(s32) = G_LOAD %ptr_addr         # 通用 load（带寄存器 bank）
%2:gprb(s32) = G_ADD %1, %0             # 通用加法

# 普通虚拟寄存器：有寄存器类
%3:gpr32 = ADDWrr %2, %2                # 目标特定指令
```

**关键概念**：
- **G_ 前缀 opcode**：通用操作码（`G_ADD`、`G_LOAD`、`G_CONSTANT`...）
- **LLT**：Low-Level Type，如 `s32` (scalar 32)、`p0` (pointer addrspace 0)、`<4 x s32>` (vector)
- **RegisterBank**：寄存器大类（`GPR` 通用、`FPR` 浮点、`VR` 向量）
- **Generic Virtual Register**：有 LLT，可能无 RegisterBank，肯定无 RegisterClass

**G_MIR 的 lowering 约束渐进**：

| 阶段 | Opcode 约束 | 虚拟寄存器约束 |
|------|-----------|-------------|
| Pre-legalization | 任何 G_ opcode，任何类型 | 必须有 LLT，寄存器 bank 可选 |
| Post-legalization | 只有合法的 G_ opcode + 合法类型 | 同上 |
| Post-RegBankSelect | 只有合法的 G_ opcode | 必须有寄存器 bank |
| Post-InstructionSelect | 无 G_ opcode | 必须有寄存器类（普通虚拟寄存器） |

## LLVM / MLIR 流程（深入）

### MLIR Lowering from GPU Dialect to LLVM Dialect (Parallel!)

MLIR 的 GPU lowering 是 **两阶段并行 lowering**：

```
Stage 1: GPU Kernel Code (device code)
┌──────────────────────────────────────┐
│ gpu.func @kernel(...)                 │
│   scf.for %i = ...                    │
│     memref.load %buf[%i]             │
│     arith.addf %x, %y                │
│     gpu.barrier                       │
│     memref.store %result, %buf[%i]   │
└──────────────────┬───────────────────┘
                   ↓ convert-gpu-to-nvvm / convert-gpu-to-rocdl
┌──────────────────────────────────────┐
│ llvm.func @kernel(...)                │
│   llvm.br ...                        │
│   llvm.load %ptr                     │
│   llvm.fadd %x, %y                   │
│   llvm.call @llvm.nvvm.barrier0()   │
│   llvm.store %result, %ptr          │
└──────────────────────────────────────┘

Stage 2: Host-side Launch Code (并行处理)
┌──────────────────────────────────────┐
│ gpu.launch_func @kernel              │
│   blocks in (%bx, %by, %bz)          │
│   threads in (%tx, %ty, %tz)         │
│   args(%buf : memref<...>)           │
└──────────────────┬───────────────────┘
                   ↓ gpu-to-llvm
┌──────────────────────────────────────┐
│ llvm.call @cudaLaunchKernel(...)      │
│   (或 HIP 等价调用)                   │
└──────────────────────────────────────┘
```

**MLIR 的 ISel 等价物**：
MLIR 没有传统意义上的 "指令选择"。MLIR 的 lowering 流程是：
1. **Pattern-based lowering**：`tosa → linalg`, `linalg → loops`, `loops → scf`
2. **Dialect conversion**：`arith → llvm`, `memref → llvm`, `func → llvm`
3. **LLVM dialect translation**：`llvm dialect → LLVM IR`

MLIR 的 pattern rewriting 与 LLVM ISel 的对比：

| 特性 | LLVM SDISel (DAG Pattern) | MLIR Pattern Rewriting |
|------|--------------------------|----------------------|
| Pattern 描述 | TableGen DAG pattern (如 `(add GPR:$a, GPR:$b)`) | MLIR C++ Pattern (或 PDL) |
| 匹配范围 | 单 DAG（单基本块） | 可跨基本块（操作间关系图） |
| 类型系统 | EVT / MVT（预定义类型） | 任意 Type（dialect 定义） |
| 回退支持 | pattern 失败 → 尝试下一条 | pattern 失败 → 尝试下一条 |
| 复杂度 | pattern matching 很快 | pattern matching 可更复杂 |

### Triton 的指令选择策略

Triton 不使用 MLIR 的 pattern rewriting 来选择指令，而是用 C++ 直接生成 LLVM IR：

```python
# Triton kernel (Python)
@triton.jit
def matmul_kernel(A, B, C, M, N, K):
    pid = tl.program_id(0)
    a_ptrs = A + pid * K
    b_ptrs = B + pid
    # ... 在 C++ backend 中生成 LLVM IR ...
```

```cpp
// Triton C++ backend (简化)
Value TritonGPUToLLVM::visit_DotOp(DotOp op) {
    // 矩阵乘法 → 直接生成 LLVM IR intrinsic
    if (useTensorCore) {
        // 生成 NVVM MMA intrinsic
        return builder.CreateIntrinsic(
            "llvm.nvvm.mma.m16n8k8.row.col.f32.f32",
            {a_val, b_val, c_val});
    } else {
        // 生成 FMA 循环展开
        return emitFMAChain(a_val, b_val, c_val);
    }
}
```

Triton 的策略是 **"在 Triton IR 层面做高级优化 (tiling, pipelining)，
在 LLVM IR 生成层面做精确的指令选择"**。最终的指令选择（LLVM IR → PTX）
由 LLVM 的 NVPTX SDISel 完成。

### GlobalISel 与 AI 加速器

GlobalISel 的设计对 AI 加速器特别有利：

1. **函数范围的优化视野**：
   ```cpp
   // GlobalISel 可以跨越基本边界匹配模式
   // 例如：将分散的矩阵加载 + 矩阵乘法合并为一个 tensor core 指令
   ```

2. **RegisterBank 概念天然适配 AI 加速器**：
   ```cpp
   // AI 加速器通常有多级寄存器文件
   enum AIAccelRegisterBank {
       ScalarRegBank,    // 标量寄存器
       VectorRegBank,    // 向量寄存器 (256-bit)
       MatrixRegBank,    // 矩阵寄存器 (用于 systolic array)
       PredicateRegBank, // 谓词/掩码寄存器
       AddrRegBank,      // 地址/索引寄存器
   };
   ```

3. **可插入的自定义 passes**：
   ```cpp
   // 在 GlobalISel pipeline 中插入 AI 加速器特定 passes
   bool AITargetPassConfig::addGlobalInstructionSelect() {
       // Legalizer 之后、ISel 之前插入自定义 pass
       addPass(createAITargetMatrixCombinePass());
       return TargetPassConfig::addGlobalInstructionSelect();
   }
   ```

## 关键机制解析（工业视角）

### SDISel 的核心 API

```cpp
// SelectionDAG - DAG 节点工厂（自动 CSE）
SDValue getNode(unsigned Opcode, SDLoc DL, SDVTList VTs,
                ArrayRef<SDValue> Ops);
SDValue getCopyFromReg(SDValue Chain, SDLoc DL, unsigned Reg, EVT VT);
SDValue getCopyToReg(SDValue Chain, SDLoc DL, unsigned Reg, SDValue Val);
SDValue getLoad(EVT VT, SDLoc DL, SDValue Chain, SDValue Ptr,
                MachinePointerInfo PtrInfo, ...);
SDValue getStore(SDValue Chain, SDLoc DL, SDValue Val, SDValue Ptr,
                 MachinePointerInfo PtrInfo, ...);

// 创建 MachineInstr 节点（用于目标特定指令）
SDNode *getMachineNode(unsigned Opcode, SDLoc DL, EVT VT,
                       MVT OtherVT, ...);  // variadic
SDNode *getMachineNode(unsigned Opcode, SDLoc DL,
                       ArrayRef<EVT> ResultTypes,
                       ArrayRef<SDValue> Ops);
```

**Continuous CSE**: `getNode()` 会自动检查是否已有相同操作数的节点，
如果有就直接返回现有节点——这是 DAG 的基本优化能力。

### SelectionDAG 的三种依赖边

```cpp
// 1. Data dependency: 数据流（与 LLVM IR 的 use-def 相反）
// SDValue %result = add %x, %y  实现为:
// SDNode *addNode = DAG.getNode(ISD::ADD, DL, VT, {xVal, yVal});

// 2. Chain dependency: 强制顺序（如内存操作）
// TokenFactor 可合并多个 chain:
SDValue Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other,
                             {chain1, chain2});

// 3. Glue dependency: 强制相邻（如 ABI lowering）
// 创建一个节点，它的输出 "粘" 到另一个输入上
SDValue Glue = ...; // glue 类型的 SDValue
SDValue newNode = DAG.getNode(ISD::CopyToReg, DL, MVT::Other,
                               {chain, Reg, Val, Glue}); // Glue 最后
```

**GPU 编译器中的依赖边使用案例**：
```cpp
// PTX bar.sync 需要 chain 依赖确保之前的 shared memory 写完成
// %ptr_st = STORE_SHARED %val, [%saddr]
// bar.sync 0                // 依赖上面的 store
// %val_ld = LOAD_SHARED [%saddr]  // 依赖 bar.sync

// 在 SDISel 中:
SDValue StoreChain = DAG.getStore(EntryChain, DL, Val, SharedAddr, ...);
SDValue Barrier = DAG.getNode(NVPTXISD::BARRIER_SYNC, DL, StoreChain);
SDValue Load = DAG.getLoad(MVT::f32, DL, Barrier, SharedAddr, ...);
```

### GlobalISel 的核心 API

```cpp
// MachineIRBuilder - GlobalISel 的构建器
MachineIRBuilder MIRBuilder(MBB);
// 创建通用虚拟寄存器
Register VReg = MRI.createGenericVirtualRegister(LLT::scalar(32));
// 构建通用指令
MachineInstrBuilder MIB = MIRBuilder.buildAdd(LLT::scalar(32), LHS, RHS);

// MachineRegisterInfo - 管理通用虚拟寄存器的类型
LLT getType(Register Reg) const;
void setType(Register Reg, LLT Ty);
const RegisterBank *getRegBank(Register Reg) const;
void setRegBank(Register Reg, const RegisterBank &RB);

// 检查是否为 GlobalISel 的通用 opcode
bool isPreISelGenericOpcode(unsigned Opcode);
```

### ISel 框架在 CodeGen Pipeline 中的连接

```cpp
// TargetMachine 构造函数中设置 ISel 类型
MyTargetMachine::MyTargetMachine(...) {
    // 选择 ISel 框架（3 选 1）
    setGlobalISel(true);   // 使用 GlobalISel
    // 或
    setFastISel(true);     // 使用 FastISel + SDISel
    // 默认: SDISel
}

// TargetPassConfig 中连接 ISel
bool MyTargetPassConfig::addInstSelector() {
    // SDISel:
    addPass(createMyTargetISelDag(getMyTargetTargetMachine()));

    // 或 GlobalISel:
    addPass(createIRTranslator());       // IR builder
    addPass(createLegalizer());          // Legalization
    addPass(createRegBankSelect());       // Register bank select
    addPass(createInstructionSelector()); // Selection
    return false;  // 返回 false 表示已完全自定义
}
```

### AsmPrinter - CodeGen Pipeline 的最后连接

```cpp
class MyTargetAsmPrinter : public AsmPrinter {
public:
    void emitInstruction(const MachineInstr *MI) override {
        MCInst TmpInst;
        // MachineInstr → MCInst 转换
        lowerToMCInst(*MI, TmpInst);
        // 通过 MCStreamer 输出
        EmitToStreamer(*OutStreamer, TmpInst);
    }

private:
    void lowerToMCInst(const MachineInstr &MI, MCInst &OutMI) {
        OutMI.setOpcode(MI.getOpcode());
        for (const MachineOperand &MO : MI.operands()) {
            MCOperand MCOp;
            if (lowerOperand(MO, MCOp))
                OutMI.addOperand(MCOp);
        }
    }

    bool lowerOperand(const MachineOperand &MO, MCOperand &MCO) {
        switch (MO.getType()) {
        case MachineOperand::MO_Register:
            if (MO.isImplicit()) return false;
            MCO = MCOperand::createReg(MO.getReg());
            break;
        case MachineOperand::MO_Immediate:
            MCO = MCOperand::createImm(MO.getImm());
            break;
        // ... more operand types ...
        }
        return true;
    }
};
```

## AI 编译器关联

### MLIR 的 Pattern Rewriting vs SDISel/GlobalISel

```
LLVM SDISel Pattern (TableGen):
┌────────────────────────────────────┐
│ def : Pat<(add GPR32:$a, GPR32:$b),│
│           (ADDWrr GPR32:$a,        │
│                   GPR32:$b)>;      │
└────────────────────────────────────┘
  ↓ gen-dag-isel TableGen backend
  ↓ 生成的 C++ switch 语句
┌────────────────────────────────────┐
│ case ISD::ADD:                     │
│   if (VT == MVT::i32)              │
│     ReplaceNode(ADDWrr, ...);      │
│   break;                           │
└────────────────────────────────────┘

MLIR Pattern Rewriting (C++):
┌────────────────────────────────────┐
│ struct AddToLLVMAdd : public       │
│     OpRewritePattern<arith::AddFOp>{│
│   LogicalResult matchAndRewrite(   │
│       arith::AddFOp op,            │
│       PatternRewriter &rewriter) { │
│     auto llvmAdd = rewriter.create │
│       <LLVM::FAddOp>(op.getLoc(),  │
│        op.getOperands());          │
│     rewriter.replaceOp(op, llvmAdd);│
│     return success();              │
│   }                                │
│ };                                 │
└────────────────────────────────────┘
```

**关键区别**：
- SDISel 的 pattern 是 **声明式的 DAG 匹配**（`(add GPR32:$a, GPR32:$b)`），编译时生成
- MLIR 的 pattern 是 **命令式的 C++ 代码**（`matchAndRewrite`），运行时执行
- SDISel 的优化范围受限于 DAG（单基本块），MLIR 的 pattern 可以跨基本块
- MLIR 正在开发 PDL (Pattern Description Language)，提供类似 TableGen 的声明式匹配

### Triton 的 GPU 指令选择流程

Triton 不使用 MLIR 的 pattern rewriting，而是使用 **直接 LLVM IR 生成**：

```
Triton IR:
  tt.dot %a, %b, %c {allowTF32 = true}
       ↓ (Triton Backend C++)
LLVM IR:
  call <2 x float> @llvm.nvvm.mma.m16n8k8.row.col.f32.f32(...)
       ↓ (LLVM NVPTX SDISel)
NVPTX MachineInstr:
  %v0:float32regs = MMA_F32_M16N8K8 %a, %b, %c, ...
       ↓ (NVPTX MCInstLower + AsmPrinter)
PTX:
  mma.sync.aligned.m16n8k8.row.col.f32.f32.f32.f32
    {%f1, %f2}, {%f3, %f4}, {%f5, %f6}, {%f7, %f8};
```

Triton 在 LLVM IR 层面选择 TensorCore 指令，因为：
1. PTX `mma` 指令在 LLVM IR 中有对应的 intrinsic (`@llvm.nvvm.mma.*`)
2. NVPTX SDISel 知道如何将这些 intrinsic 匹配为 `MMA_*` SDNode
3. 最终的 SDISel 只做简单的 1:1 映射（intrinsic → instruction）

### 用 GlobalISel 为 AI 加速器建模自定义指令

对于一个有 systolic array 的 AI 加速器：

```cpp
// 1. 定义自定义 G_ opcode（在 GenericOpcodes.td 或目标特定文件中）
def G_SYSTOLIC_MATMUL : GenericInstruction {
    let OutOperandList = (outs type0:$dst);
    let InOperandList = (ins type1:$a, type2:$b, type3:$acc);
    let hasSideEffects = false;
}

// 2. 定义 RegisterBank
def SABank : RegisterBank<"SystolicArray", [SA256RegClass]>;

// 3. 在 Legalizer 中处理通用操作
bool AILegalizerInfo::legalizeCustom(
    MachineInstr &MI, MachineRegisterInfo &MRI, ...) const {
    // 对于过大或过小的矩阵，分解为多次 G_SYSTOLIC_MATMUL
    if (MI.getOpcode() == TargetOpcode::G_SYSTOLIC_MATMUL) {
        return legalizeSystolicMatMul(MI, MRI);
    }
    return false;
}

// 4. 在 InstructionSelector 中做最终选择
bool AIInstructionSelector::select(MachineInstr &I) {
    if (I.getOpcode() == TargetOpcode::G_SYSTOLIC_MATMUL) {
        // 根据矩阵大小选择合适的硬件指令
        Register SA = MRI.createVirtualRegister(&SA256RegClass);
        BuildMI(MBB, I, DL, TII.get(AI::SYS_MM_256x256))
            .addDef(SA).addUse(I.getOperand(1))
            .addUse(I.getOperand(2)).addUse(I.getOperand(3));
        I.eraseFromParent();
        return true;
    }
    return false;
}
```

### 选择 SDISel vs GlobalISel 的实践指南（AI 编译器视角）

**选择 SDISel 的场景**：
- 你的 ISel 主要是简单的 1:1 模式匹配（如 intrinsic → instruction）
- 需要稳定的、已验证的代码生成
- 团队对 LLVM 还不熟悉，希望快速上手

**选择 GlobalISel 的场景**：
- AI 加速器有复杂的 multi-pattern 指令（如能被合并为一条的多种计算模式）
- 需要在函数范围内做跨基本块的模式匹配（如跨基本块的矩阵操作融合）
- 多级寄存器文件需要 RegisterBank 机制
- 需要高度模块化的测试（可以将每个 pass 隔离测试）

**混合策略（Triton 模式）**：
```
高级融合 + tiling → 在自有的 IR 层完成
指令 intrinsic 生成 → 在 LLVM IR 层完成
最终 ISel → 交给 SDISel 做 1:1 映射
```

## 示例说明

### 示例 1：SDISel 的模式匹配

```tablegen
// XXXInstrInfo.td 中的 pattern 定义

// 简单 pattern：add → ADD
def : Pat<(add GPR32:$a, GPR32:$b), (ADD GPR32:$a, GPR32:$b)>;

// 复杂 pattern：mul + add → MADD (乘加融合)
def : Pat<(add (mul GPR32:$a, GPR32:$b), GPR32:$c),
          (MADD GPR32:$a, GPR32:$b, GPR32:$c)>;

// 内存 pattern：load + add → 带寻址模式的 load
def : Pat<(add GPR32:$base, (shl GPR32:$idx, (i32 2))),
          (ADDR_INDEX GPR32:$base, GPR32:$idx)>;
```

### 示例 2：GlobalISel 的 pass 隔离测试

```bash
# 测试 IRTranslator: LLVM IR → G_MIR
llc -global-isel -stop-before=legalizer input.ll -o before-legalize.mir

# 测试 Legalizer: 只运行 legalize pass
llc -run-pass=legalizer before-legalize.mir -o after-legalize.mir

# 测试 RegBankSelect: 只运行 register bank select
llc -run-pass=regbankselect after-legalize.mir -o after-regbank.mir

# 测试最终的 ISel:
llc -run-pass=instruction-select after-regbank.mir -o after-isel.mir
```

### 示例 3：函数范围优化的价值

```llvm
define i32 @widening_mul(i16 %a, i16 %b) {
bb:
  %sext_a = sext i16 %a to i32
  %sext_b = sext i16 %b to i32
  br label %bb_compute

bb_compute:
  %result = mul i32 %sext_a, %sext_b
  ret i32 %result
}
```

- **SDISel**（单基本块）：无法将 `mul` 与 `sext` 融合，因为它们在不同基本块中
- **GlobalISel**（函数范围）：可以看到 `sext` 的定义在 `bb`，使用在 `bb_compute`，
  可以匹配 `G_MUL(G_SEXT(%a), G_SEXT(%b))` 模式并选择为 `MulWide` 指令

## 总结

Instruction Selection (ISel) 是 LLVM 后端中最关键的阶段，负责将 LLVM IR 映射到 Machine IR。
LLVM 提供了三个 ISel 框架：

- **SDISel (SelectionDAG)**：使用 DAG 中间表示（SDNode、SDValue），通过 TableGen pattern
  匹配做指令选择。编译时较快，但受限于单基本块范围，已成熟稳定。
- **GlobalISel (GISel)**：使用增强的 Machine IR（G_MIR），通过 pipeline of passes
  （IRTranslator → Legalizer → RegBankSelect → InstructionSelect）实现模块化指令选择。
  支持函数范围优化，是未来的方向。
- **FastISel**：SDISel 的子选择器，直接从 LLVM IR 到 Machine IR，追求极致编译速度。
  能力有限，经常回退到 SDISel。

**三个共同阶段**：
1. **IR Builder**：LLVM IR → 框架 IR（SDNode DAG 或 G_MIR）
2. **Legalization**：将框架 IR 中不支持的构造重写为支持的
3. **Selection**：框架 IR → Machine IR

**与 AI 编译器的关系**：
- MLIR 的 pattern rewriting（C++/PDL）与 LLVM ISel 有相似之处，但 MLIR 支持任意 dialect
  和跨基本块匹配
- MLIR 的 `gpu` dialect lowering 是两阶段并行：kernel 代码 → `llvm` dialect，
  host launch → LLVM call
- Triton 编译器在 Triton IR 层做高级优化，LLVM IR 生成层做指令 intrinsic 生成，
  最终由 NVPTX SDISel 完成 PTX 指令选择
- GlobalISel 的 RegisterBank + 自定义 passes 机制天然适配 AI 加速器的多级寄存器文件
- 对于自定义 AI 加速器：选择 SDISel（快速稳定）或 GlobalISel（灵活未来），
  取决于 ISA 复杂度和需要的优化范围
