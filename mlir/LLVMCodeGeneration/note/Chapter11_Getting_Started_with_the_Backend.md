# Chapter 11: Getting Started with the Backend

## 核心概念（详细展开）

### Machine IR 的本质

Machine IR (MIR) 是 LLVM 编译流程中出现的 **第二种中间表示**。它位于 Instruction Selection 之后、MC Layer 之前，
与 LLVM IR 形成鲜明的对比：

| 特性 | LLVM IR | Machine IR |
|------|---------|------------|
| SSA 形式 | 强制 SSA | 支持 SSA 也支持非 SSA |
| 类型系统 | 强类型 (i32, float, ptr) | 松散，基于寄存器类 |
| 指令层次 | 子类多态 (BinaryOperator, CallInst 等) | 单一平展类 MachineInstr |
| 目标无关性 | 完全目标无关 | 高度目标特定 |
| 指令属性 | 通过类类型区分 | 通过 opcode + MCInstrDesc 区分 |
| 表示形式 | .ll 文本 | .mir YAML 文本 |

**工业视角**：Machine IR 的设计哲学是 "为硬件建模的灵活性最大化"。在生产编译器中，
Machine IR 必须在两个完全不同的世界中搭桥：
- **SSA 世界**：早期优化阶段（MachineCSE、MachineLICM）要求干净的单赋值形式
- **非 SSA 世界**：寄存器分配后，物理寄存器会被多次重写，必须处理 live range 和 interference

这种灵活性是 MLIR 设计者在构建 MLIR 时保留的核心特性之一：MLIR 的 `llvm` dialect 允许
在 MLIR 层面做 SSA 优化后再 lower 到 LLVM 的 Machine IR 做硬件相关优化。

### Machine IR 的文本表示 (.mir 文件)

```yaml
--- |
  ; LLVM IR section
  define i32 @foo(i32 %a) {
    %b = add i32 %a, 1
    ret i32 %b
  }
...
---
name: foo
alignment: 4
tracksRegLiveness: true
registers:
  - { id: 0, class: gpr32, preferred-register: '' }
frameInfo:
  isFrameAddressTaken: false
body: |
  bb.0 (%ir-block.0):
    successors: %bb.1(0x80000000)
    liveins: $w0
    %0:gpr32 = COPY $w0
    %1:gpr32 = ADDWrr %0, %0
    $w0 = COPY %1
    RET_ReallyLR implicit $w0
...
```

关键语法元素：
- `%` 前缀 → 虚拟寄存器 (Virtual Register)
- `$` 前缀 → 物理寄存器 (Physical Register)
- `@` 前缀 → 符号 (Symbol)
- 裸数字 → 立即数 (Immediate)
- `successors:` → 后继基本块及分支概率（0x80000000 = 1.0，即 100%）
- `liveins:` → 进入基本块时活跃的物理寄存器

### 工作流程

```bash
# 生成 .mir 文件（在指定 pass 前停止）
llc -stop-before=peephole-opt input.ll -o out.mir -simplify-mir

# 运行单个 pass
llc -run-pass=peephole-opt input.mir -o out.mir

# 从指定 pass 开始运行到汇编
llc -start-before=peephole-opt input.mir -o out.s

# 缩小 .mir 文件
llc -stop-before=peephole-opt input.ll -o out.mir -simplify-mir
```

## LLVM / MLIR 流程（深入）

### Machine IR 在 LLVM 编译器全流程中的位置

```
C/C++ 源码
  ↓ Clang 前端
LLVM IR (.ll)
  ↓ 中端优化 Pass (opt)
优化后 LLVM IR
  ↓ Instruction Selection (SDISel / GlobalISel) ← 第14-17章
Machine IR (SSA, 使用虚拟寄存器)
  ↓ 机器优化 Pass (PeepholeOptimizer, MachineCombiner...) ← 第13章
Machine IR (非 SSA, COPY 替代 PHI)
  ↓ 寄存器分配 (Register Allocation)
Machine IR (仅物理寄存器)
  ↓ Post-RA Passes + Code Emission
汇编 (.s) / 目标文件 (.o)
```

### Machine IR vs MLIR 的 llvm dialect

在生产编译器中，理解二者的关系至关重要：

```
MLIR Dialect Stack (Triton/IREE 风格):
  triton dialect / linalg dialect
       ↓ (lowering)
  gpu dialect / scf dialect / arith dialect
       ↓ (lowering)
  llvm dialect  ← MLIR 的 LLVM IR 等价层
       ↓ (translate)
  LLVM IR      ← 标准 LLVM IR
       ↓ (LLVM backend)
  Machine IR   ← 硬件特定的 Machine IR
       ↓
  Assembly / Object File
```

- **MLIR llvm dialect** 在抽象层级上对应 LLVM IR，而非 Machine IR
- **Machine IR** 是低于 MLIR llvm dialect 的一层，处理寄存器分配、指令调度等硬件细节
- 当 Triton 编译器完成 Triton IR → LLVM IR 的 lowering 后，LLVM 的 Machine IR pipeline 接手做物理优化

### 为什么要理解 Machine IR 才能做 AI 编译器

1. **GPU 的寄存器建模极其复杂**：NVIDIA GPU 的寄存器文件在 SM 级别共享，
   使用 Machine IR 的 register units + pressure sets 机制来精确跟踪寄存器压力
2. **Triton 的 PTX 生成路径经过 Machine IR**：Triton 生成的 LLVM IR 需要走完整的
   LLVM backend pipeline（包括 Machine IR 的 SDISel 或 GlobalISel）才能生成 PTX
3. **IREE 的流程**：IREE 将 MLIR 的 `llvm` dialect 翻译为 LLVM IR 后，
   同样依赖 LLVM 的 Machine IR pipeline 做最终的指令选择和调度

## 关键机制解析（工业视角）

### MachineInstr - 平展指令表示

所有机器指令都是 **单一的 MachineInstr 对象**（没有子类层次）：

```cpp
// MachineInstr 的核心结构
class MachineInstr {
    MCInstrDesc *MCID;           // 不可变的指令描述（属性、操作数约束）
    SmallVector<MachineOperand> Operands;  // 操作数数组
    // 操作数顺序：显式定义 → 显式使用 → 隐式操作数
};
```

操作数顺序约束（这是所有后端代码的基础假设）：
1. **显式定义** (Explicit Defs)：只能是寄存器
2. **显式使用** (Explicit Uses)：顺序匹配指令描述
3. **隐式操作数** (Implicit)：定义和使用混合排列

```text
# x86 conditional move - 隐式操作数 $eflags:
%0:gr64 = CMOV64rr_ND %1, %2, 7, implicit $eflags

# AArch64 call - 混合的隐式定义和使用:
BL @_bar, csr_darwin_aarch64_aapcs, implicit-def dead $lr,
    implicit $sp, implicit $x0, implicit-def $sp, implicit-def $x0
```

**工业实践**：隐式操作数的关键使用场景：
- **硬件约束**：如 x86 的 `$eflags` 总是被比较指令隐式设置
- **ABI 约束**：如 call 指令隐式使用 `$sp`（栈指针）和 `$lr`（返回地址）
- **寄存器掩码** (Register Mask)：紧凑表示跨函数调用保留的寄存器集合（callee-saved registers）

### MachineOperand - 核心操作数类型

```cpp
class MachineOperand {
    // 类型检测方法
    bool isReg() const;     // 寄存器操作数
    bool isImm() const;     // 立即数
    bool isSymbol() const;  // 符号（函数名等）
    bool isRegMask() const; // 寄存器掩码（callee-saved 集合）
    bool isFI() const;      // 栈帧索引

    // 角色检测
    bool isDef() const;     // 是定义（写）还是使用（读）
    bool isImplicit() const;// 隐式 vs 显式操作数
    bool isKill() const;    // 这是该寄存器的最后一次使用
    bool isDead() const;    // 这个定义的值永远不会被使用

    // 子寄存器
    unsigned getSubReg() const; // 0 = 完整寄存器，非零 = 子寄存器索引

    // 约束
    bool isTied() const;        // 与另一个操作数共享物理寄存器
    bool isEarlyClobber() const;// 定义在读取输入前就产生
};
```

**Tied Operands（绑定操作数）**：
```text
# 建模 a += b：a 同时被读和写
# TIED_TO 约束强制两个操作数使用同一物理寄存器
%a:gpr32 = ADDWrr_tied %a, %b   # %a 的定义和使用绑定
```
这在建模 GPU 指令时很常见——许多 CUDA PTX 指令有 read-write 语义。

**Early Clobber（提前覆盖）**：
```text
# 定义在输入读取之前产生，不允许定义寄存器与任何输入共享
%a:gpr32 = COMPLEX_INSN early-clobber %a, %b, %c
```

### 寄存器体系（工业级建模）

#### 物理寄存器的层次结构

```
       q0 (128-bit quad register)
      /  \
    d0    d1 (64-bit double registers)
   /  \  /  \
  s0  s1 s2  s3 (32-bit single registers)
```

在 LLVM 中通过 `SubRegIndex` 建模：
```tablegen
def sub32_low  : SubRegIndex<32>;      // 偏移 0, 大小 32
def sub32_high : SubRegIndex<32, 32>;  // 偏移 32, 大小 32

def d0 : Register<"d0"> {
  let SubRegIndices = [sub32_low, sub32_high];
  let SubRegs = [s0, s1];
}
```

#### Register Units - 寄存器冲突检测的核心优化

对于有数百个寄存器的 GPU 后端，遍历所有可能的寄存器别名会严重拖慢编译。
Register Units 将寄存器层次压缩为不相交的最小原子单元：

```
物理寄存器:  s0, s1, s2, s3, d0, d1, q0
Register Units:  u0, u1, u2, u3, u4

映射关系:
  s0 → [u0]           d0 → [u0, u1]       q0 → [u0, u1, u2, u3]
  s1 → [u1]           d1 → [u2, u3]
  ...
```

**detect Conflict 只需检查 unit 集合交集是否为空**。GPT-4 级别的编译器工程需要理解这个
机制——它是 `LiveRegUnits` 和寄存器分配器的高效实现基础。

```cpp
// 高效实现：遍历单位而非所有寄存器别名
MCRegUnitIterator Units(PhysReg, &MRI);
for (MCRegUnit Unit : Units) {
    if (!AvailableUnits.test(Unit))
        return false; // 冲突！
}
```

#### Register Pressure（寄存器压力）

每个 RegisterClass 对一组 PressureSet 贡献权重。当任一 Set 超过 limit，
寄存器压力过高：

```cpp
// 获取一个 regclass 对 pressure sets 的贡献
TargetRegisterClass::getRegClassWeight()     // 返回 RegWeight
TargetRegisterClass::getRegClassPressureSets() // 返回受影响的 pressure set 列表
TargetRegisterInfo::getRegPressureSetLimit()   // 每个 set 的上限
```

GPU 编译器中这个机制极其重要：NVIDIA SM 的寄存器文件有限（如 65536 个 32-bit 寄存器/SM），
register pressure 直接决定 occupancy（活跃 warp 数），影响整体性能。

### 构建 MachineInstr 的两种方式

#### MachineIRBuilder（SSA 友好，用于 GlobalISel）

```cpp
MachineIRBuilder MIRBuilder(MBB);
Register Dst = MIRBuilder.buildAdd(LLT::scalar(32), Src1, Src2)
                  .getReg(0);
```

#### BuildMI / MachineInstrBuilder（传统方式，全流水线通用）

```cpp
Register NewCond = MRI.createVirtualRegister(&MyTarget::GPR32RegClass);
BuildMI(MBB, MBB.end(), DL, TII->get(MyTarget::ADD))
    .addReg(NewCond, RegState::Define)
    .addReg(Src1)
    .addReg(Src2);
```

RegState 是位域：`RegState::Define | RegState::Implicit` 表示隐式定义。

### TableGen 驱动的目标描述

#### 寄存器描述（XXXRegisterInfo.td）

```tablegen
// 子寄存器索引
def sub32 : SubRegIndex<32>;

// 物理寄存器
def s0 : Register<"s0"> { let HwEncoding = 0; }
def s1 : Register<"s1"> { let HwEncoding = 1; }

// 超寄存器
def d0 : Register<"d0"> {
  let SubRegIndices = [sub32, sub32];
  let SubRegs = [s0, s1];
  let CoveredBySubRegs = true;
}

// 寄存器类
def GPR32 : RegisterClass<"MyTarget", [i32], 32, (add s0, s1)>;
```

#### 指令描述（XXXInstrInfo.td）

```tablegen
def ADD : Instruction<> {
  let OutOperandList = (outs GPR32:$dst);
  let InOperandList = (ins GPR32:$src1, GPR32:$src2);
  let AsmString = "add $dst, $src1, $src2";
}
```

#### TableGen 后端产出

| 宏 | 产出内容 | 所在层 |
|----|---------|-------|
| `GET_REGINFO_ENUM` | 寄存器枚举 | MC |
| `GET_REGINFO_MC_DESC` | `InitXXXMCRegisterInfo` 函数 | MC |
| `GET_REGINFO_HEADER` | `XXXGenRegisterInfo` 类声明 | Target |
| `GET_REGINFO_TARGET_DESC` | `XXXGenRegisterInfo` 实现 | Target |
| `GET_INSTRINFO_ENUM` | 指令 opcode 枚举 | MC |
| `GET_INSTRINFO_MC_DESC` | `MCInstrInfo` 实现 | MC |
| `GET_INSTRINFO_HEADER` | `XXXGenInstrInfo` 类声明 | Target |
| `GET_INSTRINFO_TARGET_DESC` | `XXXGenInstrInfo` 实现 | Target |

## AI 编译器关联

### MLIR 的 Lower-Level Dialects 与 LLVM Machine IR

MLIR 有几个关键 dialect 与 LLVM Machine IR 概念对应：

| MLIR Dialect | LLVM 对应 | AI 编译器用途 |
|-------------|----------|-------------|
| `llvm` dialect | LLVM IR（非 Machine IR） | Triton/IREE 的最终 IR 阶段 |
| `gpu` dialect | 无直接对应 | GPU kernel launch 语义 |
| `nvvm` dialect | NVPTX LLVM backend | CUDA 特定指令 |
| `amdgpu` dialect | AMDGPU LLVM backend | ROCm 特定指令 |

**MLIR 的 Machine IR 等价物**：MLIR 没有严格意义上的 "Machine IR"。
MLIR 的 `llvm` dialect 停留在 LLVM IR 级别。当 MLIR 通过 `translateModuleToLLVMIR` 
转换为 LLVM IR 后，再交由 LLVM 的 CodeGen pipeline 生成 Machine IR。

### Triton IR Lowering 通过 LLVM Machine IR

```
Triton Language (Python)
  ↓ @triton.jit
Triton IR     (tt.func, tt.load, tt.store...)
  ↓ Triton-Backend (C++)
LLVM IR       (标准 LLVM IR + NVPTX 目标三元组)
  ↓ LLVM NVPTX Backend
Machine IR    (NVPTX 特定的 MachineInstr 序列)
  ↓ MC Layer (MCCodeEmitter → PTX)
PTX Text      (最终 GPU 汇编)
  ↓ ptxas / CUDA Driver
SASS (GPU 机器码)
```

关键的连接点：
- Triton 的 `tt.dot` 操作最终被 lower 为 NVPTX 后端的自定义 `SDNode` 表示
- GPU 寄存器分配：NVPTX 后端使用 `Register` / `RegisterClass` / `RegisterUnits` 来建模
  PTX 的虚拟寄存器文件（PTX 有无限虚拟寄存器，由 ptxas 做二次分配）
- Triton 的 shared memory 操作经过 NVPTX 的 Machine IR，
  使用 `MachineFrameInfo` 和 `MachineMemOperand` 跟踪内存访问模式

### GPU Register 建模的工业实践

GPU 寄存器文件与 CPU 有本质区别：

```
CPU 寄存器 (AArch64):
  - 31 个 GPR (x0-x30) + 32 个 FPR (v0-v31)
  - 简单、扁平的寄存器文件
  - 少量 sub-register (w0 = x0 的低 32 位)

GPU 寄存器 (NVIDIA SM):
  - SM 级别的统一寄存器文件 (65536 个 32-bit 寄存器)
  - 按 warp 粒度分配 (每个 warp 最多 255 个寄存器)
  - 无硬件 sub-register 层次
  - 寄存器压力直接决定 occupancy
```

在 LLVM NVPTX 后端中：
- 使用 `RegisterTuples` 来分组寄存器向量（如 `.v2.b32` 类型）
- 使用 `RegisterUnits` 跟踪寄存器分配单元的可用性
- `MachineRegisterInfo` 管理 PTX 的虚拟寄存器（`%p` 前缀的谓词寄存器和 `%r` 前缀的通用寄存器）

**生产编译器经验**：对于 AI 加速器，寄存器建模需要在 Machine IR 层面解决以下问题：
1. **多级寄存器文件**：标量寄存器 + 向量寄存器 + 谓词寄存器 + 特殊寄存器（如 TensorCore 的 fragment）
2. **寄存器 bank 概念**：GlobalISel 引入的 `RegisterBank` 概念在这个场景中非常有用
3. **Memory space 与寄存器关联**：GPU 的 shared memory 在 PTX 层面是通过 `.shared` 状态空间访问，
   Machine IR 需要通过 `MachineMemOperand` 携带地址空间信息

## 示例说明

### 示例 1：分析一段 MIR

```yaml
name: relu_kernel
body: |
  bb.0.entry:
    liveins: $r0                              # 输入指针在 r0
    %0:gpr32 = COPY $r0                       # 复制到虚拟寄存器
    %1:gpr32 = LDRWui %0, 0                   # 从指针加载 32-bit 值
    %2:gpr32 = MOVi32imm 0                    # 常量 0
    %3:gpr32 = CMPWrr %1, %2                  # 比较
    %4:gpr32 = CSELWr %1, %2, 3, implicit $nzcv  # 条件选择 (max(x,0))
    STRWui %4, %0, 0                          # 存回内存
    RET_ReallyLR                              # 返回
```

从这段 MIR 可以看出：
- 这是一个 ReLU kernel 的 IR（`max(x, 0)` 的逻辑）
- `CSELWr` 是条件选择，使用 `$nzcv`（NZCV flags）作为隐式操作数
- `implicit $nzcv` 是典型的硬件约束建模——比较指令隐式设置 flags

### 示例 2：Triton 的 PTX 生成路径中的 Machine IR

Triton 编译器内部用 C++ 直接生成 LLVM IR（不经过 MLIR），然后调用 LLVM 的 NVPTX 后端。
关键 Machine IR 阶段包括：

```
LLVM IR:     %sum = call i32 @llvm.nvvm.lg2.approx.f(i32 %x)
             ↓ (NVPTX DAGToDAGISel)
Machine IR:  %v0:Int32Regs = LG2_approx_i32_only %v1
```

这里 `LLVM IR` 中的 intrinsic `@llvm.nvvm.lg2.approx.f` 被 NVPTX 的 SDISel
匹配为特定的 `NVPTXISD::LG2` SDNode，再选择为 PTX 指令 `lg2.approx.f32`。

### 示例 3：自定义后端的基本流程

```
1. 创建 llvm/lib/Target/MyTarget/ 目录
2. 编写 MyTargetRegisterInfo.td（寄存器描述）
3. 编写 MyTargetInstrInfo.td（指令描述）
4. 编写 CMakeLists.txt，添加 tablegen() 调用
5. 创建 MCTargetDesc 目录，实现 MCRegisterInfo 注册
6. 创建 MyTargetRegisterInfo.cpp（TargetRegisterInfo 子类）
```

## 总结

Machine IR 是 LLVM 后端的核心 IR，位于 LLVM IR 和汇编代码之间。它的关键特性包括：
- **平展指令表示**：所有指令是 MachineInstr 对象，通过 opcode + MCInstrDesc 区分语义
- **操作数类型多样**：寄存器、立即数、符号、寄存器掩码、栈帧索引等
- **隐式操作数建模**：硬件约束（如 flags 寄存器）和 ABI 约束（如 call 的 sp/lr）
- **灵活的寄存器层次**：sub-register、register tuples、register units 支持复杂硬件
- **SSA ↔ 非 SSA 转换**：Machine IR 可以自由转换，SSA 状态由虚拟寄存器定义次数决定
- **TableGen 驱动**：寄存器、指令的描述通过 .td 文件声明，TableGen backend 生成 C++ boilerplate

**与 AI 编译器的关系**：
- MLIR 的 `llvm` dialect 对应 LLVM IR 级别，Machine IR 是更低的抽象层
- Triton 编译器生成的 LLVM IR 最终通过 LLVM 的 Machine IR pipeline 生成 PTX
- GPU 的复杂寄存器文件（统一寄存器文件 + memory space）需要在 Machine IR 层面精确建模
- Register Units 和 Pressure Sets 是 GPU occupancy 优化的底层机制
- 理解 Machine IR 的约束系统（tied operands, early clobber, sub-register aliasing）是
  编写正确 GPU backend 的前提
