# Chapter 12: Getting Started with the Machine Code Layer

## 核心概念（详细展开）

### MC Layer 的定位

MC (Machine Code) layer 是 LLVM 后端基础设施中的 **最低抽象层**。它将指令建模为汇编级别的概念：
- 文本表示（助记符 + 操作数）
- 二进制编码（位序列）
- 汇编/反汇编机制

MC layer 和 Machine layer 之间构成近一対一映射：

| MC Layer | Machine Layer | 职责 |
|----------|--------------|------|
| `MCInst` | `MachineInstr` | 单条指令表示 |
| `MCOperand` | `MachineOperand` | 操作数表示 |
| `MCInstrDesc` | (共享) | 指令属性/约束描述 |
| `MCInstrInfo` | `TargetInstrInfo` | 指令信息查询 |
| `MCRegisterInfo` | `TargetRegisterInfo` | 寄存器信息 |
| `MCRegisterClass` | `TargetRegisterClass` | 寄存器类分组 |
| `MCRegister` | `Register` | 物理寄存器表示 |
| `MCAsmInfo` | — | 汇编语法约定 |
| `MCInstPrinter` | — | 文本打印 |
| `MCCodeEmitter` | — | 二进制编码 |
| `MCAsmParser` | — | 文本解析 |
| `MCSubtargetInfo` | `TargetSubtargetInfo` | 子目标特性信息 |

**工业视角**：MC layer 是 "纯数据" 层——它不参与优化决策，只管 "怎么表示" 和 "怎么编码"。
Machine layer 在 MC layer 之上添加优化逻辑（如 `getCommonSubClass` 用于寻找编码约束交集）。

### MC Layer 在编译流程中的位置

```
Machine IR (MachineInstr)
       ↓ MCInstLower::Lower()
MC 表示 (MCInst)
       ↓                    ↘
MCInstPrinter              MCCodeEmitter
(文本输出)                  (二进制输出)
       ↓                    ↓
汇编文件 (.s)            目标文件 (.o)

# 反向路径（汇编器/反汇编器）
汇编文件 (.s)            目标文件 (.o)
       ↓                    ↓
MCAsmParser              MCDisassembler
       ↓                    ↓
MC 表示 (MCInst)       MC 表示 (MCInst)
```

**关键洞察**：MC 是唯一同时面向两种输出的 IR：
- **文本输出**（assembly）：通过 MCInstPrinter
- **二进制输出**（object file）：通过 MCCodeEmitter
这种设计使得汇编器和反汇编器可以共享相同的 MC 层代码。

## LLVM / MLIR 流程（深入）

### MLIR 到 LLVM IR 的 translation（不是 MC！）

MLIR 的 `mlir-translate` 工具将 MLIR 的 `llvm` dialect 模块转换为 LLVM IR 模块：

```
MLIR Module (llvm dialect)
       ↓ mlir::translateModuleToLLVMIR()
LLVM Module (标准 LLVM IR)
       ↓ LLVM CodeGen Pipeline
Machine IR (MachineFunction)
       ↓ MC Layer
汇编 / 目标文件
```

**重点**：MLIR 的 `llvm` dialect 到 LLVM IR 的转换是 **模块级翻译**，不涉及 Machine IR 或 MC。
只有当 LLVM IR 进入后端编译时，MC layer 才参与工作。

但 MLIR 同样使用 LLVM 的 MC 层基础设施：
- `mlir::LLVM::Translation` 内部调用 LLVM 的 PassManagerBuilder 构建完整的后端 pipeline
- 最终通过 LLVM 的 `TargetMachine::addPassesToEmitFile()` 输出汇编/目标文件
- 这意味着 MLIR 产生的代码也完全受 LLVM MC layer 的支持

### Triton 通过 MC Layer 生成 PTX 的详细路径

```
Triton IR
  ↓ Triton-Backend C++ code generation
LLVM IR (Module, 带 NVPTX triple)
  ↓ LLVM NVPTX TargetMachine::addPassesToEmitFile()
  1. NVPTX DAGToDAGISel (SDISel): LLVM IR → NVPTX MachineInstr
  2. NVPTX Machine Passes: 机器级优化
  3. NVPTX MCInstLower: MachineInstr → MCInst
  4. NVPTX MCCodeEmitter: MCInst → PTX 文本字节流
PTX Source Text
  ↓ ptxas (NVIDIA 专有工具)
SASS (GPU 机器码)
```

在 Triton 中，PTX 输出不经过 MCAsmParser（因为不需要汇编器）。
Triton 使用 LLVM 的 `NVPTX MCCodeEmitter` 直接将 MCInst 编码为 PTX 文本指令。
这利用了 PTX 的特殊性——PTX 是 JIT 编译的虚拟 ISA，其 "编码" 实际上是文本格式。

**Triton 的特殊处理**：
```cpp
// Triton 内部：调用 LLVM 后端生成 PTX
void NVPTXTargetMachine::emitAssembly(
    Module &M, raw_pwrite_stream &OS) {
    // ... 运行 Machine pass pipeline ...
    // 最后通过 NVPTXAsmPrinter 输出 PTX 文本
    NVPTXAsmPrinter AP(OS, *this, ...);
    AP.runOnMachineFunction(MF);
}
```

### CUDA 二进制生成 pipeline（工业全貌）

```
.cu 文件
  ↓ nvcc / clang (CUDA frontend)
LLVM IR (device code, NVPTX triple)
  ↓ 路径 A: 生成 PTX 文本
  LLVM NVPTX backend → PTX text (.ptx)
  ↓ ptxas
  cubin (SASS 二进制)
  ↓ fatbin 包装
  fatbin (嵌入最终可执行文件)

  ↓ 路径 B: 生成 LLVM bitcode (用于 "fatbinary" 多架构支持)
  LLVM IR bitcode (.bc)
  ↓ 运行时 JIT (CUDA driver / NVRTC)
  PTX text → SASS
```

**生产编译器关键点**：
- CUDA 11+ 支持直接在设备代码中嵌入多个架构的 fatbin
- LLVM 的 MC layer 在两条路径中都参与 PTX 文本生成
- 对于自定义 AI 加速器，MC layer 需要支持：
  1. 自定义的指令编码格式（可能与标准 ISA 不同）
  2. 多级指令集（pseudo-instructions → real instructions 的 expansion pass）
  3. Relaxation（某些指令的编码大小依赖最终地址偏移）

## 关键机制解析（工业视角）

### MCInstrDesc - 指令的完整属性契约

每条指令的 opcode 映射到一个 **MCInstrDesc** 对象：

```cpp
class MCInstrDesc {
    // 静态属性
    uint64_t Flags;           // mayLoad, mayStore, isBranch, isReturn...
    uint8_t NumOperands;     // 静态操作数数量（可以更多，不能更少）
    uint16_t NumDefs;        // 定义操作数的数量
    uint16_t Size;           // 指令字节大小

    // 操作数信息
    const MCOperandInfo *OpInfo; // 每个操作数的类型/约束
    const uint16_t *ImplicitUses; // 隐式使用的寄存器列表
    const uint16_t *ImplicitDefs; // 隐式定义的寄存器列表
};
```

**操作数约束**（编码在 MCOperandInfo 中）：
- `MCOI::OPERAND_REGISTER`：寄存器操作数
- `MCOI::OPERAND_IMMEDIATE`：立即数操作数
- `MCOI::TIED_TO`：绑定操作数约束
- `MCOI::EARLY_CLOBBER`：提前覆盖约束

### MCAsmInfo - 汇编语法约定的抽象

每个目标文件格式需要一个 MCAsmInfo 子类：

```cpp
class MyTargetMCAsmInfoELF : public MCAsmInfoELF {
public:
    explicit MyTargetMCAsmInfoELF(const Triple &TT,
                                   const MCTargetOptions &Options) {
        CommentString = "#";
        ZeroDirective = "\t.zero\t";
        AsciiDirective = "\t.ascii\t";
        Data8bitsDirective = "\t.byte\t";
        Data32bitsDirective = "\t.word\t";
        // ... 其他汇编器指令约定 ...
    }
};
```

注册方式：
```cpp
static MCAsmInfo *createMyTargetMCAsmInfo(const MCRegisterInfo &MRI,
                                           const Triple &TT,
                                           const MCTargetOptions &Opts) {
    MCAsmInfo *MAI;
    if (TT.isOSBinFormatELF())
        MAI = new MyTargetMCAsmInfoELF(TT, Opts);
    else
        report_fatal_error("Unsupported binary format");
    return MAI;
}
// 在 Target 的 MC 初始化函数中注册
RegisterMCAsmInfoFn X(TheTarget, createMyTargetMCAsmInfo);
```

### 指令编码 - TableGen 方式

```tablegen
// 32-bit 指令编码
def ADD32rr : Instruction {
    let OutOperandList = (outs GPR32:$rd);
    let InOperandList = (ins GPR32:$rs1, GPR32:$rs2);
    let AsmString = "add $rd, $rs1, $rs2";

    // 编码：bits<32> Inst 字段
    bits<32> Inst;

    // 固定 opcode 位 (31-22)
    let Inst{31-22} = 0b1100100111;

    // 操作数占位符：变量名必须与 ins/outs 中一致
    bits<5> rd;
    let Inst{21-17} = rd;

    bits<5> rs1;
    let Inst{16-12} = rs1;

    bits<5> rs2;
    let Inst{11-7} = rs2;

    // 剩余的位填 0（固定为 0）
    let Inst{6-0} = 0b0000000;
}
```

**关键注意事项**：
- `Inst{31-22}` 使用递减索引（从高位到低位），匹配 ISA 手册的编写方式
- `Inst{0-3}` vs `Inst{3-0}` 含义不同——前者将数值的最低有效位放在索引 0
- 操作数变量名必须与 `outs`/`ins` 中的 `$variableName` 一致
- 伪指令（PHI, COPY 等）和 `isCodeGenOnly` 指令不需要编码信息

### MCCodeEmitter - 从 MCInst 到字节序列

核心实现接口：

```cpp
class MyTargetMCCodeEmitter : public MCCodeEmitter {
public:
    // 主编码入口
    void encodeInstruction(const MCInst &MI,
                           SmallVectorImpl<char> &CB,
                           SmallVectorImpl<MCFixup> &Fixups,
                           const MCSubtargetInfo &STI) const override;

    // 编码单个操作数（由 TableGen 生成的代码调用）
    uint64_t getMachineOpValue(const MCInst &MI,
                                const MCOperand &MO,
                                SmallVectorImpl<MCFixup> &Fixups,
                                const MCSubtargetInfo &STI) const;
};

// 典型实现（32-bit 小端指令）
void MyTargetMCCodeEmitter::encodeInstruction(
    const MCInst &MI, SmallVectorImpl<char> &CB,
    SmallVectorImpl<MCFixup> &Fixups,
    const MCSubtargetInfo &STI) const {
    // TableGen 生成的方法计算完整编码
    uint64_t Encoding = getBinaryCodeForInstr(MI, Fixups, STI);
    // 按小端字节序写入
    support::endian::write<uint32_t>(CB, Encoding,
                                      llvm::endianness::little);
}

unsigned MyTargetMCCodeEmitter::getMachineOpValue(
    const MCInst &MI, const MCOperand &MO,
    SmallVectorImpl<MCFixup> &Fixups,
    const MCSubtargetInfo &STI) const {
    if (MO.isReg())
        return getRegisterEncoding(MO.getReg()); // 查表获取 HW 编码
    if (MO.isImm())
        return static_cast<unsigned>(MO.getImm());
    return 0;
}
```

### MCInstPrinter - 文本汇编输出

```cpp
class MyTargetInstPrinter : public MCInstPrinter {
public:
    void printInst(const MCInst *MI, uint64_t Address,
                   StringRef Annot, const MCSubtargetInfo &STI,
                   raw_ostream &O) override {
        // 尝试别名打印，失败则用标准格式
        if (!printAliasInstr(MI, Address, O))
            printInstruction(MI, Address, O);  // TableGen 生成
        printAnnotation(O, Annot);
    }

    void printOperand(const MCInst *MI, unsigned OpNo,
                      raw_ostream &O) {
        const MCOperand &Op = MI->getOperand(OpNo);
        if (Op.isReg())
            O << getRegisterName(Op.getReg());  // TableGen 生成
        else if (Op.isImm())
            O << formatImm(Op.getImm());
        else
            Op.getExpr()->print(O, &MAI);
    }

    // TableGen 要求声明的辅助方法
    static const char *getRegisterName(MCRegister Reg);
    std::pair<const char *, uint64_t> getMnemonic(const MCInst &MI) override;
};
```

### AsmParser - 文本输入到 MCInst

汇编解析器是最具目标特定性的部分，LLVM 仅提供基础框架：

```cpp
class MyTargetAsmParser : public MCTargetAsmParser {
    bool MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                                  OperandVector &Operands,
                                  MCStreamer &Out,
                                  uint64_t &ErrorInfo,
                                  bool MatchingInlineAsm) override;
    bool ParseRegister(unsigned &RegNo,
                        SMLoc &StartLoc, SMLoc &EndLoc) override;
    bool ParseInstruction(ParseInstructionInfo &Info,
                           StringRef Name, SMLoc NameLoc,
                           OperandVector &Operands) override;
};
```

汇编器的主要挑战在于处理：
1. 多种操作数语法（寄存器、立即数、内存寻址模式）
2. 指令别名（如 `mov` 可能是多种编码的快捷方式）
3. 伪指令和宏展开

### MCInstLower - Machine IR 到 MC 的桥梁

连接 Machine layer 和 MC layer 的关键组件：

```cpp
class MyTargetMCInstLower {
public:
    void Lower(const MachineInstr *MI, MCInst &OutMI) const {
        OutMI.setOpcode(MI->getOpcode());
        for (const MachineOperand &MO : MI->operands()) {
            MCOperand MCOp;
            switch (MO.getType()) {
            case MachineOperand::MO_Register:
                if (!MO.isImplicit())
                    MCOp = MCOperand::createReg(MO.getReg());
                break;
            case MachineOperand::MO_Immediate:
                MCOp = MCOperand::createImm(MO.getImm());
                break;
            case MachineOperand::MO_GlobalAddress:
                MCOp = LowerSymbolOperand(MO, ...);
                break;
            // ... 其他操作数类型 ...
            }
            OutMI.addOperand(MCOp);
        }
    }
};
```

## AI 编译器关联

### MLIR 到 LLVM IR Translation（工具链集成）

```
IREE / Triton MLIR 流程:
┌─────────────────────────────────────────┐
│ MLIR Pipeline                            │
│  stablehlo → linalg → gpu → llvm dialect │
└──────────────────┬──────────────────────┘
                   ↓ mlir-translate --mlir-to-llvmir
┌─────────────────────────────────────────┐
│ LLVM IR Module (标准 LLVM IR)            │
└──────────────────┬──────────────────────┘
                   ↓ llc / clang (CodeGen)
┌─────────────────────────────────────────┐
│ LLVM Backend Pipeline                    │
│  1. SDISel/GlobalISel → MachineInstr     │
│  2. Machine Passes → 优化 MachineInstr    │
│  3. MCInstLower → MCInst                 │
│  4. MCCodeEmitter / MCInstPrinter → 输出 │
└──────────────────┬──────────────────────┘
                   ↓
               PTX / 汇编 / 目标文件
```

**工业洞察**：MLIR 的 `mlir-translate` 工具只做模块级别的翻译（IR 级别的 lowering），
实际的指令选择、寄存器分配、指令编码全部发生在 LLVM 的 CodeGen pipeline 中。
这意味着 MLIR 的输出质量仍然依赖 LLVM 后端的质量。

### Triton 的 PTX Emitting 与 MC Layer

Triton 编译器生成 PTX 时，不走 MCInstPrinter 的文本路径，而是直接使用 MCCodeEmitter：

```cpp
// Triton 内部 NVPTX 代码生成流程（简化）
class TritonNVPTXMCInstLower {
    // 将 Triton 的 IR 操作映射到 NVPTX MachineInstr
    void lower_tt_dot(const DotOp &op, MachineIRBuilder &builder) {
        // tt.dot → NVPTX::MMA (Matrix Multiply-Accumulate)
        auto MIB = builder.buildInstr(NVPTX::MMA_F32);
        MIB.addReg(dst).addReg(srcA).addReg(srcB).addReg(acc);
    }
};

// NVPTX AsmPrinter 最终输出 PTX 文本
void NVPTXAsmPrinter::emitInstruction(const MachineInstr *MI) {
    // 对于 PTX，emitInstruction 直接输出 PTX 文本
    // （不经过二进制编码，因为 PTX 是虚拟 ISA）
    MCInst TmpInst;
    lowerToMCInst(*MI, TmpInst);
    EmitToStreamer(*OutStreamer, TmpInst);
}
```

### CUDA PTX 生成的完整 Pipeline

```
Triton Kernel → LLVM IR (NVPTX)
  ↓ NVPTX Target Pass Pipeline
NVPTX ISel (DAGToDAG):
  - llvm.nvvm.barrier0 → NVPTX::BAR_SYNC
  - llvm.nvvm.shfl.sync.bfly → NVPTX::SHFL_BFLY
  - tt.dot (via call) → NVPTX::MMA_* (TensorCore)
  ↓
NVPTX MachineInstr:
  %vreg0:int32regs = LDG_i32 %ptr  # global load
  %vreg1:int32regs = LDS_i32 %sptr  # shared load
  %vreg2:float32regs = MMA_F32 %vreg0, %vreg1, %acc
  ↓ NVPTX MCInstLower
NVPTX MCInst:
  ld.global.ca.b32 %r0, [%rd1];
  ld.shared.b32 %r1, [%rd2];
  mma.sync.aligned.m16n8k8.row.col.f32.f32.f32.f32 ...
  ↓ NVPTX AsmPrinter (直接文本输出，不走 MCCodeEmitter)
PTX 文本字符串
```

**生产经验**：
- PTX 与其他 ISA 不同，MCInst 到 PTX 文本是直接格式化输出，不经过二进制编码
- 这就是为什么 NVPTX MCCodeEmitter 的实现非常薄——PTX 本质上是文本格式
- 对于自定义 AI 加速器使用 MC Layer 时，同样需要考虑：你的输出是文本还是二进制？
  - 文本 → 重载 MCInstPrinter
  - 二进制 → 重载 MCCodeEmitter

### 自定义 AI 加速器的汇编生成策略

对于自研 AI 加速器，有三种方式使用 LLVM MC layer：

**方案 A：生成标准汇编文件（.s）**
```
LLVM IR → Machine IR → MCInst → MCInstPrinter → 自定义 .s 格式
```
与 GCC 兼容的汇编器输出，便于与现有工具链集成。

**方案 B：直接生成二进制**
```
LLVM IR → Machine IR → MCInst → MCCodeEmitter → 二进制 .o 文件
```
适合没有汇编器的场景，直接输出可加载的机器码。

**方案 C：生成文本中间表示（类似 PTX）**
```
LLVM IR → Machine IR → MCInst → 自定义文本格式
```
类似 CUDA PTX 模式：生成设备无关的文本表示，由 runtime 做最终编译。
这是最灵活的方案，也是 AI 编译器中最常见的方案（Triton、IREE 都是这种模式）。

## 示例说明

### 示例 1：TableGen 描述一条完整的指令

```tablegen
// ALU 指令类的基类（共享编码格式）
class ALU_RR<bits<10> opcode, string asmStr, dag outs, dag ins>
    : Instruction {
    let OutOperandList = outs;
    let InOperandList = ins;
    let AsmString = asmStr;
    let Size = 4;  // 32-bit 指令

    bits<32> Inst;
    let Inst{31-22} = opcode;
    bits<5> rd;
    let Inst{21-17} = rd;
    bits<5> rs1;
    let Inst{16-12} = rs1;
    bits<5> rs2;
    let Inst{11-7} = rs2;
    let Inst{6-0} = 0b0000000;
}

// 具体指令实例
def ADD  : ALU_RR<0b0000000000, "add  $rd, $rs1, $rs2",
                   (outs GPR32:$rd), (ins GPR32:$rs1, GPR32:$rs2)>;
def SUB  : ALU_RR<0b0000000001, "sub  $rd, $rs1, $rs2",
                   (outs GPR32:$rd), (ins GPR32:$rs1, GPR32:$rs2)>;
def AND  : ALU_RR<0b0000000010, "and  $rd, $rs1, $rs2",
                   (outs GPR32:$rd), (ins GPR32:$rs1, GPR32:$rs2)>;
```

### 示例 2：使用 llvm-mc 测试编码

```bash
# test.s
.text
add r1, r2, r3
sub r4, r5, r6

# 汇编并显示编码
$ llvm-mc -triple=mycpu -show-encoding test.s
    add r1, r2, r3    # encoding: [0x00,0x00,0x88,0x02]
    sub r4, r5, r6    # encoding: [0x01,0x00,0x18,0x03]
```

### 示例 3：MC Layer 组件连接全貌

```
TableGen 输入:
  XXXRegisterInfo.td  (寄存器 AsmName + HwEncoding)
  XXXInstrInfo.td     (指令 AsmString + Inst 编码)
       ↓ TableGen backends
XXXGenRegisterInfo.inc
  → GET_REGINFO_ENUM        (Register enum)
  → GET_REGINFO_MC_DESC     (InitXXXMCRegisterInfo)
XXXGenInstrInfo.inc
  → GET_INSTRINFO_ENUM      (Opcode enum)
  → GET_INSTRINFO_MC_DESC   (InitXXXMCInstrInfo)
  → GET_INSTRINFO_EMITTER   (getBinaryCodeForInstr)
       ↓ C++ 实现
MCTargetDesc/XXXMCTargetDesc.cpp:
  → RegisterMCAsmInfoFn      (createXXXMCAsmInfo)
  → RegisterMCRegInfoFn       (createXXXMCRegisterInfo)
  → RegisterMCInstrInfoFn     (createXXXMCInstrInfo)
  → RegisterMCInstPrinterFn   (createXXXMCInstPrinter)
  → RegisterMCCodeEmitterFn   (createXXXMCCodeEmitter)
       ↓
LLVMInitializeXXXTargetMC() → 将所有组件注册到 TargetRegistry
```

## 总结

MC (Machine Code) layer 是 LLVM 后端的最低抽象层，建模了汇编层面的概念：
- **MCInst / MCOperand**：纯数据层面的指令和操作数表示
- **MCInstrDesc**：指令的完整属性契约（标志位、操作数类型、隐式操作数）
- **MCAsmInfo**：目标特定的汇编语法约定（注释符、段指令、数据指令等）
- **AsmString（TableGen）**：用 `$variableName` 占位符描述汇编语法字符串
- **指令编码（TableGen）**：用 `bits<N> Inst` 字段按位描述编码，操作数变量自动关联
- **MCCodeEmitter**：将 MCInst 编码为字节序列（二进制输出）
- **MCInstPrinter**：将 MCInst 格式化为人类可读文本（汇编输出）
- **MCAsmParser**：将汇编文本解析为 MCInst（汇编器输入）
- **MCInstLower**：Machine IR → MC 的桥梁（MachineInstr → MCInst）
- **三个 TableGen backend**：gen-register-info、gen-instr-info、gen-emitter 生成大量样板代码

**与 AI 编译器的关系**：
- MLIR 的 `llvm` dialect 翻译为 LLVM IR 后，完整走 LLVM CodeGen pipeline（包括 MC layer）输出目标代码
- Triton 编译器利用 NVPTX 的 AsmPrinter（MC layer 的一部分）直接生成 PTX 文本
- 对于自定义 AI 加速器，有三种 MC 输出策略：标准汇编文件、直接二进制、文本中间表示（类似 PTX）
- MC layer 是 LLVM 工具链（llvm-mc, llvm-mca, llvm-objdump）的基础，这些工具在 AI 编译器开发中
  用于验证指令编码和调试代码生成
- 理解 MC layer 是实现自定义 AI 加速器汇编输出的必要前提
