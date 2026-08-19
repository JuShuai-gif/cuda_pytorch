# Chapter 21: Getting Started with the Assembler

> **From the perspective of a production AI compiler engineer who needs to understand LLVM deeply to work on MLIR/Triton/AI compiler stacks.**

## 核心概念（详细展开）

### 汇编器在编译器栈中的位置

汇编器（Assembler）是 LLVM 后端的最终阶段，负责将 Machine IR（寄存器分配、栈布局已完成）转换为可执行格式：

```
前端 → LLVM IR → Selection → Scheduling → RA → Stack Lowering → AsmPrinter → MC Layer → Object File
                                                                          ↑
                                                                    本章焦点
```

MC（Machine Code）层的完整能力：

```
                    写入方向（编译器输出）        读取方向（汇编器输入）
                    ─────────────────→         ←─────────────────
                    MCInst → MCCodeEmitter      MCAsmParser → MCInst
                       ↓                              ↓
                    MCObjectStreamer             MCObjectStreamer
                       ↓                              ↓
                    MCAssembler                 MCAssembler
                       ↓                              ↓
                    Object File (.o/.obj)       Object File (.o/.obj)
                    (binary encoding)           (from textual assembly)

                    MCInst → MCInstPrinter
                                ↓
                          Assembly Text (.s)
```

**为什么汇编器对 AI 编译器至关重要：**

1. **MLIR 到最终可执行格式**：MLIR 通过 `ConvertToLLVMDialect` → LLVM AsmPrinter → MC Layer 生成目标代码
2. **PTX 汇编和 cubin 生成**：Triton 编译器生成 PTX（虚拟 ISA），然后 ptxas 将其汇编为 cubin（SASS 本机代码）
3. **IREE 可执行格式**：IREE 支持多种目标后端（Vulkan/SPIR-V、CUDA/PTX、CPU/ELF），每种都需要自己的"汇编器"
4. **函数调用重定位（relocation）**：AI 模型中大量的函数调用（kernel launch、library call）需要正确的重定位信息

### MC Layer 架构

MC Layer 是 LLVM 用来处理机器码的**独立库**（可用于 codegen 和独立的 assembler/disassembler）：

```
┌───────────────────────────────────────────────────────────────┐
│                        MC Layer                               │
│                                                               │
│  Core Classes:                                                │
│    MCInst          - 轻量级单条指令表示                         │
│    MCOperand       - 操作数（reg/imm/expr）                   │
│    MCInstrDesc     - 指令描述元数据                            │
│    MCInstrInfo     - 所有指令的查找表                          │
│    MCRegisterInfo  - MC 级别的寄存器信息                       │
│    MCSubtargetInfo - 子目标特性（用于编码决策）                 │
│                                                               │
│  Streamer Classes:                                            │
│    MCStreamer      - 抽象输出流（emit instruction/label/data）  │
│    MCAsmStreamer   - 输出为文本汇编                            │
│    MCObjectStreamer- 输出为二进制目标文件                      │
│    MCNullStreamer  - 空操作（/dev/null）                      │
│                                                               │
│  Assembler Classes:                                           │
│    MCAssembler     - 完整汇编器（fixup 解析、layout、relaxation）│
│    MCAsmBackend    - 目标特定的汇编后端                         │
│    MCCodeEmitter   - 将 MCInst 编码为二进制字节                 │
│    MCObjectWriter  - 写入目标文件格式（ELF/MachO/COFF）          │
└───────────────────────────────────────────────────────────────┘
```

## LLVM / MLIR 流程（深入）

### AsmPrinter：从 MachineInstr 到 MCInst

`AsmPrinter` 是将 MachineFunction（已完成寄存器分配）转换为 MC 层构造的桥梁：

```cpp
class MyAsmPrinter : public AsmPrinter {
public:
  // 转换每个 MachineInstr → MCInst
  void emitInstruction(const MachineInstr *MI) override;

  // 处理整个 MachineFunction
  bool runOnMachineFunction(MachineFunction &MF) override;

  // 发射常量池、跳转表等
  void emitConstantPool() override;
  void emitJumpTableInfo() override;

  // 函数级别构造
  void emitFunctionHeader() override;
  void emitFunctionBody() override;
  void emitFunctionFooter() override;
};
```

#### MachineInstr → MCInst 的降低

```cpp
void MyAsmPrinter::emitInstruction(const MachineInstr *MI) {
  MCInst TmpInst;

  // 方式 1：使用 TableGen 生成的 lowering（推荐）
  lowerMyInstrToMCInst(MI, TmpInst, *this);

  // 方式 2：手动构建 MCInst
  // TmpInst.setOpcode(MyTarget::ADD32);
  // TmpInst.addOperand(MCOperand::createReg(MI->getOperand(0).getReg()));
  // ...

  // 通过 Streamer 发射
  EmitToStreamer(*OutStreamer, TmpInst);
}
```

### MCCodeEmitter：二进制编码

`MCCodeEmitter` 将 MCInst 转换为字节序列：

```cpp
class MyMCCodeEmitter : public MCCodeEmitter {
public:
  void encodeInstruction(const MCInst &MI,
                         SmallVectorImpl<char> &CB,
                         SmallVectorImpl<MCFixup> &Fixups,
                         const MCSubtargetInfo &STI) const override {
    // 1. 获取基础编码（来自 TableGen 生成）
    uint64_t Bits = getBinaryCodeForInstr(MI, Fixups, STI);

    // 2. 编码每个操作数
    for (unsigned i = 0; i < MI.getNumOperands(); ++i) {
      const MCOperand &MO = MI.getOperand(i);
      if (MO.isReg()) {
        Bits |= encodeRegister(MO.getReg()) << getRegOffset(i);
      } else if (MO.isImm()) {
        Bits |= encodeImmediate(MO.getImm()) << getImmOffset(i);
      } else if (MO.isExpr()) {
        // 表达式需要 fixup（由链接器解析）
        Fixups.push_back(MCFixup::create(
            getFixupOffset(i), MO.getExpr(),
            getFixupKindForOperand(i)));
      }
    }

    // 3. 写入字节（按目标字节序）
    support::endian::write(CB, Bits,
        STI.getTargetTriple().isLittleEndian() ? support::little
                                                : support::big);
  }
};
```

### Fixups 和 Relocations

Fixups 代表汇编时未知的值引用（由链接器解析）：

```
Fixup 的生命周期:
  1. MCCodeEmitter 创建 MCFixup（编码指令时）
  2. MCAssembler 保存 MCFixup（布局时）
  3. MCAsmBackend::applyFixup 尝试解析（如果可能）
  4. 如果无法解析 → MCObjectWriter::recordRelocation 记录重定位

Fixup → Relocation 的转换条件:
  - Fixup 引用了一个外部符号（跨 object file）
  - Fixup 的值在当前上下文中无法计算
```

#### 定义 Fixup Kinds

```cpp
namespace MyTarget {
enum Fixups {
  fixup_my_32 = FirstTargetFixupKind,       // 32-bit absolute
  fixup_my_16_pcrel,                         // 16-bit PC-relative
  fixup_my_hi16,                             // upper 16 bits
  fixup_my_lo16,                             // lower 16 bits
  fixup_my_got_pcrel,                        // GOT PC-relative
  // ...
  LastTargetFixupKind,
  NumTargetFixupKinds = LastTargetFixupKind - FirstTargetFixupKind
};
}
```

#### MCAsmBackend：应用 Fixups 和管理 Relaxation

```cpp
class MyAsmBackend : public MCAsmBackend {
public:
  // 应用 fixup（将值写回到字节缓冲区）
  void applyFixup(const MCAssembler &Asm, const MCFixup &Fixup,
                  const MCValue &Target, MutableArrayRef<char> Data,
                  uint64_t Value, bool IsResolved,
                  const MCSubtargetInfo *STI) const override {
    switch (Fixup.getKind()) {
    case MyTarget::fixup_my_32:
      support::endian::write32le(Data.data() + Fixup.getOffset(), Value);
      break;
    case MyTarget::fixup_my_16_pcrel:
      // PC-relative: Value = target - address_of_fixup
      uint64_t Adjusted = Value - Fixup.getOffset();
      support::endian::write16le(Data.data() + Fixup.getOffset(), Adjusted);
      break;
    }
  }

  // 是否需要 relaxation（指令可能有长/短形式）
  bool mayNeedRelaxation(const MCInst &Inst,
                          const MCSubtargetInfo &STI) const override;

  // Relax 指令（长 → 短 或 短 → 长）
  bool relaxInstruction(const MCInst &Inst, const MCSubtargetInfo &STI,
                         MCInst &Res) const override;
};
```

### MCObjectWriter 和目标文件格式

```cpp
class MyELFObjectWriter : public MCObjectTargetWriter {
public:
  Triple::ObjectFormatType getFormat() const override {
    return Triple::ELF;
  }

  // 写入目标文件
  void writeObject(MCAssembler &Asm, const MCAsmLayout &Layout) override {
    // 写入 ELF header
    // 写入 section headers
    // 写入符号表
    // 写入重定位表
    // 写入 section 内容
  }

  // 记录重定位信息
  bool recordRelocation(MCAssembler &Asm, const MCAsmLayout &Layout,
                        const MCFragment *Fragment, const MCFixup &Fixup,
                        MCValue Target, uint64_t &FixedValue) override;
};
```

### TableGen 生成的 MC 组件

大多数 MC 层代码由 TableGen 自动生成：

| TableGen Backend | 输出文件 | 生成内容 |
|-----------------|---------|---------|
| `gen-instr-info` | `XXXGenInstrInfo.inc` | MCInstrInfo 查找表 |
| `gen-register-info` | `XXXGenRegisterInfo.inc` | MCRegisterInfo 表 |
| `gen-subtarget` | `XXXGenSubtargetInfo.inc` | MCSubtargetInfo（包含调度模型） |
| `gen-asm-writer` | `XXXGenAsmWriter.inc` | MCInstPrinter 自动实现 |
| `gen-asm-matcher` | `XXXGenAsmMatcher.inc` | MCAsmParser 模式匹配表 |
| `gen-disassembler` | `XXXGenDisassemblerTables.inc` | 反汇编表 |

**CMake 集成示例：**

```cmake
tablegen(LLVM MyGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM MyGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM MyGenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM MyGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM MyGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM MyGenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM MyGenMCCodeEmitter.inc -gen-emitter)
```

## 关键机制解析（工业视角）

### 完整汇编发射 Pipeline

```
MachineFunction (post stack lowering)
    │
    ▼
┌─────────────────────────────────────────┐
│  AsmPrinter::runOnMachineFunction       │
│    ├─ emitFunctionHeader (label, CFI)   │
│    ├─ emitConstantPool                  │
│    ├─ for each MBB:                     │
│    │    ├─ emitBasicBlockStart          │
│    │    └─ for each MachineInstr:       │
│    │         emitInstruction(MI)        │
│    │           → lowerMI_To_MCInst      │
│    │             → OutStreamer.emit()   │
│    └─ emitFunctionFooter                │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  MCStreamer (MCAsmStreamer or           │
│              MCObjectStreamer)          │
│    ├─ emitInstruction(MCInst)           │
│    ├─ emitLabel(MCSymbol)               │
│    └─ emitBytes(StringRef)              │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  MCAssembler (仅 binary output)         │
│    ├─ Layout (计算 fragments 偏移)        │
│    ├─ applyFixup (解析已确定的引用)       │
│    ├─ relaxInstruction (如果需要)        │
│    └─ writeObject (通过 MCObjectWriter)  │
└──────────────┬──────────────────────────┘
               │
               ▼
           Object File (.o / .obj)
           或 Assembly Text (.s)
```

### MCStreamer 的多态性

```cpp
// MCStreamer 抽象接口
class MCStreamer {
public:
  virtual void emitInstruction(const MCInst &Inst, const MCSubtargetInfo &STI);
  virtual void emitLabel(MCSymbol *Symbol);
  virtual void emitBytes(StringRef Data);
  virtual void emitValueToAlignment(unsigned ByteAlignment, ...);
  virtual void switchSection(MCSection *Section, ...);
  virtual void emitCFI* ();  // Call Frame Information
};

// MCObjectStreamer: emitInstruction → MCAssembler → object file
// MCAsmStreamer:   emitInstruction → MCInstPrinter → text output
// MCNullStreamer:  emitInstruction → /dev/null (for size estimation)
```

### 指令 Relaxation

指令可能有长/短形式，取决于偏移大小：

```
示例：条件分支
  BEQ offset
    - 短形式 (16-bit offset): branch range = ±32KB → 2 bytes encoding
    - 长形式 (32-bit offset): branch range = ±2GB → 4 bytes encoding

Relaxation 过程:
  1. 初始假设短形式（optimistic）
  2. Layout 计算实际偏移
  3. 如果偏移超出短形式范围 → 替换为长形式
  4. 因为指令变大了 → 其他指令的偏移改变 → 重新布局
  5. 迭代直到收敛或无法收敛（报错）
```

```cpp
bool MyAsmBackend::relaxInstruction(const MCInst &Inst,
                                     const MCSubtargetInfo &STI,
                                     MCInst &Res) const {
  // 检查是否需要长形式
  if (Inst.getOpcode() == MyTarget::BEQ_SHORT) {
    int64_t Offset = Inst.getOperand(1).getImm();
    if (Offset < -32768 || Offset > 32767) {
      // 替换为长形式
      Res.setOpcode(MyTarget::BEQ_LONG);
      Res.addOperand(Inst.getOperand(0));  // 条件码
      Res.addOperand(Inst.getOperand(1));  // 偏移（现在编码为 32-bit）
      return true;  // relaxed
    }
  }
  return false;  // 不需要 relaxation
}
```

## AI 编译器关联

### MLIR 到汇编的完整 Pipeline

MLIR 编译器通过 LLVM 后端生成最终目标代码：

```mlir
// 输入：MLIR（LLVM Dialect）
llvm.func @kernel(%arg0: !llvm.ptr<f32>, %arg1: i64) {
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.add %arg1, %0 : i64
  %2 = llvm.getelementptr %arg0[%1] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
  %3 = llvm.load %2 : !llvm.ptr<f32> -> f32
  llvm.store %3, %arg0 : f32, !llvm.ptr<f32>
  llvm.return
}

// Step 1: TranslateToLLVMIR → LLVM IR (in-memory Module)
// Step 2: LLVM backend pipeline (Selection → Scheduling → RA → StackLower → AsmPrinter)
// Step 3: MC Layer → Object File

// 最终 x86-64 汇编（示例）
kernel:
  lea (%rdi,%rsi,4), %rax
  movss (%rax), %xmm0
  movss %xmm0, (%rdi)
  ret
```

**MLIR 中控制目标格式：**

```cpp
// 将 MLIR Module 编译为目标文件
mlir::OwningOpRef<mlir::ModuleOp> module = ...;
mlir::registerLLVMDialectTranslation(*module->getContext());

// 创建 LLVM TargetMachine
auto targetMachine = ...; // 从 Triple 创建

// 生成目标文件
// → 内部调用 LLVM AsmPrinter + MC Layer
```

### PTX 汇编和 cubin 生成

Triton 编译器生成 PTX（Parallel Thread Execution），这是 NVIDIA 的虚拟 ISA：

```
Triton → MLIR → LLVM IR (nvvm dialect) → PTX (text) → ptxas → cubin (binary SASS)
                                    ↑              ↑
                              LLVM AsmPrinter    NVIDIA's assembler
                              (NVPTX backend)    (closed source)
```

**PTX 汇编的关键特性：**

```ptx
// PTX 示例 (虚拟 ISA，独立于具体 GPU 代数):
.visible .entry kernel_function(
    .param .u64 input_ptr,
    .param .u64 output_ptr
) {
    .reg .f32 %f<4>;          // 虚拟寄存器（无限数量）
    .reg .pred %p<2>;         // 谓词寄存器

    ld.global.f32 %f1, [%rd1];     // 加载（虚拟指令）
    ld.global.f32 %f2, [%rd2];
    add.f32 %f3, %f1, %f2;         // 加法（虚拟指令）
    st.global.f32 [%rd3], %f3;     // 存储

    ret;
}

// ptxas 的转换（→ SASS，特定于 GPU 代数，例如 sm_80 = A100）:
// 将虚拟寄存器映射到物理寄存器
// 调度指令（重排顺序）
// 选择具体指令变体（例如选择哪个 load 指令）
// 生成 cubin 二进制
```

**为什么 PTX 对 AI 编译器重要：**

1. **虚拟 ISA**：PTX 是 NVIDIA 的唯一公开可编程接口。你不能直接生成 SASS（本机指令）
2. **向后兼容**：同一 PTX 可在不同 GPU 代数上运行（ptxas 负责适配）
3. **源码级优化**：Triton 在生成 PTX 前做所有的优化（tiling、fusion、SW pipelining）
4. **ptxas 的角色**：ptxas 做寄存器分配（虚拟→物理）和最终调度。Triton 依赖 ptxas 做这些事情

### IREE 可执行格式

IREE 支持多种目标后端，每种都有自己的"汇编器"等价物：

```
IREE MLIR → IREE Flow → IREE HAL → 目标后端

目标后端：
  1. CPU (LLVM)
     MLIR → LLVM IR → LLVM AsmPrinter → ELF binary (.so/.dylib)
     → 标准可执行格式，支持动态链接

  2. CUDA (NVVM)
     MLIR → NVVM IR → PTX → ptxas → cubin
     → cubin 嵌入到 IREE executable 中
     → 通过 CUDA driver API 加载

  3. Vulkan (SPIR-V)
     MLIR → SPIR-V dialect → spirv-translate → SPIR-V binary
     → SPIR-V 嵌入到 IREE executable 中
     → 通过 Vulkan API 创建 shader module

  4. Metal (MSL)
     MLIR → Metal Shading Language source → Metal compiler → metallib
     → metallib 嵌入到 IREE executable 中
```

**IREE 的 executable 格式：**

```
FlatBuffer 格式（IREE 自定义）:
┌──────────────────────────────────────────┐
│  Header: magic, version, flags           │
├──────────────────────────────────────────┤
│  Executable format:                      │
│    for each entry point:                 │
│      - function name                     │
│      - workgroup size / layout           │
│      - binary payload (cubin/SPIR-V/...) │
│      - constants / push constants        │
│      - binding info (descriptor sets)    │
└──────────────────────────────────────────┘
```

### CPU 目标文件与 GPU 制品封装必须分层理解

对 x86/AArch64 等目标，LLVM MC layer 可以直接产生 ELF/Mach-O/COFF，fixup 由
`MCObjectWriter` 转换成目标格式定义的 relocation。NVPTX 后端则主要输出 PTX 文本；
`ptxas` 或 CUDA driver JIT 再生成 cubin。host 侧 fatbinary 如何封装 PTX/cubin，和
NVPTX AsmPrinter 发射 PTX，是两个不同阶段，不能虚构一个 `NVPTXObjectWriter` 把二者混在一起。

生产排查时分别保存并检查：

```bash
# CPU/标准 ELF：LLVM 自己生成并检查对象文件
llc -filetype=obj input.bc -o input.o
llvm-readobj --file-headers --sections --symbols --relocations input.o
llvm-objdump -dr input.o

# NVIDIA：分别观察 PTX 和 ptxas 输出，寄存器数/spill 以 ptxas 报告为准
llc -march=nvptx64 input.bc -o input.ptx
ptxas -arch=sm_80 -v input.ptx -o input.cubin
```

## 示例说明

### 示例 1：完整汇编流程（从 Machine IR 到 Object File）

```
Machine IR (post RA, post stack lowering):
  bb.0:
    %0:gpr32 = LD_ri %stack.0.fixed_arg, 0   // load 参数
    %1:gpr32 = ADD_rr %0, %0                   // 加倍
    STR_ri %1, %stack.1.local_var, 0          // store 结果
    RET

Step 1: AsmPrinter 转换为 MCInst
  → LD_ri → Opcode=MyTarget::LDR, Op0=Reg(R0), Op1=FI(0), Op2=Imm(0)
  → ADD_rr → Opcode=MyTarget::ADD, Op0=Reg(R1), Op1=Reg(R0), Op2=Reg(R0)
  → STR_ri → Opcode=MyTarget::STR, Op0=Reg(R1), Op1=FI(1), Op2=Imm(0)

Step 2: MCStreamer 发射
  (如果是 MCAsmStreamer)
    ldr r0, [sp, #0]
    add r1, r0, r0
    str r1, [sp, #4]
    bx lr

  (如果是 MCObjectStreamer)
    → MCCodeEmitter::encodeInstruction (每次调用)
      LDR: 0xE59D0000 (= cond:1110, op:01, ... 编码为 32-bit)
      ADD: 0xE0800000
      STR: 0xE58D1004
    → 写入字节到 section fragment
    → MCAssembler 处理 fixups
    → MCObjectWriter 写入 .o 文件

Step 3: .o 文件内容
  .text section: E5 9D 00 00  E0 80 00 00  E5 8D 10 04
  .rel.text section: (如果有重定位)
    Offset=0x0, Type=R_ARM_..., Symbol=external_func
  .symtab section:
    function_name: .text+0x0, size=12
```

### 示例 2：PTX 生成与 ptxas 汇编

```
输入 (Triton kernel, MLIR → LLVM NVVM IR):
define void @kernel(ptr %A, ptr %B, ptr %C) {
  %a = load float, ptr %A
  %b = load float, ptr %B
  %c = fadd float %a, %b
  store float %c, ptr %C
  ret void
}

NVPTX AsmPrinter 输出 (PTX 文本):
  .visible .entry kernel(
    .param .u64 kernel_param_0,
    .param .u64 kernel_param_1,
    .param .u64 kernel_param_2
  ) {
    .reg .f32 %f<3>;
    .reg .u64 %rd<4>;

    ld.param.u64 %rd1, [kernel_param_0];   // 加载指针参数
    ld.param.u64 %rd2, [kernel_param_1];
    ld.param.u64 %rd3, [kernel_param_2];

    ld.global.f32 %f1, [%rd1];             // 加载 A
    ld.global.f32 %f2, [%rd2];             // 加载 B
    add.f32 %f3, %f1, %f2;                 // 加法
    st.global.f32 [%rd3], %f3;             // 存储 C

    ret;
  }

ptxas 处理 (内部黑盒):
  Step 1: 解析 PTX → 内部 IR
  Step 2: 寄存器分配（虚拟 → 物理）
  Step 3: 指令调度（重排指令以隐藏延迟）
  Step 4: 选择具体指令（SASS 编码，如: LDS, STS, FFMA, HADD2 等）
  Step 5: 生成 cubin（二进制）

cubin (二进制，嵌入到 fatbinary 中):
  → CUDA driver 在 kernel launch 时加载 cubin 到 GPU
```

### 示例 3：用工具确认真实重定位，而不是猜名称

```bash
# 生成保留外部函数调用的对象文件
clang -target aarch64-linux-gnu -c caller.c -o caller.o

# 同时查看符号和 relocation
llvm-readobj --symbols --relocations caller.o
llvm-objdump -dr caller.o
```

你应在 AArch64 上看到该 ABI/对象格式定义的真实 relocation（外部调用常见为
`R_AARCH64_CALL26`）；换成 x86-64 后名称和编码会改变。CUDA cubin 的 relocation
由 NVIDIA 工具链和其对象格式决定，不应在 LLVM NVPTX 教学代码里臆造 `R_CUDA_*`
枚举。工业文档必须以当前工具输出和格式规范为准。

## 工业落地：对象文件是必须验收的产品边界

后端能输出 `.s` 不代表工具链完成。发布前至少检查：

```bash
llvm-readobj --file-headers --sections --symbols --relocations output.o
llvm-objdump -dr --no-show-raw-insn output.o
```

验收项包括：

- 文件格式、架构、endianness、ABI flags 与 code model 正确。
- section 名称、权限、对齐和 COMDAT/group 符合平台约定。
- 符号 binding、visibility、size、TLS 模型与链接预期一致。
- relocation 的类型、addend、作用 section 和目标符号正确。
- prologue/epilogue、栈对齐、callee-saved、CFI/unwind 信息可被系统工具读取。
- debug build 的 DWARF/CodeView 可用，strip 后生产制品不泄漏不应发布的信息。
- 静态/动态链接、加载、执行和最小 ABI 互操作测试通过。

CPU `.o`、PTX、cubin、SPIR-V 是不同产品边界，应分别使用对应工具验证。尤其不要用 PTX
文本“看起来正确”替代 `ptxas` 的寄存器、spill、错误报告和目标 GPU 运行测试。

## 总结

### 核心要点

1. **MC Layer 是独立的机器码库**：它不仅用于编译器后端（codegen），也用于独立的 assembler/disassembler
2. **AsmPrinter 是 MachineInstr → MCInst 的桥梁**：配合 MCStreamer 决定输出格式（二进制 .o 或文本 .s）
3. **Fixups → Relocations**：Fixups 在汇编时记录未解析的引用，在输出时转换为 object file relocations
4. **TableGen 生成大部分 MC 代码**：包括指令编码、反汇编、汇编解析器的 pattern matching

### AI 编译器工程师的关键理解

| 概念 | LLVM 实践 | AI 编译器实践 |
|------|----------|-------------|
| 代码生成 | MC Layer (MCCodeEmitter + MCObjectWriter) | NVPTX backend → PTX → ptxas → cubin |
| 目标格式 | ELF / MachO / COFF | cubin (NVIDIA), SPIR-V (Vulkan), metallib (Metal) |
| 汇编器 | MCAsmParser + gen-asm-matcher | ptxas (闭源), spirv-as (开源), metal compiler (闭源) |
| 重定位 | ELF relocations (R_*) | CUDA relocations (R_CUDA_*) |
| 指令编码 | MCCodeEmitter (TableGen 驱动) | ptxas 内部（NVIDIA 闭源） |
| 可执行格式 | .o/.obj → linker → executable | IREE FlatBuffer executable（嵌入 cubin/SPIR-V） |

### 进阶话题

- **MachO / COFF 目标文件格式**：除了 ELF，LLVM 还支持 MachO（macOS/iOS）和 COFF（Windows）格式。每种格式的重定位类型、段（section）命名约定都不同
- **Debug Info (DWARF)**：AsmPrinter 也负责发射调试信息（DWARF/CodeView），通过 MCStreamer 的 `emitDwarf*` 方法系列
- **CFI (Call Frame Information)**：用于栈回溯（stack unwinding）的元数据，在 prologue/epilogue 中由 `emitCFI*` 方法发射
- **JIT 编译的 Code Emission**：LLVM JIT（ORCv2/MCJIT）直接使用 MC Layer 生成内存中的代码（无需写文件），这在对延迟敏感的 AI 推理场景中非常重要
- **Triton 的未来方向**：Triton 团队正在探索绕过 ptxas 直接生成 SASS，以获得更确定的性能（但目前 NVIDIA 不公开 SASS 规范）
