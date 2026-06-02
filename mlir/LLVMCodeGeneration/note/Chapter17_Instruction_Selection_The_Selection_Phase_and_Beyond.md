# Chapter 17: Instruction Selection - The Selection Phase and Beyond

> **From the perspective of a production AI compiler engineer who needs to understand LLVM deeply to work on MLIR/Triton/AI compiler stacks.**

## 核心概念（详细展开）

### 选择阶段的本质

Selection Phase（选择阶段）是指令选择管线的最后阶段：将合法化的通用 IR 转换（翻译）为目标特定的 Machine IR。这是编译器后端中将**抽象语义**映射到**具体硬件指令**的关键一环。

**为什么选择阶段对 AI 编译器至关重要：**
- 在 MLIR 中，`InstructionSelect` pass（或等价转换）是 dialect lowering 的核心
- Triton 需要将通用操作选择到 PTX mma（Tensor Core）、shuffle、ld.global 等指令
- AI 加速器通常有高度定制的指令集，需要比标准 LLVM backend 更灵活的选择逻辑
- 选择阶段的 pattern matching 能力直接影响生成的代码质量（指令融合、addressing mode 折叠等）

### 本章涵盖内容

本章不仅涵盖选择本身，还涵盖选择前后所有相关机制：

1. **RegBankSelect**：GlobalISel 独有的寄存器银行分配（介于 legalization 和 selection 之间）
2. **Selection Patterns**：TableGen 描述的选择规则（跨 SDISel/FastISel/GlobalISel 共享）
3. **Advanced Pattern Matching**：PatFrag、ComplexPattern、SDNodeXForm 等高级机制
4. **Optimizations**：DAGCombiner（SDISel）、GICombiner（GlobalISel）用于选择前/后优化
5. **Finalization**：FinalizeISel pass 的定制（custom inserter、reserved registers）
6. **Debugging**：match table 的调试方法

## LLVM / MLIR 流程（深入）

### SDISel 指令选择流程

```
┌──────────────────────────────────────────────────────────┐
│                    SelectionDAGISel Pass                  │
│                                                          │
│  SelectionDAG (legalized)                                │
│      │                                                    │
│      ▼                                                    │
│  ┌──────────────┐                                        │
│  │ Select(SDNode*)│ ← 对 DAG 中的每个节点                    │
│  │   ├─ isMachineOpcode()? → skip（已选择）                 │
│  │   └─ SelectCode(N) → TableGen 生成的 match table       │
│  └──────┬───────┘                                        │
│         │                                                 │
│         ▼                                                 │
│  Match Table Interpreter (state machine)                 │
│    → 匹配 pattern → 生成 MachineInstr                     │
│    → 无匹配 → 保留 ISD:: opcode（后续报错）                │
│      │                                                    │
│      ▼                                                    │
│  MachineInstr (target-specific)                          │
└──────────────────────────────────────────────────────────┘
```

#### TableGen 选择模式基础

```tablegen
// 方式 1：Pattern 字段（嵌入在 Instruction 定义中）
let Pattern = [(set GPR16:$dst, (add GPR16:$src0, GPR16:$src1))] in
def ADDi16rr : H2BLBBinaryInstruction<"addi16", ...>;

// 方式 2：Pat 类（独立模式）
def : Pat<(i32 (mul (sext GPR16:$src0), (sext GPR16:$src1))),
          (WIDENING_SMUL GPR16:$src0, GPR16:$src1)>;
```

**Pattern 匹配的工作原理：**
- `add` → 匹配 ISD::ADD opcode（SDNode 定义在 `TargetSelectionDAG.td` 中）
- `GPR16` → 类型由 RegisterClass 的 type 列表决定（`def GPR16 : RegisterClass<..., [i16], ...>`）
- `$dst, $src0, $src1` → 映射到 `OutOperandList`/`InOperandList` 的命名操作数
- Pattern 可以嵌套任意深度以匹配复杂 DAG 子图

#### 选择模式的高级机制

**1. Pattern Fragments (PatFrag) - 可复用的子模式片段**

```tablegen
// 定义：匹配特定范围的立即数
def uimm7 : ImmLeaf<i16, [{return Imm >= 0 && Imm < 128;}]>, ...>;

// 使用：在 load 指令中
def LD16imm7 : H2BLBInstruction<"ldi16", "$dst, $imm7", ...> {
  let Pattern = [(set GPR16:$dst, uimm7:$imm7)];
}
```

**AI 编译器应用**：匹配 Tensor Core MMA 指令的不同 shape：
```tablegen
def MMA_M16N8K16 : PatFrag<(ops node:$a, node:$b, node:$c),
  [{ return isMMACompatible(node:$a, node:$b, 16, 8, 16); }]>;
```

**2. SDNodeXForm - 在匹配时变换 SDNode**

```tablegen
// 定义：将 generic frameindex 转为 target-specific frameindex
def to_tframeindex : SDNodeXForm<frameindex, [{
  return CurDAG->getTargetFrameIndex(N->getIndex(), N->getValueType(0));
}]>;

// 使用
def : Pat<(i16 (frameindex:$ptr)),
          (MOVFROMSP (i16 (to_tframeindex $ptr)))>;
```

**SDNodeXForm 的局限性：** 与 SDNode 实例绑定，**不能跨选择器使用**（FastISel 和 GlobalISel 不支持）。

**3. ComplexPattern - C++ 自定义匹配逻辑**

```tablegen
// 定义：addressing mode 的 complex pattern（产生 2 个 value）
def addrmode : ComplexPattern<iPTR, 2, "selectAddrMode", []>;
```

```cpp
// 实现：在 XXXDAGToDAGISel 中
bool H2BLBDAGToDAGISel::selectAddrMode(SDValue N, SDValue &Base,
                                        SDValue &OffImm) {
  // 匹配 base + offset 的 addressing mode
  if (N.getOpcode() == ISD::ADD &&
      isa<ConstantSDNode>(N.getOperand(1))) {
    Base = N.getOperand(0);
    OffImm = CurDAG->getTargetConstant(
        cast<ConstantSDNode>(N.getOperand(1))->getSExtValue(), ...);
    return true;
  }
  Base = N;
  OffImm = CurDAG->getTargetConstant(0, ...);
  return true;  // 总是匹配（fallback: base + 0）
}
```

```tablegen
// 使用
def : Pat<(load (addrmode GPR16:$addr, uimm4:$offset)),
          (LDR16 $addr, $offset)>;
```

**AI 编译器应用：** 匹配 AI 加速器的复杂 addressing mode：
```tablegen
// TPU 的 2D tensor addressing: base + (row * stride) + col
def tpu_2d_addr : ComplexPattern<iPTR, 3, "selectTPU2DAddr", []>;
```

#### AddedComplexity 字段

当多个 pattern 可以匹配同一个输入时，由 `AddedComplexity` 决定优先级：

```tablegen
// 通用 pattern（低复杂度 → 后尝试）
def : Pat<(load GPR32:$addr), (LDR32 $addr, 0)>;

// 更具体的 pattern（高复杂度 → 先尝试）
let AddedComplexity = 10 in
def : Pat<(load (addrmode GPR32:$addr, uimm8:$offset)),
          (LDR32_OFF $addr, $offset)>;
```

### FastISel 指令选择

FastISel 是 SDISel 的**子选择器**，仅处理简单情况，失败时回退到 SDISel：

```cpp
class H2BLBFastISel final : public FastISel {
public:
  bool fastSelectInstruction(const Instruction *I) override {
    // 1. 检查是否回退到 SDISel
    if (TLI.fallBackToDAGISel(*I)) return false;

    // 2. 自定义快速选择（针对简单 LLVM IR opcode）
    switch (I->getOpcode()) {
    case Instruction::Add:
      return selectAdd(I);
    default:
      break;
    }

    // 3. TableGen 生成的选择代码（仅简单 patterns）
    return selectOperator(I, I->getOpcode());
  }
};
```

**FastISel 的限制与模式导入：**
- 仅导入**简单 patterns**（无 ComplexPattern、SDNodeXForm 等高级构造）
- `skipTargetIndependentISel=true`：先运行自定义代码再运行 TableGen 代码（更多自定义机会）
- `skipTargetIndependentISel=false`：先运行 TableGen 代码再运行自定义代码

### GlobalISel 指令选择

GlobalISel 的选择通过 `InstructionSelect` pass 和 `InstructionSelector` 类实现。

#### InstructionSelector 类设置

```cpp
class H2BLBInstructionSelector : public InstructionSelector {
public:
  bool select(MachineInstr &I) override;

private:
  // TableGen 生成的 selectImpl
  bool selectImpl(MachineInstr &I, CodeGenCoverage &CoverageInfo) const;
};
```

```cpp
bool H2BLBInstructionSelector::select(MachineInstr &I) {
  // 已选择的指令直接返回成功
  if (!isPreISelGenericOpcode(I.getOpcode()))
    return true;

  // 尝试 TableGen 生成的选择
  if (selectImpl(I, *CoverageInfo))
    return true;

  // 回退到手动选择（自定义 C++）
  switch (I.getOpcode()) {
  case TargetOpcode::G_FRAME_INDEX:
    return selectFrameIndex(I);
  case TargetOpcode::G_BRCOND:
    return selectBrCond(I);
  }
  return false;
}
```

#### GlobalISel 的模式导入机制

GlobalISel **自动导入** SDISel 的 TableGen patterns，但需要满足条件：

```tablegen
// 模式 1：可以导入（类型完全指定）
def : Pat<(vector_extract (v2i16 GPR32:$rs), (i16 0)),
          (i16 (EXTRACT_SUBREG GPR32:$rs, sub_low16))>;
// ✓ - 所有类型显式指定

// 模式 2：无法导入（缺少类型信息）
def : Pat<(vector_extract (v2i16 GPR32:$rs), 0),
          (i16 (EXTRACT_SUBREG $rs, sub_low16))>;
// ✗ - 常量 0 缺少类型；输出 pattern 未指定 $rs 的寄存器类

// 使用 --warn-on-skipped-patterns 检查导入状态
```

#### ComplexPattern 的 GlobalISel 桥接

```tablegen
// 为 SDISel complex pattern 建立 GlobalISel 映射
def gi_addrmode :
  GIComplexOperandMatcher<p0, "selectAddrMode">,
  GIComplexPatternEquiv<addrmode>;
```

```cpp
// 在 InstructionSelector 中实现
InstructionSelector::ComplexRendererFns
H2BLBInstructionSelector::selectAddrMode(MachineOperand &Root) const {
  // 返回 renderer 函数（每个输出 operand 一个）
  return {{
      [=](MachineInstrBuilder &MIB) { MIB.addReg(BaseReg); },
      [=](MachineInstrBuilder &MIB) { MIB.addImm(Offset); },
  }};
}
```

**GlobalISel vs SDISel ComplexPattern 的关键差异：**
- SDISel：通过引用参数返回匹配值
- GlobalISel：返回 **renderer 函数**列表（延迟执行）

#### Custom SDNode 的 GlobalISel 等价

```tablegen
// 告诉 importer：G_EXTRACT_VECTOR_ELT 等价于 vector_extract SDNode
def : GINodeEquiv<G_EXTRACT_VECTOR_ELT, vector_extract>;
```

## 关键机制解析（工业视角）

### DAGCombiner 框架（SDISel 优化）

DAGCombiner 在 SDISel pipeline 的 4 个固定点运行：

```
SelectionDAG 生命周期中的 DAGCombine 位置：
  1. BeforeLegalizeTypes    ← DAGCombine1
  2. AfterLegalizeTypes     ← DAGCombine2
  3. AfterLegalizeVectorOps ← DAGCombine3
  4. AfterLegalizeDAG       ← DAGCombine4
```

```cpp
// 注册自定义 rewrite
// 在 TargetLowering 构造函数中：
setTargetDAGCombine(ISD::ADD);
setTargetDAGCombine(ISD::LOAD);

// 实现自定义 rewrite
SDValue XXXTargetLowering::PerformDAGCombine(SDNode *N,
                                              DAGCombinerInfo &DCI) const {
  switch (N->getOpcode()) {
  case ISD::ADD:
    return combineAdd(N, DCI);
  }
  return SDValue();  // 无 rewrite
}
```

**DAGCombine 的陷阱：** 自定义 rewrite 可能与通用 rewrite 冲突，导致无限循环：
```
Your rewrite:   f32 load → i32 load + bitcast
Generic rewrite: i32 load + bitcast → f32 load  （与你相反！）
→ 无限循环！
```

**解决方案：** 引入 target-specific opaque SDNode（`H2BLBISD::MY_BITCAST`）来打破循环。

### GICombiner 框架（GlobalISel 优化）

GlobalISel 支持创建独立的 combiner MachineFunctionPass：

```tablegen
// TableGen 定义 combiner 规则
def registers_matchinfo: GIDefMatchData<"SmallVector<Register>">;

def insertvectorelt_to_build_vector : GICombineRule<
  (defs root:$root, registers_matchinfo:$matchinfo),
  (match (wip_match_opcode G_INSERT_VECTOR_ELT):$root,
         [{ return matchInsertVectorElt(*${root}, ${matchinfo}); }]),
  (apply [{ applyInsertVectorElt(*${root}, ${matchinfo}); }])>;
```

```cpp
// 创建 combiner pass
class MyCombiner : public Combiner {
  bool runOnMachineFunction(MachineFunction &MF) override {
    CombinerInfo Info(/*...*/);
    MyCombinerImpl Impl(Info);
    return Impl.combineMachineInstrs(MF, /*...*/);
  }
};
```

**Combiner 的注入点：**
- Pre-legalizer combiner：`TargetPassConfig::addPreLegalizeMachineIR()`
- Post-legalizer combiner：`TargetPassConfig::addPreRegBankSelect()`
- Pre-select combiner：`TargetPassConfig::addPreGlobalInstructionSelect()`

### Custom Inserter（选择后的自定义插入器）

```cpp
// TableGen: 标记伪指令需要 custom inserter
let usesCustomInserter = true in
def LD16imm16 : H2BLBPseudoInstruction<...>;

// C++: 实现 custom inserter
MachineBasicBlock *H2BLBTargetLowering::EmitInstrWithCustomInserter(
    MachineInstr &MI, MachineBasicBlock *BB) const {
  switch (MI.getOpcode()) {
  case H2BLB::LD16imm16:
    return emitLDimm(MI, BB);  // 展开为复杂指令序列
  }
}
```

**AI 编译器应用：** Custom inserter 可用于展开 AI 加速器的微码指令、复杂常量加载等。

### FinalizeISel 的阶段定制

```cpp
void XXXTargetLowering::finalizeLowering(MachineFunction &MF) const {
  // ⚠️ 重要：GlobalISel 会调用此方法两次
  //   - 一次在 instruction selection phase 内
  //   - 一次在 FinalizeISel pass 中
  // 必须用 MachineFunction property 防重复：
  if (MF.getProperties().hasProperty(
          MachineFunctionProperties::Property::Selected))
    return;

  // 自定义逻辑...

  // 必须调用父类方法（freezes reserved registers）
  TargetLowering::finalizeLowering(MF);
}
```

## AI 编译器关联

### MLIR 的 Pattern Rewrite Engine vs LLVM Selectors

| 维度 | LLVM (SDISel/GISel) | MLIR Pattern Rewrite |
|------|---------------------|---------------------|
| **模式描述** | TableGen (Pat/PatFrag/ComplexPattern) | C++ RewritePattern / DRR (Declarative Rewrite Rules) |
| **匹配引擎** | Match Table Interpreter (state machine) | GreedyPatternRewriteDriver / DialectConversion |
| **匹配范围** | Single SDNode or MachineInstr root | Multi-operation subgraph |
| **新值创建** | 通过 TableGen 模式或 C++ custom | 通过 `rewriter.create<>()` |
| **多结果** | 有限支持（ComplexPattern 可拆分为多个） | 原生支持（任意数量的 result value） |
| **收敛性** | 单次遍历（DAGCombine 除外） | 迭代到固定点（greedy rewriter） |

**MLIR DRR 示例：**

```tablegen
// MLIR 的声明式重写规则（类似 LLVM 的 Pat 类）
def : Pat<(Arith_AddFOp (Arith_MulFOp $a, $b), $c),
          (FMADOp $a, $b, $c)>;
// 将 (a * b) + c 融合为 fma(a, b, c)
```

```cpp
// MLIR C++ RewritePattern（类似 LLVM Custom + ComplexPattern）
struct FuseMulAddPattern : public OpRewritePattern<arith::AddFOp> {
  LogicalResult matchAndRewrite(arith::AddFOp op,
                                 PatternRewriter &rewriter) const override {
    auto mul = op.getLhs().getDefiningOp<arith::MulFOp>();
    if (!mul) return failure();
    rewriter.replaceOpWithNewOp<FMADOp>(op, mul.getLhs(),
                                         mul.getRhs(), op.getRhs());
    return success();
  }
};
```

### Triton GPU 指令选择

Triton 编译器在 MLIR 基础上的指令选择流程：

```
Triton IR (triton dialect)
    │
    ▼
┌──────────────────────────────────────────┐
│ Pattern-based dialect lowering           │
│  tl.dot → nvgpu.mma.sync (A100/H100)     │ ← TableGen DRR / C++ pattern
│  tl.load → nvgpu.ldmatrix + coalescing   │ ← Complex addressing mode matching
│  tl.store → nvgpu.stmatrix               │
│  tl.atomic_add → nvgpu.atom + loop       │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│ GPU-specific instruction selection       │
│  nvgpu → llvm.inline_asm (PTX)           │ ← 类似 LLVM 的 TableGen selection
│  或 nvgpu → nvvm intrinsics              │
└──────────────────┬───────────────────────┘
                   │
                   ▼
              PTX Assembly
```

**Triton 选择的关键特点：**

1. **Tensor Core MMA 指令选择**：
   - `tl.dot` → 根据 GPU 代数选择 `mma.sync.aligned.m16n8k16` 或 `mma.sync.aligned.m16n8k8` 等
   - 需要分析 dot 操作的 shape 和数据类型来匹配合适的 MMA 变体
   - 类似 LLVM 的 `AddedComplexity`：更具体的 shape 匹配优先级更高

2. **Memory Coalescing**：
   - `tl.load` 的选择需要考虑是否是 coalesced access
   - Coalesced：`ld.global.v4.f32`（128-bit 单指令加载）
   - Non-coalesced：多个 `ld.global.b32` + shuffle 重组

3. **Warp-Level Primitives**：
   - `tl.reduce` → `__shfl_xor_sync` 序列（类似 LLVM 的 Lower pattern）
   - `tl.broadcast` → `__shfl_sync`（类似 LLVM 的 Legal action）

### 自定义 Combiner 用于 AI 加速器

AI 加速器通常需要大量指令融合和 pattern rewrite：

```cpp
// 示例：AI 加速器的 layer normalization 融合
// 输入: (x - mean) / sqrt(var + eps) * gamma + beta
//
// Pattern: sub → div → mul → add
//   → 替换为单个 layernorm 加速器指令

struct LayerNormFusionPattern : public GICombineRule {
  bool match(MachineInstr &Root) {
    // 匹配 sub+div+mul+add 的 DAG 子图
    return matchLayerNormDAG(Root, matchInfo);
  }
  void apply(MachineInstr &Root, SmallVector<Register> &matchInfo) {
    // 替换为单个加速器指令
    replaceWithAccelNorm(Root, matchInfo);
  }
};
```

类似地，attention 机制（`softmax(QK^T/sqrt(d))V`）可以通过 combiner 融合为单个 AI 加速器指令。

## 示例说明

### 示例 1：完整的选择模式生成指令序列

```tablegen
// 输入 LLVM IR: %res = shl i32 %val, %amt
// 目标架构: 只有 16-bit shift，需要先 EXTRACT_SUBREG

// Pattern 定义
def : Pat<(i32 (shl (i32 GPR32:$src0), (i32 GPR32:$src1))),
          (SHL32rr (i32 GPR32:$src0),
                   (i16 (EXTRACT_SUBREG GPR32:$src1, sub_low16)))>;
```

```
生成的 Machine IR:
  %1:gpr16 = EXTRACT_SUBREG %src1, sub_low16
  %res:gpr32 = SHL32rr %src0, %1
```

### 示例 2：SDISel Match Table 调试

```
debug-only=isel 输出:

ISEL: Starting selection on root node: t9: v2i32 = add t2, t2
ISEL: Starting pattern match
  Initial Opcode index to 31041
  Match failed at index 31046
  Continuing at 158464
  Match failed at index 158468
  Continuing at 127419
  ...

// 在 XXXGenDAGISel.inc 中查找 index 31041:
/* 31041*/  OPC_Scope, 59|128,99|128,7/*127419*/, /*->158464*/
/* 31045*/   OPC_MoveChild0,
/* 31046*/   OPC_SwitchOpcode /*2 cases */, 125|128,120|128,3/*64637*/,
            TARGET_VAL(ISD::MUL), // ->95689

// 说明：在 index 31046，匹配器期望第一个 child 是 ISD::MUL
// 但实际是 ISD::ADD → 匹配失败 → 跳转到 158464
```

### 示例 3：MLIR Pattern 与 LLVM Pattern 的对比

```tablegen
// === LLVM TableGen Pattern ===
def : Pat<(add GPR32:$a, (mul GPR32:$b, GPR32:$c)),
          (MADD32 $a, $b, $c)>;
```

```tablegen
// === MLIR DRR Pattern ===
def : Pat<(Arith_AddIOp $a, (Arith_MulIOp $b, $c)),
          (MADDOp $a, $b, $c)>;
```

```cpp
// === MLIR C++ Pattern（等价实现） ===
struct FoldMulAdd : public OpRewritePattern<arith::AddIOp> {
  LogicalResult matchAndRewrite(arith::AddIOp op,
                                 PatternRewriter &r) const override {
    auto mul = op.getLhs().getDefiningOp<arith::MulIOp>();
    if (!mul) return failure();
    r.replaceOpWithNewOp<MADDOp>(op, op.getRhs(),
                                  mul.getLhs(), mul.getRhs());
    return success();
  }
};
```

## 总结

### 核心要点

1. **Selection patterns（TableGen）** 是 LLVM 中**跨选择器共享**代码的关键机制（SDISel/FastISel/GlobalISel 均可导入）
2. **GlobalISel 的 pattern 导入**有类型限制：必须显式指定所有类型；complex patterns 需要通过 `GIComplexPatternEquiv` 桥接
3. **DAGCombiner 和 GICombiner** 提供了选择前后优化的框架，但要注意与通用优化的循环冲突
4. **MLIR 的 pattern rewrite engine** 是 LLVM selection 思想的泛化版，支持多结果、迭代到固定点、dialect 转换

### AI 编译器工程师的关键理解

| 概念 | LLVM 实践 | AI 编译器实践 |
|------|----------|-------------|
| Pattern 描述 | TableGen Pat/PatFrag/ComplexPattern | MLIR DRR / C++ RewritePattern |
| 复杂匹配 | ComplexPattern + C++ function | MLIR OpRewritePattern::matchAndRewrite |
| 指令融合 | DAGCombiner / GICombiner | MLIR Canonicalizer / Custom Combine pass |
| 新指令插入 | Custom Inserter (usesCustomInserter) | MLIR RewritePattern::replaceOpWithNewOp |
| Match Table | gen-dag-isel state machine | MLIR PatternApplicator 驱动的 greedy rewrite |
| 优先级控制 | AddedComplexity field | MLIR pattern benefit（数值越高优先级越高） |
| 跨框架共享 | TableGen patterns for SDISel+FastISel+GISel | DRR patterns for MLIR dialects |

### 进阶话题

- **Selection 的 complete vs. incomplete**：GlobalISel 允许在 selection 中留下未选择的指令（通过 fallback），而 SDISel 要求每个节点都被选择
- **COPY/PHI 指令的选择**：这些指令在前选择和后选择都合法，但在选择阶段必须验证/赋值寄存器类
- **Pattern 的 Added Complexity 调优**：高复杂度 pattern 可能爆炸性地增加 match table 大小→需要在覆盖率和编译速度之间权衡
- **Machine Verifier 的作用**：在 selection 后运行 `-verify-machineinstrs` 是发现选择错误的廉价方法
