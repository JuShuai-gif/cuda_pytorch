# Chapter 16: Instruction Selection - The Legalization Phase

> **From the perspective of a production AI compiler engineer who needs to understand LLVM deeply to work on MLIR/Triton/AI compiler stacks.**

## 核心概念（详细展开）

### Legalization 的本质

Legalization（合法化）是将通用 IR 中"疯狂"的构造转换为目标架构可以直接执行的指令序列的过程。在生产编译器中，legalization 是连接高层抽象与底层硬件的桥梁。

**为什么 Legalization 对 AI 编译器至关重要：**
- MLIR 中每个 dialect 本质上定义了不同的"合法性"概念——linalg dialect 合法的操作（如 `linalg.matmul`）在 GPU dialect 中是非法的，需要 legalization
- Triton 编译器需要将高层次的操作（如 `tl.dot`）legalize 到适合 NVIDIA GPU 的 PTX 指令
- IREE 的多层 legalization pipeline 需要逐步将通用 MLIR 操作 legalize 到特定硬件后端

### 两个核心机制

Legalization 围绕两种策略展开：

1. **使用更大的支持计算来模拟不支持的计算**
   - 例如：`add i13` → `add i32`（需要配合 sign/zero extension 和 truncation）
   - 在 AI 编译器中常见：将 fp8 类型 widen 到 fp16 再计算

2. **将大计算拆分为较小支持计算的序列**
   - 例如：`add i64` → `add i32` + 进位传播
   - 在 AI 编译器中常见：将 2D 卷积拆分为 im2col + matmul 或多个 1D 操作

### Legalization Actions 完整参考

| Action | SDISel 名称 | 典型 AI 编译器场景 |
|--------|------------|-------------------|
| `Legal` | `Legal` | 直接支持的 Tensor Core 操作 |
| `NarrowScalar` | `Expand` | 将 fp32 拆为两个 fp16（模拟 bf16 不支持的情况） |
| `WidenScalar` | `Promote` | 将 fp8 提升到 fp16 进行计算（常见于 NVIDIA Hopper fp8 tensor core） |
| `FewerElements` | `Expand` | 将 `<8 x f32>` 向量拆分为 `<4 x f32>` + `<4 x f32>` |
| `MoreElements` | `Promote` | 将 `<3 x f32>` 填充到 `<4 x f32>` |
| `Bitcast` | N/A | 类型重解释（如 `bitcast <4 x f16>` 到 `<2 x f32>`） |
| `Lower` | `Expand` | 将 `linalg.matmul` 分解为 loop + `arith.mulf` + `arith.addf` |
| `LibCall` | `LibCall` | 将 `math.exp` 替换为调用 `__expf` 运行时函数 |
| `Custom` | `Custom` | 将 `tl.dot` 映射为自定义的 Tensor Core PTX 指令序列 |

### 关键认知：Legalization Artifacts

合法化过程中产生的中间构件（artifacts）是理解 legalization 生产力的关键：

```
原始指令:  %res = and <3 x i8> %a, %b

Legalization (MoreElements 到 <4 x i8>):
  %a0, %a1, %a2 = unmerge_values <3 x i8> %a
  %moreEltA = build_vector %a0, %a1, %a2, i8 undef
  %b0, %b1, %b2 = unmerge_values <3 x i8> %b
  %moreEltB = build_vector %b0, %b1, %b2, i8 undef
  %moreEltRes = and <4 x i8> %moreEltA, %moreEltB
  %res0, %res1, %res2, %res3 = unmerge_values <4 x i8> %moreEltRes
  %res = build_vector %res0, %res1, %res2
```

**重要洞察：** 这些构件在完整的 def-use 链中往往会相互抵消（如 `trunc(zext(x))` → `x`），最终在 ABI 边界、load/store 处消失。不理解这一点会过度担忧合法化的复杂性。

## LLVM / MLIR 流程（深入）

### SDISel 的 Legalization 流程

SDISel 采用**两阶段**合法化：

```
SelectionDAG (post-IR-building)
    │
    ▼
┌─────────────────────────────┐
│ Type Legalization           │ ← 将非法类型转换为合法类型
│ (LegalizeTypes pass)        │   使用 Promote/Expand 等策略
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Operation Legalization      │ ← 对合法类型上的操作应用 action
│ (LegalizeDAG pass)          │   Legal/Expand/Lower/LibCall/Custom
└─────────────┬───────────────┘
              │
              ▼
        DAGCombine (可选)
              │
              ▼
     Instruction Selection
```

**关键设计决策：** SDISel 的"合法类型"概念意味着一旦标记某类型为合法，该类型上**所有操作默认合法**。必须显式标记不支持的操作。

#### 描述合法类型

```cpp
// 在 XXXTargetLowering 构造函数中
addRegisterClass(MVT::i16, &H2BLB::GPR16RegClass);
addRegisterClass(MVT::i32, &H2BLB::GPR32RegClass);
addRegisterClass(MVT::v2i16, &H2BLB::GPR32RegClass);  // 与 i32 共享寄存器类
addRegisterClass(MVT::f16, &H2BLB::GPR16RegClass);
// ...
computeRegisterProperties(Subtarget.getRegisterInfo());
```

**`computeRegisterProperties` 的关键作用：** 此方法根据注册的寄存器类计算所有类型相关属性，包括：
- 哪些类型是合法的
- 类型之间的转换关系（如 i32 ↔ v2i16 通过相同寄存器类关联）
- 寄存器类的选择优先级

#### 描述合法化 Actions

```cpp
// 设置 (opcode, type) 对的处理方式
setOperationAction(ISD::FADD,       MVT::f32,  LibCall);
setOperationAction(ISD::UREM,       MVT::i32,  Expand);
setOperationAction(ISD::MUL,        MVT::i32,  Custom);
setOperationAction(ISD::CTPOP,      MVT::i16,  Legal);
setOperationAction(ISD::SDIV,       MVT::i32,  Promote);

// 特殊操作的 action（使用专用 API）
setTruncStoreAction(MVT::i32, MVT::i16, Expand);    // 截断 store
setLoadExtAction(ISD::SEXTLOAD, MVT::i16, Expand);   // 符号扩展 load
setCondCodeAction(ISD::SETGT, MVT::i32, Expand);     // 条件码
setOperationAction(ISD::BR_CC, MVT::i32, Expand);    // 条件分支
```

#### Custom Legalization 实现

当 `Custom` 被设置时，SDISel 调用 `LowerOperation`：

```cpp
SDValue XXXTargetLowering::LowerOperation(SDValue Op,
                                           SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  case ISD::MUL:
    return lowerMUL(Op, DAG);  // 委托给专用方法
  }
}

SDValue XXXTargetLowering::lowerMUL(SDValue Op, SelectionDAG &DAG) const {
  // 检查是否可以 custom lower
  if (PlainLHS.getValueType() != MVT::i16)
    return SDValue();  // 失败 → 回退到 Expand → LibCall → abort

  // 创建目标特定节点（如 widening multiply）
  unsigned Opcode = isSigned ? H2BLBISD::WIDENING_SMUL
                              : H2BLBISD::WIDENING_UMUL;
  return DAG.getNode(Opcode, SDLoc(Op), ValTy, PlainLHS, PlainRHS);
}
```

**失败回退链：** Custom 失败 → Expand 失败 → LibCall 失败 → **abort（编译器崩溃）**。必须确保回退路径可达。

### GlobalISel 的 Legalization 流程

GlobalISel **没有"合法类型"概念**，每个 `(opcode, type)` 对必须被显式覆盖。

```
Machine IR (post-IRTranslator, generic opcodes)
    │
    ▼
┌─────────────────────────────┐
│ Legalizer Pass              │ ← 独立 MachineFunctionPass
│ (LegalizerInfo 驱动)         │   使用 LegalizeRuleSet API
└─────────────┬───────────────┘
              │
              ▼
    RegBankSelect Pass
              │
              ▼
    InstructionSelect Pass
```

#### LegalizerInfo 类实现

```cpp
class H2BLBLegalizerInfo : public LegalizerInfo {
public:
  H2BLBLegalizerInfo(const H2BLBSubtarget &ST);
  bool legalizeCustom(LegalizerHelper &Helper, MachineInstr &MI,
                      LostDebugLocObserver &LocObserver) const override;
};
```

#### LegalizeRuleSet API 详解

```cpp
H2BLBLegalizerInfo::H2BLBLegalizerInfo(const H2BLBSubtarget &ST) : ST(ST) {
  const LLT s16 = LLT::scalar(16);
  const LLT s32 = LLT::scalar(32);
  const LLT p0 = LLT::pointer(0, 16);

  // 为 G_LOAD 和 G_STORE 定义规则
  getActionDefinitionsBuilder({TargetOpcode::G_LOAD, TargetOpcode::G_STORE})
      // 1. 标记特定内存描述符组合为合法
      .legalForTypesWithMemDesc({{s8, p0, s8, 8},   // 8-bit load/store
                                  {s16, p0, s16, 8}, // 16-bit 对齐 load/store
                                  {s32, p0, s32, 8}}) // 32-bit 对齐 load/store
      // 2. 将 type index 0 的范围限制在 [s16, s32]
      .clampScalar(0, s16, s32)
      // 3. 条件 Lower：当 memory type ≠ register type 时（如 anyext/truncstore）
      .lowerIf([=](const LegalityQuery &Query) {
        return Query.Types[0].isScalar() &&
               Query.Types[0] != Query.MMODescrs[0].MemoryTy;
      })
      // 4. 条件 Legal：特定大小
      .legalIf([=](const LegalityQuery &Query) {
        TypeSize Size = Query.Types[0].getSizeInBits();
        return Size == 16 || Size == 32;
      })
      // 5. 向量 → 标量化
      .scalarize(0)
      // 6. 其他情况 Lower
      .lower();

  // 自定义 G_MUL 的 legalization
  getActionDefinitionsBuilder(TargetOpcode::G_MUL)
      .customIf([=](const LegalityQuery &Query) {
        return !Query.Types[0].isVector() &&
               Query.Types[0].getSizeInBits() == 32;
      });

  // **必须调用** - 将规则编译为高效的查找表
  getLegacyLegalizerInfo().computeTables();
}
```

#### Type Index 的重要性

Type index 指指令类型列表中的位置：

- `G_ADD`：type index 0（result = arg1 = arg2，单一类型）
- `G_SEXT`：type index 0（输出类型），type index 1（输入类型）
- `G_LOAD`：type index 0（value type），type index 1（pointer type）
- `G_ICMP`：type index 0（result，通常是 s1），type index 1（operand type）

**查找 type index 的方法：** 查看 `llvm/include/llvm/Target/GenericOpcodes.td` 中对应指令的 `OutOperandList` 和 `InOperandList`。

#### GlobalISel Custom Legalization

```cpp
bool H2BLBLegalizerInfo::legalizeCustom(LegalizerHelper &Helper,
                                         MachineInstr &MI,
                                         LostDebugLocObserver &LocObserver) const {
  MachineIRBuilder &MIRBuilder = Helper.MIRBuilder;
  MachineRegisterInfo &MRI = *MIRBuilder.getMRI();
  GISelChangeObserver &Observer = Helper.Observer;

  switch (MI.getOpcode()) {
  case TargetOpcode::G_MUL:
    return legalizeMul(MI, MRI, MIRBuilder, Observer);
  }
}

bool legalizeMul(MachineInstr &MI, ...) {
  // 使用 mi_match 进行模式匹配
  Register PlainLHS, PlainRHS;
  bool isSigned = false;
  if (mi_match(MI.getOperand(1).getReg(), MRI, m_GSExt(m_Reg(PlainLHS))) &&
      mi_match(MI.getOperand(2).getReg(), MRI, m_GSExt(m_Reg(PlainRHS))))
    isSigned = true;

  // 通知 observer 即将修改
  Observer.changingInstr(MI);
  // 变形为 target-specific 指令
  MI.setDesc(TII.get(isSigned ? H2BLB::WIDENING_SMUL : H2BLB::WIDENING_UMUL));
  // 约束寄存器操作数（设置寄存器类、插入必要的 COPY）
  constrainSelectedInstRegOperands(MI, TII, *MRI.getTargetRegisterInfo(),
                                   *ST.getRegBankInfo());
  Observer.changedInstr(MI);
  return true;
}
```

**`GISelChangeObserver` 为何重要：** 它通知基础设施 IR 变更，使可选的 CSE map 保持更新。如果跳过 observer 通知，可能导致 CSE 产生错误结果。

## 关键机制解析（工业视角）

### SDISel 与 GlobalISel Legalization 对比

| 维度 | SDISel | GlobalISel | 工业启示 |
|------|--------|------------|---------|
| **合法类型** | 显式定义；合法类型上的操作默认合法 | 无此概念；每个 (op, type) 对必须覆盖 | MLIR 更像 GlobalISel：每个 dialect conversion 必须显式覆盖 |
| **规则描述** | `setOperationAction()` 在 TargetLowering 构造函数中 | `LegalizeRuleSet` 程序化 API | MLIR 的 `ConversionPattern` 提供类似的声明式匹配 |
| **自定义实现** | `LowerOperation()` 在 TargetLowering 中 | `legalizeCustom()` 在 LegalizerInfo 中 | 对应 MLIR 的 `RewritePattern::matchAndRewrite()` |
| **类型范围** | 通过 `addRegisterClass()` 固定 | 通过 `clampScalar()` 等动态控制 | MLIR 支持更灵活的类型控制 |
| **Pass 结构** | 嵌入在 SelectionDAGISel 内部 | 独立 `Legalizer` MachineFunctionPass | 对应 MLIR 的独立 conversion pass |
| **调试** | `debug-only=legalize-types,legalize-dag` | `debug-only=gisel-legalizer` | 都需要 step-through 调试能力 |

### 大型类型空间的 Legalization 策略

GlobalISel 面临的关键挑战：如何管理从 `i1` 到 `i1942652` 的巨大类型空间？

**解决方案组合：**
1. **`clampScalar(typeIdx, min, max)`**：限制合法类型范围
2. **`widenScalarToNextPow2()`**：自动将非法大小扩展为下一个 2 的幂
3. **`legalIf(predicate)`**：基于运行时条件判断
4. **`lowerIf(predicate)` + LegalizeMutation**：条件 lower 并可施加变换
5. **规则评估顺序：** LegalizeRuleSet 按方法调用顺序评估，第一个匹配的规则生效

### 生产级 Legalization 模式

```
模式 1：自底向上 (Bottom-up)
  - 从最具体的规则开始（如 legalForTypesWithMemDesc）
  - 逐步退化为更通用的规则（如 scalarize、lower）
  - 最后一行通常是 .lower() 作为通配 fallback

模式 2：边界管理
  - Legalization artifacts 在 def-use 链中自抵消
  - 重点关注 ABI 边界（函数参数/返回值）
  - 重点关注内存边界（load/store）
  - 中间计算可以依赖 artifacts 的自消除

模式 3：Opaque Node 策略
  - 当 generic legalization 与 custom pattern 冲突时
  - 引入 target-specific opaque SDNode/MachineInstr
  - Generic infrastructure 无法穿透 opaque node → 避免无限循环
```

## AI 编译器关联

### MLIR 的 Legalization 框架（ConversionPattern）

MLIR 的 dialect conversion 框架与 LLVM legalization 的对应关系：

```cpp
// MLIR 的 ConversionPattern - 等价于 LLVM 的 Custom Legalization
struct LinalgMatmulToGPUPattern : public OpConversionPattern<linalg::MatmulOp> {
  LogicalResult matchAndRewrite(linalg::MatmulOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const override {
    // 1. 获取操作数和类型信息（对应 LLVM 的 LegalityQuery）
    Value lhs = adaptor.getInputs()[0];
    Value rhs = adaptor.getInputs()[1];

    // 2. 构建 legal 的替代操作序列（对应 LLVM 的 Lower Operation）
    auto gpuMatmul = rewriter.create<gpu::MatmulOp>(
        op.getLoc(), /*result type*/, lhs, rhs);

    // 3. 替换原操作（对应 LLVM 的 return SDValue / changedInstr）
    rewriter.replaceOp(op, gpuMatmul.getResult());
    return success();
  }
};

// 注册 pattern（类似 LLVM 的 setOperationAction + Custom）
void populateLinalgToGPUConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<LinalgMatmulToGPUPattern>(patterns.getContext());
}
```

**核心相似性：**
- MLIR `ConversionPattern` ≈ LLVM `Custom` legalization action
- MLIR `ConversionTarget` ≈ LLVM `LegalizeRuleSet`（定义什么算合法）
- MLIR `TypeConverter` ≈ LLVM type legalization（类型转换）
- MLIR `RewritePatternSet` ≈ LLVM 的 legalization rule 集合

**核心差异：**
- MLIR 支持**部分转换**（partial conversion）和**完整转换**（full conversion）
- MLIR 的 pattern 可以**递归应用**（greedy rewriter），LLVM 通常是单次遍历
- MLIR 的 dialect conversion 可以处理多种源 dialect，LLVM 只有一种 IR

### Triton 的 Legalization 策略

Triton 编译器从高层次 Triton IR 到 PTX 的 legalization 流程：

```
Triton IR (高层次)
    │  tl.dot, tl.load, tl.store, tl.reduce
    │
    ▼
┌──────────────────────────────────────────────────┐
│ Triton-MLIR Dialect Conversion                   │
│  - tl.dot → llvm.inline_asm (PTX mma.sync)       │ ← 类似 Custom Legalization
│  - tl.load → llvm.load (coalesced access)         │ ← 类似 Load Legalization
│  - tl.atomic → llvm.cmpxchg + loop               │ ← 类似 Lower + Expand
│  - tl.reduce → shuffle + warp-level primitives    │ ← 类似 Custom
└──────────────────────────────────────────────────┘
    │
    ▼
Triton GPU IR (低层次，接近 PTX)
    │
    ▼
PTX Assembly / cubin
```

**Triton 的 legalization 特点：**
1. **Memory Coalescing Legalization**：将 `tl.load` 的跨步访问 legalize 为 coalesced 访问（2D → 1D 重排）
2. **Tensor Core Legalization**：`tl.dot` 需要根据 MMA 指令的 shape（如 `m16n8k16` on A100）进行分块 legalization
3. **Operator Lowering**：如 `tl.reduce` 通过 warp shuffle (`__shfl_xor_sync`) 实现，不是简单的 Expand
4. **Type Promotion**：fp8、int8 类型需要 promote 到更大类型用于非 Tensor Core 操作

### IREE 的多层 Legalization

IREE 的 legalization pipeline 展示了一个完整的 AI 编译器如何利用多层 legalization：

```
┌──────────────────────────────────────────────────────┐
│ Layer 1: Frontend → Linalg-on-Tensors                │
│  - tf.MatMul → linalg.matmul                         │
│  - torch.addmm → linalg.matmul + linalg.generic      │
│  Legalization: 将框架特定操作映射到标准 MLIR dialect    │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│ Layer 2: Linalg → Flow/HAL                           │
│  - linalg.matmul → flow.dispatch (tiling + fusion)   │
│  - 将 tensor 级别操作映射到 dispatch region            │
│  Legalization: tiling size、workgroup mapping         │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│ Layer 3: Flow → HAL (Hardware Abstraction Layer)      │
│  - flow.dispatch → hal.executable                    │
│  - 将 dispatch region 映射到具体硬件后端                │
│  Legalization: 后端特定类型和设备约束                   │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│ Layer 4: HAL → Target Backend                        │
│  - Vulkan/SPIR-V: hal → spirv dialect                │
│  - CUDA: hal → nvvm dialect                          │
│  - CPU: hal → llvm dialect                           │
│  Legalization: 最终目标指令集约束                      │
└──────────────────────────────────────────────────────┘
```

**IREE 对 LLVM Legalization 的借鉴：**
- 分层 legalization = 多步类型+操作 legalization
- `ConversionTarget` 的 `isLegal()` = `LegalizeRuleSet` 的 `legalIf()`
- `TypeConverter` 的 `convertType()` = SDISel 的 Type Legalization

## 示例说明

### 示例 1：SDISel Legalization 跟踪

```
输入 LLVM IR:
  %0 = fadd half %a, %b

Legalization 策略: f16 is NOT legal, f32 IS legal
  1. Type Legalization: half(f16) → Promote to float(f32)
  2. 插入 fpext 指令: %a_ext = fpext half %a to float
                      %b_ext = fpext half %b to float
  3. Operation Legalization: fadd f32 → Legal (本机支持)
  4. 插入 fptrunc 指令: %0_trunc = fptrunc float %0_ext to half
  5. fpext + fptrunc 可能被 DAGCombine 消除（如果 def-use 链允许）

最终 DAG:
  t0: f32 = fpext %a
  t1: f32 = fpext %b
  t2: f32 = fadd t0, t1
  t3: f16 = fptrunc t2
```

### 示例 2：GlobalISel 复杂类型 Legalization

```
输入 G_MIR:
  %0:_(s1) = G_ICMP intpred(eq), %a:_(s32), %b:_(s32)
  %1:_(s32) = G_ADD %0:_, %c:_                    ← 非法！s1 + s32

LegalizeRuleSet 配置:
  getActionDefinitionsBuilder(G_ADD)
    .legalFor({{s16, s16}, {s32, s32}})
    .widenScalarToNextPow2(0, 16)
    .lower();

执行:
  1. widenScalarToNextPow2: s1 → s16
     → 插入 G_ANYEXT %0:_(s1) → %ext:_(s16)
  2. G_ADD s16: 不在 {s16, s16} 中（第二个操作数是 s32）→ lower
     → 将操作拆解为更简单的指令序列
  3. 最终产生合法的 G_MIR
```

### 示例 3：AI 编译器中的 Custom Legalization 模式

```cpp
// 将 linalg.matmul legalize 为 target-specific tiled matmul + loop nest
struct MatmulLegalizationPattern {
  LogicalResult legalize(linalg::MatmulOp op) {
    // 1. 分析 tile size（从 target description 获取）
    int64_t tileM = targetInfo.getTileSize("matmul", "m");
    int64_t tileN = targetInfo.getTileSize("matmul", "n");
    int64_t tileK = targetInfo.getTileSize("matmul", "k");

    // 2. 创建 tiled loops（类似 LLVM 的 Expand）
    auto tiled = createTiledLoops(op, {tileM, tileN, tileK});

    // 3. 在 innermost tile 中插入 target-specific intrinsic
    //    （类似 LLVM 的 Custom + Opaque Node）
    auto accelMatmul = builder.create<Accelerator::TiledMatmul>(
        loc, resultType, tiled.lhs, tiled.rhs);

    // 4. 替换原始操作
    op.replaceAllUsesWith(accelMatmul);
    return success();
  }
};
```

## 总结

### 核心要点

1. **Legalization 是 AI 编译器中最核心的 pass 之一**，负责将高层抽象逐步映射到硬件可执行的指令
2. **SDISel 的两阶段法律化（Type + Operation）** 提供了清晰的分层抽象，但也限制了灵活性
3. **GlobalISel 的 LegalizeRuleSet** 提供更灵活的程序化规则定义，适合 AI 编译器等需要处理复杂类型空间的场景
4. **MLIR 的 dialect conversion 框架** 是 LLVM legalization 思想的泛化版，支持多个源/目标 dialect 的部分转换

### AI 编译器工程师的关键理解

| 概念 | LLVM 实践 | AI 编译器实践 |
|------|----------|-------------|
| 合法化规则 | `setOperationAction` / `LegalizeRuleSet` | `ConversionPattern` + `ConversionTarget` (MLIR) |
| 类型转换 | Type Legalization (SDISel) / `clampScalar` (GISel) | `TypeConverter` (MLIR) |
| 自定义合法化 | `LowerOperation` (SDISel) / `legalizeCustom` (GISel) | `OpConversionPattern::matchAndRewrite` (MLIR) |
| Opaque 节点 | Custom SDNode / target-specific MachineInstr | Custom dialect ops (MLIR) |
| Legalization artifacts | fpext/fptrunc/bitcast 等 | dialect conversion 产生的中间 dialect 操作 |
| 分层合法化 | Type → Op Legalization | Multi-stage dialect conversion (MLIR/IREE) |

### 进阶话题

- **Legalization 与 Combine 的交互**：combine 可能 undo legalization 的结果（如将 bitcast 重新折叠到 load），需要使用 opaque node 打破循环
- **Legalization 的收敛性**：需要确保 legalization 规则最终收敛到合法状态，否则陷入无限 legalization
- **性能影响**：过于 aggressive 的 legalization（如过早 scalarize 向量操作）可能严重损害性能——需要在通用性和性能之间权衡
- **调试技巧**：`-debug-only=legalize-types`（SDISel）、`-debug-only=gisel-legalizer`（GISel）；对于 MLIR，使用 `--mlir-print-ir-after-all` 跟踪每次 conversion
