# Chapter 8: Survey of the Existing Passes

## 核心概念（详细展开）

LLVM 中端（middle-end）提供了丰富的分析（analysis）和优化（optimization）passes。对于生产级编译器工程师而言，理解这些 passes 不仅是学会调用它们，更是掌握如何组合、定制和扩展它们以构建自己的优化流水线。

### Pass 的四种分类

LLVM 中端的 passes 按功能可分为四类：

1. **辅助 Passes（Helper）**：不修改 IR，用于编译器开发和调试
   - Verifier：验证 IR 的结构正确性
   - Printer：在流水线各点打印 IR 状态

2. **分析 Passes（Analysis）**：计算 IR 的特性信息，供其他 passes 查询
   - DominatorTree、LoopInfo、AliasAnalysis、TargetTransformInfo

3. **规范化 Passes（Canonicalization）**：将 IR 转换为"一致认可的表示形式"
   - InstCombine、mem2reg、LCSSA

4. **优化 Passes（Optimization）**：改进代码性能的变换
   - IPO（内联、实参提升）、Scalar（LICM、DCE、循环展开）、Vectorize

### 发现未知 Pass 的方法论

**自上而下法（opt 工具驱动）**：

```bash
opt --help                      # 列出所有可用 passes（含 legacy 和新 PM）
opt --print-passes              # 仅列出新 PM 支持的 passes
git grep "description" llvm     # 通过描述定位实现
git grep "pass-name" llvm/lib/Passes/PassRegistry.def  # 定位注册宏
```

**自下而上法（源码目录驱动）**：
- `Analysis/` → 分析 passes 和辅助类
- `Transforms/IPO/` → 跨过程优化
- `Transforms/InstCombine/` → 指令合并（最核心的规范化 pass）
- `Transforms/Scalar/` → 标量优化
- `Transforms/Vectorize/` → 向量化

**生产经验**：`opt --help` 列表中的 passes 并不都属于中端——部分属于代码生成后端，部分只被 legacy pass manager 支持。使用 `--print-passes` 过滤出新 PM 支持的 passes，再排除 `Machine XXX` 分类下的，即可获得真正的中端 passes 列表。

### 规范化（Canonicalization）的核心地位

规范化的思想是 LLVM 中端设计的基石：将同一语义的多种等价表示转换为统一的"规范形式"，使得优化 passes 只需处理这种规范形式即可。

```
非规范化的多种表示 ──→ InstCombine/mem2reg/LCSSA ──→ 统一规范形式
                                                              │
                                                              ▼
                                                      优化 passes（只需处理一种形式）
```

**为什么不在 IR 构造时就规范化？**

因为"什么是好的规范形式"是目标相关的：
- 对支持减法指令的目标：`sub a, b` 是最优的
- 对只有加法和取反的目标：`add a, (neg b)` 是最优的

因此规范化在流水线早期进行，但在引入目标特定 pass 后应该停止运行 InstCombine。

**生产陷阱**：InstCombine 在某些情况下可能"反规范化"你的目标特定 IR（例如把你的 `add + neg` 组合回 `sub`）。解决方案：1）在目标特定 pass 之前运行 InstCombine，之后不再运行；2）使用 `SimplifyXXXInst` 局部 API 代替完整的 InstCombine。

## LLVM / MLIR 流程（深入）

### LLVM 中端 Pass 流水线的典型结构

```
Input Module
  │
  ├── Helper: Verifier（检查输入 IR 合法性）
  ├── Printer（可选：打印初始状态）
  │
  ├── Canonicalization:
  │   ├── mem2reg（提升 alloca → SSA 值，所有优化的前提）
  │   ├── InstCombine（规范化 + 简单优化）
  │   └── LCSSA（循环规范化，为循环优化准备）
  │
  ├── Analysis 计算（按需，lazy evaluation）:
  │   ├── DominatorTree（支配关系）
  │   ├── LoopInfo（循环结构）
  │   ├── ScalarEvolution（归纳变量分析）
  │   ├── AliasAnalysis（指针别名）
  │   └── TargetTransformInfo（成本模型）
  │
  ├── IPO Optimizations:
  │   ├── DeadArgElim（移除无用参数）
  │   ├── ArgPromotion（指针参数 → 值参数）
  │   └── Inliner（内联函数调用）
  │
  ├── Scalar Optimizations (repeat N times):
  │   ├── InstCombine（重新规范化）
  │   ├── SimplifyCFG（CFG 简化）
  │   ├── LICM（循环不变量外提）
  │   ├── IndVarSimplify（归纳变量简化）
  │   └── DCE（死代码消除）
  │
  ├── Vectorization:
  │   ├── LoopVectorize（循环向量化）
  │   ├── SLPVectorize（直线代码向量化）
  │   └── LoadStoreVectorizer（内存访问合并向量化）
  │
  ├── Helper: Verifier（检查输出 IR 合法性）
  ├── Printer（可选：打印最终状态）
  │
  ▼
Output Module
```

### MLIR 中对应的 Pass 体系

MLIR 的 pass 体系在设计上受到了 LLVM 的深刻影响，但在以下几个方面做了演进：

| 概念 | LLVM | MLIR |
|------|------|------|
| Pass Manager | Module/CGSCC/Function/Loop PM | Op-specific PM（任意 Op 级别） |
| Pass Pipeline | `PassBuilder` + adaptors | `PassPipelineRegistration` + nesting |
| 分析缓存 | `XXXAnalysisManager` | `AnalysisManager`（per-IR-unit） |
| Canonicalization | InstCombine | `CanonicalizerPass` + `PatternRewriter` |
| 验证器 | Verifier on `Module`/`Function` | Verifier on 每个 `Operation` |
| 打印 | PrintModule/PrintFunction | `print-ir-before-all` / `-mlir-print-ir-after-all` |

**MLIR 规范化的关键差异**：MLIR 的规范化是通过声明式 Rewrite Patterns（可以在 TableGen 中定义 DRR 或直接用 C++ PatternRewriter）实现的，而非 LLVM InstCombine 的一体式模式匹配。

## 关键机制解析（工业视角）

### 分析 Passes 详解

**TargetTransformInfo（TTI）——成本模型的核心**

TTI 是优化器与目标硬件之间的关键桥梁。每个优化 pass 在做决策时查询 TTI 以估计特定 IR 序列的成本。

```cpp
// 查询指令成本
int Cost = TTI.getInstructionCost(I, TargetTransformInfo::TCK_SizeAndLatency);
// 查询向量化因子
unsigned VF = TTI.getLoadVectorFactor(VF, LoadSize, ChainSize, VecTy);
// 查询内联成本
int InlineCost = TTI.getInlineCost(CallSite);
```

**生产经验**：TTI 的值直接影响优化决策，因此错误实现的 TTI 会导致严重的性能退化：
- TTI 返回值过高 → 优化器过于保守，不执行本应有益的变换
- TTI 返回值过低 → 优化器过于激进，产生次优代码（如不必要的循环展开）
- 在 GPU 编译器上，TTI 必须正确建模 SIMT 执行模式、warp 发散惩罚、shared memory 带宽等

**AliasAnalysis（别名分析）**

AliasAnalysis 是所有涉及指针/内存优化（LICM、GVN、MemCpyOpt 等）的基础。它是一个组合分析器，aggregaes 多个子分析的结果：

```
AAManager ◀── 查询接口（AAResults）
  ├── BasicAA：基础分析（便宜、保守）
  ├── TypeBasedAA：基于 C/C++ 类型系统的别名规则
  ├── ScopedNoAliasAA：基于 noalias 属性
  └── 目标特定的 AA 实现
```

查询结果有三种：
- `NoAlias`：两个指针不可能指向同一内存
- `MayAlias`：可能别名（可能相同也可能不同）
- `MustAlias`：一定指向同一内存（精确相同）

**DominatorTree（支配树）**

支配树是 SSA 形式优化的基础分析。判断定义是否支配使用的核心 API：

```cpp
bool dominates = DT.dominates(DefBB, UseBB);
// 或者更通用的：
bool dominates = DT.dominates(Definition, User);
```

**ValueTracking 辅助类**

ValueTracking 不作为一个 pass 但被广泛使用，提供值级别的推理：

```cpp
// 推断已知比特位（哪些位一定是 0 或 1）
KnownBits KB = computeKnownBits(V, DL);
// 检查值是否为二的幂
bool isPow2 = isKnownToBeAPowerOfTwo(V, DL);
// 推断非零
bool nonnull = isKnownNonZero(V, DL);
```

**生产 bug 案例**：在 GPU 编译器上，`computeKnownBits` 曾被错误地用在线程 ID 变量上。由于 warp 内线程 ID 对所有线程是常量，该函数认为它"已知为某常量"，但实际在不同线程上该值不同。修复方式是在 `computeKnownBits` 的递归过程中检查值是否来自 `threadIdx`。

### 规范化 Passes 详解

**InstCombine（指令合并/组合器）**

InstCombine 是 LLVM 中最复杂也是最重要的 pass，包含 ~1500 个测试文件和约 32K 个 IR 测试函数。它同时做规范化（canonicalization）和优化（optimization）。

**规范化示例**：
```llvm
; 输入：inttoptr 隐式改变位宽
%c = inttoptr i64 %b to ptr
; 输出：显式截断
%b32 = trunc i64 %b to i32
%c = inttoptr i32 %b32 to ptr
```

**优化示例**：
```llvm
; 输入：恒等操作
%res = xor i64 %x, %x
; 输出：常量 0
%res = i64 0
```

**mem2reg（内存到寄存器提升）**

mem2reg 是所有 SSA 优化的前提。它消除 `alloca`/`load`/`store` 序列，代之以 SSA 值和 phi 指令。

这是绝大多数优化 passes 有效工作的前提条件。如果没有 mem2reg：
- 所有变量都是内存位置，优化器需要通过别名分析来推理
- 无法利用 def-use chains 和 SSA 支配性质
- 优化的精确度和效率都会大幅降低

**LCSSA（Loop-Closed SSA Form）**

LCSSA 确保循环内定义的值不在循环外被直接使用——通过在 exit blocks 中插入额外的 phi 节点：

```llvm
; LCSSA 前：%iv_plus_1 定义在循环内，使用在循环外
end:
  %res = add i64 %iv_plus_1, %src  ; ← 直接使用了循环内定义的值

; LCSSA 后：
end:
  %iv_plus_1.lcssa = phi i64 [%iv_plus_1, %loop]
  %res = add i64 %iv_plus_1.lcssa, %src
```

**LCSSA 对 GPU 编译器的特殊价值**：在 AMD GPU 上，循环内的值可以存储在 scalar register（所有线程共享）中，但循环外可能不同线程有不同值。LCSSA 形式的 phi 节点正好标记了这个分界点——`%iv_plus_1.lcssa` 必须放入 vector register。

### 优化 Passes 详解

**Inliner（内联器）**

内联器使用 CGSCC（Call Graph SCC）确定内联顺序，从叶子函数向根函数推进。内联决策的核心是成本效益分析：
- 成本：内联后代码增大（指令数增加）
- 效益：暴露跨过程优化机会（常量传播、死代码消除等）

TTI 通过 `getInlineCost` 控制内联决策。

**LICM（循环不变量外提）**

将循环体内不随迭代变化的值移到循环外：

```llvm
; 优化前：每次迭代都执行 load
loop:
  %offset = load i64, ptr %addr   ; ← 不变
  %iv1 = add i64 %iv, %offset

; 优化后：
entry:
  %offset = load i64, ptr %addr   ; ← 外提到循环前
loop:
  %iv1 = add i64 %iv, %offset
```

LICM 的合法性依赖于别名分析——需要确认循环体内没有可能别名的 store。

**Loop Strength Reduction（循环强度削减）**

将昂贵的地址计算替换为更便宜的形式（尤其针对目标寻址模式）：

```llvm
; 优化前：
%addr = getelementptr i64, ptr %base, i64 %idx  ; idx * 8（乘法）

; 优化后（AArch64 寻址模式友好）：
%scaled = shl i64 %idx, 3                        ; idx << 3（移位）
%addr = getelementptr i8, ptr %base, i64 %scaled  ; 无缩放
```

## AI 编译器关联

### MLIR Canonicalization Patterns

MLIR 的 `CanonicalizerPass` 是 InstCombine 的 MLIR 对应物。它通过声明式和程序化的 Rewrite Patterns 实现规范化：

```cpp
// 程序化 Pattern（C++）
struct SimplifyRedundantCast : public OpRewritePattern<CastOp> {
  LogicalResult matchAndRewrite(CastOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getOperand().getType() == op.getType()) {
      rewriter.replaceOp(op, op.getOperand());
      return success();
    }
    return failure();
  }
};
```

MLIR 的规范化优势：
- 每个 Dialect 定义自己的规范化规则（`getCanonicalizationPatterns`）
- Greedy Pattern Rewriter 自动迭代应用 patterns 直到收敛
- 支持声明式 DRR（Declarative Rewrite Rules）降低开发门槛

### XLA HLO 优化 Passes 与 LLVM 的对应关系

XLA（Accelerated Linear Algebra）是 Google 的 AI 编译器，其 HLO（High-Level Optimizer）passes 与 LLVM 中端 passes 有明确的对应：

| LLVM Pass | XLA/MLIR 对应 | 作用 |
|-----------|-------------|------|
| InstCombine | `CanonicalizerPass` | 代数简化、常量折叠 |
| Inliner | `InlinerPass` | 函数内联 |
| LICM | `Looppipeline` + `hoist_loop_invariant` | 循环不变量外提 |
| mem2reg | HLO → Linalg lowering | 缓冲区 → 值语义 |
| LoopVectorize | `VectorizePass` + layout optimization | 向量化/张量化 |
| DCE | `CSE` + `symbolic_shape_optimization` | 死代码消除 + 形状优化 |

### Triton 的优化 Pipeline

Triton 编译器在 MLIR 层面定义了一套优化 pipeline，其设计借鉴了 LLVM 的多级优化思想：

```
Triton IR
  │
  ├── TritonCombineOps（自定义规范化 + 优化）
  │   - 融合 pointwise 操作
  │   - 消除冗余 load/store
  │   - 优化 masking 模式
  │
  ├── TritonGPUCombineOps（GPU 特定优化）
  │   - shared memory promotion
  │   - 布局转换优化（blocked → MMA → dot operand）
  │   - prefetch insertion
  │
  ├── TritonGPUAccelerateMatmul（矩阵乘法加速）
  │   - dot → MMA layout 转换
  │   - padding to tile sizes
  │
  └── TritonGPUToLLVM（Triton → LLVM Dialect lowering）
      - 地址空间映射
      - barrier 同步插入
      - tensor core intrinsics 生成
```

**生产经验**：Triton 的 `TritonCombineOps` 类似于 InstCombine 的角色，但专注于 GPU 特定的优化模式。一个常见的生产 bug 是过早运行 `TritonGPUToLLVM`（在 `TritonGPUCombineOps` 之前），导致 high-level 的优化信息丢失（如 tile shape 在 LLVM Dialect 中不可见）。

### IREE 的 Lowering Pipeline 设计

IREE 的完整编译 pipeline 展示了现代 AI 编译器 pass 管理的复杂性：

```
HLO/TOSA Input
  │
  ├── GlobalOptimization Pipeline
  │   ├── canonicalizer（规范化）
  │   ├── inliner（内联）
  │   └── cse（公共子表达式消除）
  │
  ├── Flow Pipeline
  │   ├── dispatch region formation
  │   ├── fusion strategies
  │   └── shape simplification
  │
  ├── Stream Pipeline
  │   └── async scheduling, DMA planning
  │
  ├── HAL Pipeline
  │   ├── bufferization
  │   ├── device assignment
  │   └── executable translation
  │
  └── Target-specific Lowering (Vulkan SPIRV / CUDA NVVM / LLVM)
```

## 示例说明

### 示例 1：InstCombine 的规范化效果

```llvm
; Input: 非规范化减法的两种表示
%neg = sub i64 0, %c        ; 方式 1：取反 + 加法
%a = add i64 %b, %neg

%a2 = sub i64 %b, %c         ; 方式 2：直接减法

; InstCombine 输出（统一为规范形式）
%a = sub i64 %b, %c
%a2 = sub i64 %b, %c
```

### 示例 2：Loop Unroll 的编译时权衡

```llvm
; Input: 3 次迭代的循环
loop:
  %iv = phi i64 [0, ...], [%next, ...]
  %val = load i64, ptr %arr, !align
  %cond = icmp eq i64 %val, 0
  br i1 %cond, label %exit, label %continue
continue:
  %next = add i64 %iv, 1
  br label %loop

; Output: 完全展开（3 次迭代 → 3 个连续基本块）
; 每个迭代块: bb4 (i=0), bb4.1 (i=1), bb4.2 (i=2)
; 消除了 phi 和循环控制开销
; 但代码体积增大了约 3×
```

**生产决策**：完全展开的阈值（unroll-threshold）由 TTI 的 `getUnrollingPreferences` 控制。在 GPU 编译器中，这个阈值通常设置得较高，因为 warp 内的分支发散惩罚很大，展开可以减少分支。

## 总结

1. **LLVM 中端 Passes 的四层结构**：辅助 passes（调试工具）→ 分析 passes（信息提供）→ 规范化 passes（IR 形态统一）→ 优化 passes（性能改进）。

2. **InstCombine 是中端的核心**：它既是规范化器也是优化器，包含最丰富的重写 patterns。理解它是理解 LLVM 优化管道的起点。

3. **TTI 是优化器与硬件的桥梁**：正确实现 TTI 是获得有效优化的关键。GPU 编译器中 TTI 的实现尤其复杂（需要建模 SIMT 执行、warp divergence、shared memory 延迟等）。

4. **MLIR 的 Pass 体系是对 LLVM 的继承和发展**：MLIR 保留了 pass/analysis 分离的概念，但增加了 per-Operation verifier、声明式 DRR patterns、Greedy Pattern Rewriter 等创新。

5. **AI 编译器中的 Pass 设计经验**：
   - 不要过早 lowering：保持高层信息以支持更智能的优化
   - 规范化要针对目标：GPU 上的"好 IR"不同于 CPU
   - 成本模型要精确：一个错误的 TTI 值可能抵消所有其他优化的收益
   - 充分利用现成的 LLVM 优化：不要让 MLIR 重新实现 LICM 或 CSE
