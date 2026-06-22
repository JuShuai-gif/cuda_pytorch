# 06 · 改写模式：RewritePattern / 规范化 / ConversionPattern

> 对应代码：`src/Transforms/ConvBnReluFusion.cpp`（`OpRewritePattern` + 贪心驱动）、
> `ConstantOp::fold`、各算子的 `Pure` 规范化语义

---

## 1. 中文原理讲解

MLIR 的所有 IR 改写都建立在 **Pattern** 之上。两大类：

### (1) `RewritePattern` / `OpRewritePattern<OpT>`（同方言内改写）
实现 `matchAndRewrite(op, PatternRewriter&)`：匹配成功就通过 `rewriter` 增删改算子。
**必须用 `rewriter` 的 API**（`replaceOp` / `replaceOpWithNewOp` / `eraseOp` / `create`），
而不是直接改 IR——因为驱动器要追踪 use-def 变化、维护 worklist、支持回滚。本模块的
`ConvBnReluFusionPattern : OpRewritePattern<ReluOp>` 就是范例：匹配 relu→bn→conv 链，
`replaceOpWithNewOp<ConvBnReluOp>(...)` 一步替换。

### (2) 改写引擎（Rewrite Driver）
- **Greedy driver**（`applyPatternsGreedily`）：不断对 worklist 里的算子尝试所有 pattern，
  直到不动点；同时做 fold + 死代码消除。本模块融合 pass 用的就是它。
- **Dialect Conversion driver**（`applyFullConversion` / `applyPartialConversion`）：用于**跨方言**
  lowering，配 `ConversionPattern` + `TypeConverter` + `ConversionTarget`(legality)。见 Module 10。

### (3) Canonicalization（规范化）
`--canonicalize` 把 IR 收敛到规范形：调用各算子的 `fold()`（如 `ConstantOp::fold`）、`getCanonicalizationPatterns`、
以及通用清理。结合 `Pure` 语义自动做 DCE/常量传播。把"局部恒等式"（如 `x+0`、`reshape∘reshape`）
写成 canonicalization pattern，是保持 IR 干净、让后续 pass 简单的关键工程习惯。

### (4) `ConversionPattern`（lowering 专用，预览）
与 `RewritePattern` 的区别：它接收**已转换的操作数**（`adaptor.getXxx()`）和 `TypeConverter`，
配合 `ConversionTarget` 声明哪些算子合法/非法，驱动器据此把非法算子逐步重写为合法算子。
Module 10 的 EdgeToLinalg 会用它。

### (5) PDL / PDLL（声明式 pattern）
MLIR 还支持用 `.pdll` 声明式写 pattern，由 `mlir-pdll` 生成。适合大量结构化匹配，TPU-MLIR/部分项目
用它替代手写 C++ pattern。本项目暂用 C++ pattern（更直观、易调试）。

## 2. 工业背景

“匹配子图 → 重写”是图编译器的通用机制：融合、代数化简、layout 变换、量化插入、lowering 全靠它。
能写出**正确、收敛、可组合**的 pattern 是编译器工程师的核心技能。

## 3. TensorRT 对应模块

TensorRT 不暴露 pattern API，但 builder 内部就是一套 layer 匹配/重写规则（fusion patterns +
tactic 选择）。其 plugin 也是"匹配不识别的子图 → 用自定义 kernel 替换"。

## 4. TVM 对应模块

- `RewritePattern` ≈ Relay 的 `DFPatternRewrite`（`rewrite_call` + `DFPattern`）。
- canonicalization ≈ `SimplifyExpr` / 各种 `Legalize` pass。
- PDLL ≈ TVM 的 pattern language（DFPattern DSL）。

## 5. TPU-MLIR 对应模块

TPU-MLIR 直接用 MLIR `OpRewritePattern` + greedy driver 实现 `top`/`tpu` 层的所有融合/化简；
跨方言 lowering（top→tpu）用 `ConversionPattern`。与本项目机制完全一致。

## 6. Ascend CANN 对应模块

GE 的各种 `*FusionPass` 内部就是"图模式匹配 + 子图替换"，对应 RewritePattern 的思想；
TBE 的算子选择对应 lowering 阶段的 ConversionPattern + 代价模型。

## 7. 性能收益

pattern 本身是编译期机制；收益来自它实现的优化（融合/化简）。工程上，贪心驱动的复杂度取决于
pattern 数与 worklist 收敛轮数，正确设置 `PatternBenefit`（优先级）能减少无效尝试。

## 8. Trade-off

- 贪心驱动简单但可能**不收敛/震荡**（两个 pattern 互相撤销对方）——必须保证 pattern 单调推进。
- C++ pattern 直观易调试，但量大时维护成本高；PDLL 更声明式但调试更难。
- `replaceOpWithNewOp` 方便，但要保证新算子结果类型与原结果一致，否则破坏下游。

## 9. 常见 Bug

1. **不用 rewriter 直接改 IR**：绕过驱动器会导致 worklist/use-def 失配，出现崩溃或漏改。
2. **pattern 不收敛**：A 把 X→Y，B 把 Y→X，greedy 驱动死循环（本项目用 guard 上限 + 单调设计避免）。
3. **忘记检查 `hasOneUse`/支配关系**：融合多使用值会改变语义。
4. **`applyPatternsGreedily` 返回 failure 未处理**：应 `signalPassFailure()`，否则错误被吞。
5. **ConversionPattern 里用了原始操作数而非 `adaptor`**：lowering 时类型已变，必须用 adaptor。

## 10. 调试方法

- `--debug-only=greedy-rewriter`：打印每个 pattern 的 match/apply/rollback。
- `--debug-only=dialect-conversion`：lowering 时看 legality 与 pattern 选择。
- `--mlir-print-ir-after=<pass>` 配合 `--edge-ir-printer` 观察改写前后。
- 收敛问题：临时把 greedy 的最大迭代次数调小, 看 IR 是否震荡。

## 11. Profiling 方法

- `--mlir-pass-statistics` 看各 pattern 触发次数（自定义 pattern 可加 `LLVM_DEBUG` 计数）。
- `--mlir-timing` 看规范化/转换 pass 的耗时占比；pattern 过多时是编译瓶颈。

## 12. 在机器人 / VLA 中的应用

部署侧常需为特定硬件写定制 pattern：把某段子图替换成厂商融合 kernel（如 FlashAttention、
NPU 专用 conv），或插入量化/反量化对。掌握 RewritePattern 就能把"针对机器人 SoC 的图改写"
做成可组合、可测试的 pass，而不是一次性脚本。本项目的融合 pattern 即为模板。

> 下一步：用 `ConversionPattern` 把 EdgeDialect lowering 到 Linalg（Module 10），打通到 LLVM 的后端路径。
