# 01 · MLIR 基础：Operation / Region / Block / Type / Attribute / SSA

> 对应代码：`tools/edge-opt`、`tests/Dialect/edge-ops.mlir`（可直接观察 IR 结构）
> 后续工具：`tools/edge-introspect`（Module 01 计划，遍历/打印/分析 IR）

---

## 1. 中文原理讲解

MLIR 的 IR 是一棵"可嵌套的 SSA 图"，核心概念只有几个，但组合出极强的表达力：

- **Operation（操作）**：IR 的基本单元。一个 Op 有：操作数（Operands，都是 SSA Value）、结果
  （Results，也是 Value）、属性（Attributes，编译期常量）、零个或多个 **Region**，以及所属
  方言与名字（如 `edge.conv2d`）。注意：在 MLIR 里**几乎一切都是 Op**——`func.func`、
  `module`、甚至常量都是 Op。
- **Region（区域）**：Op 内部承载的"代码块容器"，由一串 **Block** 组成。`func.func` 的函数体、
  `scf.for` 的循环体都是 Region。
- **Block（基本块）**：一串顺序执行的 Op，带 **Block Arguments**（MLIR 用块参数代替传统 PHI 节点）。
- **Value（值）**：SSA 值，要么是某个 Op 的结果，要么是块参数。每个 Value 恰有一个定义点
  （def），可被多处使用（use）——这就是 **use-def 链**。
- **Type（类型）**：描述 Value 的静态类型（`tensor<1x3x224x224xf32>`、`!edge.qtensor<...>`）。
- **Attribute（属性）**：编译期常量信息，挂在 Op 上（`strides = array<i64: 2, 2>`、
  `#edge.quant_params<...>`）。

**SSA + 嵌套 Region** 是 MLIR 区别于 LLVM IR 的关键：LLVM IR 是扁平的函数级 CFG，MLIR 允许任意
层级的嵌套（module 套 func 套 scf.for 套 ...），这让"高层结构化控制流"和"低层 CFG"能在同一框架里共存。

可在已验证的 IR 上直观观察这些概念：`build/bin/edge-opt tests/Dialect/edge-ops.mlir
--allow-unregistered-dialect`——输出的 `module { func.func { edge.conv2d ... return } }` 就是
"Op 含 Region，Region 含 Block，Block 含 Op，Op 之间用 SSA Value 连接"。

## 2. 工业背景

任何编译器的第一课都是"IR 长什么样、怎么遍历"。能否高效地 walk IR、查 use-def、改写而不破坏
SSA，是编译器工程师的基本功。MLIR 把这套基础设施标准化（`Operation::walk`、`Value::getUses`、
`OpBuilder`、`IRRewriter`），省去每个项目自造一遍。

## 3. TensorRT 对应模块

TensorRT 不暴露通用 IR，但 `INetworkDefinition`/`ILayer`/`ITensor` 就是它的"Op/Value/Type"：
`ILayer` ≈ Operation，`ITensor` ≈ Value，`getInput/getOutput` ≈ operands/results。遍历网络做
分析（如统计层数、找融合机会）对应 MLIR 的 `walk`。

## 4. TVM 对应模块

TVM Relay/Relax 的 `Expr`/`Call`/`Var` 对应 Op/Value；TIR 的 `Stmt`/`PrimExpr` 是更低层的 IR。
TVM 的 `ExprVisitor`/`ExprMutator` 对应 MLIR 的 walk + rewrite。

## 5. TPU-MLIR 对应模块

TPU-MLIR 直接用 MLIR，因此概念完全一致：它的 `top`/`tpu` 方言 Op 同样是 Operation，分析/改写也用
MLIR 的 walk + RewritePattern。读 TPU-MLIR 源码就是读这些基础设施的真实用法。

## 6. Ascend CANN 对应模块

GE 的 `ComputeGraph` 由 `Node`（≈ Op）和 `Anchor`/`Edge`（≈ use-def 连接）组成；`NodePtr`、
`GetInDataNodes()` 等接口就是 GE 版的 IR 遍历。

## 7. 性能收益

IR 基础设施本身不直接提速，但**遍历/分析的复杂度决定了 pass 的可扩展性**。MLIR 的 `walk` 是
O(N) 单遍；use-def 链是 O(1) 取用户。写 pass 时避免重复全图扫描，是大模型图编译可接受编译时延的前提。

## 8. Trade-off

- 嵌套 Region 表达力强，但**分析要处理任意嵌套**（如做活跃性分析要跨 Region），比扁平 CFG 复杂。
- "一切皆 Op"统一了基础设施，但新手容易对 `module`/`func` 也是 Op 感到困惑。

## 9. 常见 Bug

- **在 walk 过程中删除/插入 Op** 导致迭代器失效。应使用 `walk` 的返回值控制，或收集后批量改写，
  或用 `IRRewriter` + greedy driver。
- **破坏 SSA 支配关系**：把一个 Value 的使用移到其定义之前 → verifier 报 "does not dominate"。
- **忘记 verify**：手工建 IR 后应 `module.verify()`，否则错误会在后续 pass 里以诡异方式爆出。

## 10. 调试方法

- `op->dump()` / `value.dump()` 即时打印任意 Op/Value。
- `--mlir-print-ir-after-all` / `--mlir-print-ir-before-all`：每个 pass 前后 dump IR。
- `--mlir-print-debuginfo`：打印 Location，定位 IR 来源。
- `op->getParentOfType<func::FuncOp>()` 等沿嵌套向上查询。

## 11. Profiling 方法

- `--mlir-timing` 看 pass/分析耗时。
- 对自写分析，用 `llvm::TimeTraceScope` 打点，配 `-ftime-trace` 在 Chrome trace 里看热点。

## 12. 在机器人 / VLA 中的应用

机器人推理图通常不大但要求**编译可重复、可分析**：要能快速 walk 出"有多少 attention / conv"、
"哪些张量最大"，为后续的延迟/内存优化提供依据。Module 01 的 `edge-introspect` 工具就是把这些
遍历/统计封装成命令行，作为部署前的"图体检"。
