# 03 · EdgeDialect 图层方言设计

> 对应代码：`include/Edge/*.td`、`src/EdgeDialect/*.cpp`、`tools/edge-opt`
> 验证方式：`ninja -C build check-edge`（lit + FileCheck 已通过）

---

## 1. 中文原理讲解

`EdgeDialect` 是本编译器的**图层 IR（graph-level IR）**：用一组带语义的、命名清晰的算子
（`conv2d` / `batch_norm` / `relu` / `matmul` / `attention` / `conv_bn_relu` / `constant`）
表达神经网络计算图。它处在前端（ONNX/PyTorch）与底层算子 IR（Linalg）之间。

为什么要自定义这一层，而不是直接用 Linalg/TOSA？因为**融合与量化决策需要"语义级"算子**。
一旦把 `conv` 拆成 `linalg.generic` + 循环，再想识别"这是一个能和后面 BN 融合的卷积"就要做
昂贵的模式匹配。保留一个高层 typed op 集合，让融合 pass 变得简单、快速、可验证——这正是
TensorRT 在 layer 图上工作、TPU-MLIR 保留 `top` 方言的根本原因。

工程要点：
- **算子的输入/输出统一用内建 `RankedTensorType`**，使得 lowering 到 Linalg/TOSA 无摩擦；
  量化信息用专门的自定义类型 `!edge.qtensor` 和自定义属性 `#edge.quant_params` 承载，而不是
  去重载内建 tensor。
- `edge.constant` 标记为 `ConstantLike` 且可折叠，方言实现了 `materializeConstant`，从而支持
  常量折叠与规范化时"物化"新常量（见 Module 05）。
- `edge.conv_bn_relu` 是融合的**目标算子**（一等公民），而不是给 conv 挂个布尔属性——这样代价
  模型与 lowering 都更显式。
- `Layout` 枚举（NCHW/NHWC）建模了"数据布局选择"问题。

方言定义关键开关（`EdgeDialect.td`）：`useDefaultTypePrinterParser`、
`useDefaultAttributePrinterParser`、`hasConstantMaterializer`，以及手动声明的
`registerTypes()` / `registerAttributes()`。ODS 生成在 `include/Edge/CMakeLists.txt` 里**手动**
罗列全部 `mlir_tablegen`（op/dialect/type/attr/enum），而不用一把梭的 `add_mlir_dialect`——
因为后者不生成 attribute/enum。这是 torch-mlir / IREE 的标准做法。

## 2. 工业背景

所有推理编译器都有一层"图 IR"：TensorRT 的 `INetworkDefinition`（layer 图）、TVM 的 Relay/Relax、
TPU-MLIR 的 `top`、ONNX-MLIR 的 `onnx` 方言、torch-mlir 的 `torch` 方言。它们的共性是：**先在
高层做图变换（融合/折叠/量化/布局），再逐级下降到 kernel**。`EdgeDialect` 就是这一层的最小可用实现。

## 3. TensorRT 对应模块

- `EdgeDialect` ≈ TensorRT `INetworkDefinition` + `ILayer`（高层 layer 图）。
- `edge.conv_bn_relu` ≈ TensorRT builder 自动做的 Conv+BN+Activation 融合（生成融合后的单 layer）。
- `Layout` 枚举 ≈ TensorRT 的 `TensorFormat`（kLINEAR/kCHW4/kHWC8…），builder 会自动插入 reformat。
- `!edge.qtensor` / `#edge.quant_params` ≈ TensorRT 的 INT8 dynamic range / per-tensor scale。

## 4. TVM 对应模块

- `EdgeDialect` ≈ Relay/Relax 的高层算子（`nn.conv2d`、`nn.batch_norm`…）。
- 融合目标算子 ≈ Relay 的 `FuseOps`（按融合规则把子图打包成 `Function`，再交给 TE/TIR）。
- 常量折叠 ≈ Relay `FoldConstant`。

## 5. TPU-MLIR 对应模块

- `EdgeDialect` ≈ TPU-MLIR 的 `top` 方言（框架无关的高层算子）。
- conv+bn 折叠 ≈ TPU-MLIR 的 `top` 层 BN→Scale→Conv 融合。
- `!edge.qtensor` ≈ TPU-MLIR 的量化类型 + calibration table（threshold/scale）。
- `top` 之后下降到 `tpu` 方言 ≈ 我们的 `EdgeToLinalg`（下降到可生成 kernel 的层）。

## 6. Ascend CANN 对应模块

- `EdgeDialect` 图 ≈ GE（Graph Engine）的 `ComputeGraph`。
- 融合 pass ≈ GE 的 `GraphFusionPass` / `BufferFusionPass`（如 Conv+BN+ReLU UB 融合）。
- 下降到 kernel ≈ TBE（基于 TVM 的算子实现）。
- 运行时 ≈ ACL（AscendCL）。

## 7. 性能收益

图层算子本身不产生性能，**收益来自它使能的优化**：
- Conv+BN+ReLU 融合：减少 2 次全张量读写 + 2 次 kernel 启动，典型可省 10%–30% 的卷积段延迟。
- 常量折叠：把 BN 参数、reshape 常量在编译期算掉，去掉运行期算子。
- 保留高层语义使 `attention` 能整体下降为 FlashAttention 融合 kernel（否则拆开后访存爆炸）。

## 8. Trade-off

- **高层算子越多 → 表达力越强，但 lowering 与维护成本越高**。需要克制：只为"值得特化的融合/量化
  机会"定义一等算子（如 `conv_bn_relu`），其余靠通用算子 + pattern。
- 自定义类型 `!edge.qtensor` 提升类型安全，但**所有 pass 都要处理它**，增加适配面；因此核心算子
  仍用内建 tensor，量化类型只在量化边界出现。
- 用 `FloatAttr` 存 scale（而非裸 `double`）牺牲了一点简洁度，换来声明式 assembly 可用（见下）。

## 9. 常见 Bug（本模块真实踩坑）

1. **`FieldParser<double>` 未定义**：在 TypeDef/AttrDef 里用 `"double":$scale` 做声明式
   `assemblyFormat` 会编译失败——MLIR 的 `FieldParser` 只为 Attribute 派生类、整型、`std::string`、
   container、`AffineMap` 提供特化，**没有浮点特化**。修复：把 `scale` 改为 `::mlir::FloatAttr`
   （属于 Attribute 派生，命中特化）。这是非常典型的 ODS 面试题。
2. **`Couldn't find class 'AnyRankedTensor'`**：新版 MLIR 中 `OpBase.td` 不再传递包含
   `CommonTypeConstraints.td`，必须显式 `include "mlir/IR/CommonTypeConstraints.td"`。
3. **`incomplete type 'mlir::Builder'`**：生成的 `get()` builder 需要完整 `Builder` 定义，
   `*.cpp` 里要 `#include "mlir/IR/Builders.h"`。
4. **ABI 不匹配**：本地 MLIR 是 `-fno-rtti` 静态库，工程必须经 `HandleLLVMOptions` 同步关闭 RTTI，
   否则链接期符号/ABI 冲突。ASan 同理（本地库未开 → 默认 `EDGE_ENABLE_ASAN=OFF`）。

## 10. 调试方法

- **Roundtrip 自检**：`edge-opt file.mlir | edge-opt`，输出与输入一致即 parser/printer 正确。
- **看生成代码**：`build/include/Edge/EdgeOps.{h,cpp}.inc` 是 TableGen 产物，accessor/verifier 一目了然。
- **`mlir-tblgen` 直接跑**：复现 `.td` 错误，配 `-I` 把 include 路径补全。
- **`--mlir-print-op-generic`**：用通用形式打印，绕开自定义 assemblyFormat，定位 print 问题。
- **`--debug-only=dialect-conversion`**：后续 lowering 时观察 pattern 匹配与 legality。

## 11. Profiling 方法

- 编译期：`--mlir-timing` 看每个 pass 耗时；`-ftime-trace`（clang）看 TableGen/编译热点。
- IR 规模：`StatisticsPass`（Module 04）统计各算子数量，量化融合前后对比。
- 运行期延迟在 Module 12 的 Profiler 里做（per-op breakdown / timeline）。

## 12. 在机器人 / VLA 中的应用

VLA（Vision-Language-Action）策略网络通常是 ViT/Transformer + 动作头。`edge.attention` 作为一等
算子，使我们能：
- 把多头注意力整体下降为 **FlashAttention 风格融合 kernel**，这是把控制环延迟压进 10–50 Hz 预算的关键。
- 对 KV / 线性层做 INT8/混合精度量化（`!edge.qtensor`），在 Jetson/Ascend 边缘 SoC 上换取吞吐。
- 多相机输入用 `Layout` 选择 NHWC 以匹配硬件卷积单元，减少 reformat。

> 下一步（Module 04/05）：在此方言上实现 `ShapeInferencePass` 与 `ConvBnReluFusion`，把"图层 IR
> 使能优化"的价值真正兑现。
