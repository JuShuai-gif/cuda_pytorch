# 02 · ODS / TableGen：声明式定义算子与生成代码

> 对应代码：`include/Edge/*.td`、`include/Edge/CMakeLists.txt`、生成物
> `build/include/Edge/Edge*.{h,cpp}.inc`

---

## 1. 中文原理讲解

ODS（Operation Definition Specification）是 MLIR 用 **TableGen** 声明式定义方言/算子/类型/属性的
DSL。你写 `.td`，`mlir-tblgen` 据此生成大量 C++ 样板代码（`.inc`），再被 `.cpp` `#include` 进来。

本项目用到的生成器（在 `include/Edge/CMakeLists.txt` 手动罗列）：

| 生成器 | 产物 | 内容 |
|--------|------|------|
| `-gen-op-decls` / `-gen-op-defs` | `EdgeOps.{h,cpp}.inc` | 每个 Op 的 C++ 类：accessor、builder、verifier、parser/printer |
| `-gen-dialect-decls/defs` | `EdgeDialect.{h,cpp}.inc` | 方言类与构造 |
| `-gen-typedef-decls/defs` | `EdgeTypes.{h,cpp}.inc` | 自定义类型 `QuantTensorType` |
| `-gen-attrdef-decls/defs` | `EdgeAttrs.{h,cpp}.inc` | 自定义属性 `QuantParamsAttr` |
| `-gen-enum-decls/defs` | `EdgeEnums.{h,cpp}.inc` | `Layout` 枚举 + stringify/symbolize |

关键设计：**不用一把梭的 `add_mlir_dialect`**（它只生成 op/dialect/type，不含 attribute/enum），
而是手动逐条 `mlir_tablegen` 再 `add_public_tablegen_target(MLIREdgeIncGen)`。这给了我们对
attribute/enum 的完整控制，是 torch-mlir / IREE 的标准做法。

`assemblyFormat` 是 ODS 的精华：用一行声明式语法描述算子的文本格式，`mlir-tblgen` 自动生成
parser 与 printer。例如：
```tablegen
let assemblyFormat = "$input `,` $weight (`,` $bias^)? attr-dict `:` functional-type(operands, results)";
```
`(`,` $bias^)?` 表示可选操作数，`functional-type(operands, results)` 自动处理输入输出类型列表。

## 2. 工业背景

手写算子的 parser/printer/verifier/builder 是巨量重复劳动且易错。ODS 把"算子的契约"集中在一处，
生成一致的样板，使一个方言几十上百个算子可维护。所有基于 MLIR 的编译器（TPU-MLIR、torch-mlir、
ONNX-MLIR、IREE、Triton）都重度依赖 ODS。

## 3. TensorRT 对应模块

TensorRT 无 ODS，但其 plugin 机制（`IPluginV2`/`IPluginCreator`）要求手写 enqueue/序列化/
clone——正是 ODS 想消灭的样板。对比之下能体会声明式定义的价值。

## 4. TVM 对应模块

TVM 用 `TVM_REGISTER_NODE_TYPE` + C++ 宏注册节点，并用 Python 装饰器注册算子属性
（`@register_relay_op`）。功能类似 ODS 但更偏运行期 + 宏，而非编译期代码生成。

## 5. TPU-MLIR 对应模块

TPU-MLIR 的 `TopOps.td`/`TpuOps.td` 就是 ODS 文件，结构和本项目 `EdgeOps.td` 完全同构；阅读它
能看到工业级方言如何用 ODS 组织上百算子（含 interface、trait、自定义 builder）。

## 6. Ascend CANN 对应模块

CANN 的算子原型用 `REG_OP(...).INPUT(...).OUTPUT(...).ATTR(...).OP_END_FACTORY_REG` 这套宏 DSL
定义（IR 原型注册），思想与 ODS 一致：声明式描述算子契约，工具据此校验/生成。

## 7. 性能收益

ODS 不影响运行期性能，但**直接影响编译期与开发效率**：生成代码经过优化（如 trait 静态分派），
`assemblyFormat` 生成的 parser 比手写的更不易出错。把工程师从样板中解放出来才能去写真正的优化 pass。

## 8. Trade-off

- 声明式 `assemblyFormat` 覆盖 90% 场景，但**复杂/历史格式只能回退到 `hasCustomAssemblyFormat=1`
  手写 parse/print**（如 builtin tensor 类型本身）。
- TableGen 语法学习曲线陡、报错晦涩；生成代码不直接可读，调试需看 `.inc`。

## 9. 常见 Bug（真实踩坑）

1. **`FieldParser<double>` 未定义**：TypeDef/AttrDef 参数用裸 `double` + 声明式 assembly 会失败
   （MLIR 无浮点 FieldParser 特化）。改用 `::mlir::FloatAttr` 承载 scale。→ 见 `EdgeTypes.td`。
2. **`Couldn't find class 'AnyRankedTensor'`**：需显式 `include "mlir/IR/CommonTypeConstraints.td"`，
   新版 OpBase.td 不再传递包含它。
3. **`def X : AnyRankedTensor;` 非法**：`AnyRankedTensor` 是 def（Type 实例）不是 class，不能被继承；
   直接在算子里用 `AnyRankedTensor` 即可。
4. **可选操作数报 "expected AttrSizedOperandSegments"**：多个可选操作数需加该 trait；单个尾部可选
   操作数（如 `conv2d` 的 bias）可不加。
5. **`incomplete type 'mlir::Builder'`**：包含 `*Types/Attrs.cpp.inc` 的 `.cpp` 要先
   `#include "mlir/IR/Builders.h"`。

## 10. 调试方法

- 直接跑 `mlir-tblgen -gen-op-defs EdgeOps.td -I<includes>` 复现 `.td` 错误，比走 CMake 快。
- 读 `build/include/Edge/*.inc`：accessor/verifier/builder 的真实生成结果一目了然。
- `-print-records`：dump TableGen 解析后的所有 record，排查继承/字段问题。
- `mlir-tblgen --help` 列全部生成器；不确定某 Op 生成什么时逐个试。

## 11. Profiling 方法

- TableGen 本身很快；若 `.td` 巨大，`mlir-tblgen` 的耗时可用 `time` 观察。
- 生成代码的编译耗时用 clang `-ftime-trace` 看（大量模板实例化是 MLIR 编译慢的主因，
  本项目已启用 PCH/`Precompiled headers` 缓解）。

## 12. 在机器人 / VLA 中的应用

为机器人新算子（如自定义的传感器融合 op、动作解码 op）扩展方言时，ODS 让你只写几十行 `.td`
就得到完整的、可验证的、可 roundtrip 的算子——这对快速迭代部署侧的图变换至关重要。本项目的
`edge.attention` 就是 VLA 关注的算子用 ODS 定义的范例。
