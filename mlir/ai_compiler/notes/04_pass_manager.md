# 04 · Pass 基础设施：Analysis / Transform / PassManager

> 对应代码：`include/Edge/Passes.td`、`include/Edge/ShapeInferenceOpInterface.td`、
> `src/Transforms/{ShapeInference,Statistics,IRPrinter}.cpp`
> 验证：`ninja -C build check-edge`（shape-inference / statistics 测试已通过）

---

## 1. 中文原理讲解

MLIR 把"对 IR 的处理"抽象成 **Pass**，由 **PassManager** 编排成流水线。两类 Pass：
- **Analysis Pass / Analysis**：只读，产出分析结果（如形状信息、活跃区间、统计量），可被
  缓存与失效管理（`getAnalysis<T>()` / `markAnalysesPreserved()`）。
- **Transform Pass**：改写 IR（融合、折叠、lowering）。

本模块实现三个 pass（用 ODS `Passes.td` + `-gen-pass-decls` 生成基类与注册函数，这是 MLIR 内建
pass 的标准写法）：

1. **`edge-shape-inference`**（Transform，作用域 `func::FuncOp`）：基于自定义
   `ShapeInferenceOpInterface`，做**定点迭代**——只要算子的操作数已是 ranked，就调用其
   `inferShapes()` 把结果类型里的动态维 `?` 细化为静态维，并沿 use-def 链传播，直到不再变化。
   已验证：`matmul(4x8,8x16)->? ` 推断为 `4x16`，再喂给下一个 matmul 传播出 `4x32`；
   `conv2d` 按 `(H+pb+pe-d*(k-1)-1)/s+1` 算出 `224->112`。

2. **`edge-statistics`**（Analysis 风格，作用域 `ModuleOp`）：遍历图统计各算子数量，对 conv/matmul
   估算 MAC（乘加次数），输出 Markdown 报告。已验证 MAC 估算与手算一致（conv 118M + matmul）。

3. **`edge-ir-printer`**（调试 Transform）：带标签横幅打印当前 IR，比 `--mlir-print-ir-after-all`
   更可控，可放在流水线任意位置。

**自定义 OpInterface 是关键设计**：把"如何推断形状"的知识下放到每个算子（`Conv2DOp::inferShapes`
等），pass 本身只负责调度。这正是 MLIR 可扩展性的核心——加新算子只需实现接口，pass 不用改。

## 2. 工业背景

静态 shape 是推理编译器的"地基"：没有它，内存规划、tiling、量化、kernel 选择都无从下手。
ONNX/TensorRT/TVM/TPU-MLIR 都有独立的 shape inference 阶段。统计/profile 则是部署前评估算力、
定位瓶颈的标准工具。

## 3. TensorRT 对应模块

- shape inference ≈ TensorRT 的 `IShapeLayer` + builder 的维度推导；dynamic shape 用
  optimization profile（min/opt/max）约束。
- statistics ≈ `IEngineInspector` / verbose builder log 打印的 per-layer 维度与 FLOPs。

## 4. TVM 对应模块

- shape inference ≈ Relay 的 type inference（`relay.transform.InferType`），它同时做类型与 shape。
- statistics ≈ `relay.analysis.get_total_mac_number` / `relay.analysis.count_layers`。

## 5. TPU-MLIR 对应模块

- shape inference ≈ TPU-MLIR `top` 层的 `shape_infer` interface（每个 Op 实现 `shape_inference()`），
  与本模块的 `ShapeInferenceOpInterface::inferShapes()` 几乎同名同构。
- statistics ≈ TPU-MLIR 的 `model_tool --info` 输出的算子/算力信息。

## 6. Ascend CANN 对应模块

- shape inference ≈ GE 的 `InferShape`（每个算子注册 `INFER_FUNC`，由 GE 在图编译期调用）。
- statistics ≈ Profiling 工具链里的算子级 FLOPs/耗时统计。

## 7. 性能收益

- 形状推断本身是编译期一次性开销，但它**使能**后续所有优化；缺了它，融合/量化/内存复用全停摆。
- 把推断逻辑放进 OpInterface + 定点迭代，复杂度 O(算子数 × 迭代轮数)，DAG 上通常 2-3 轮收敛。

## 8. Trade-off

- 定点迭代简单稳健，但**最坏情况多轮扫描**；大图可改成基于 use-def 的 worklist（只重算受影响的下游）。
- OpInterface 下放推断知识，灵活但**每个算子都要正确实现**，错误会静默传播错误 shape（见下）。

## 9. 常见 Bug（本模块真实注意点）

1. **`setType` 后下游/return 类型不匹配**：改了某 Value 的类型，若 func 签名或 `return` 仍是旧类型，
   verifier 报错。测试里用 `test.sink`（未注册算子，无类型约束）承接，避免签名回填问题；真实流水线里
   要配合"函数签名也参与推断"或在末尾统一 reconcile。
2. **动态维判断**：必须用 `ShapedType::isDynamic(d)` / `ShapedType::kDynamic`，不要拿 `-1` 硬比。
3. **walk by interface**：`func.walk([](ShapeInferenceOpInterface op){...})` 只回调实现了该接口的算子；
   忘了给算子加 `DeclareOpInterfaceMethods<...>` 会导致 pass"什么都没做"。
4. **接口 .cpp.inc 必须恰好编译一次**：`ShapeInferenceOpInterface.cpp.inc` 只在 `EdgeOps.cpp` include
   一次，否则重复定义。

## 10. 调试方法

- `--mlir-print-ir-after=edge-shape-inference`：看该 pass 后的 IR。
- 在流水线里插 `--edge-ir-printer`（带 `label`）观察任意步骤前后。
- `--mlir-pass-statistics` / `--mlir-timing`：看 pass 运行与耗时。
- pass 不生效时：先确认算子是否真的实现了接口（`--debug-only=...` 或临时打印）。

## 11. Profiling 方法

- `edge-statistics` 直接给算子计数与 MAC，作为"模型算力体检"。
- `--mlir-timing` 看哪个 pass 是编译瓶颈；shape inference 多轮迭代是常见热点。

## 12. 在机器人 / VLA 中的应用

机器人模型常含 dynamic batch / 变长序列（多帧、变长指令）。`edge-shape-inference` 在给定具体输入
形状后把图固化为静态 shape，是把 VLA 策略网络编译成低延迟、可预测（无运行期分配）部署包的前提。
`edge-statistics` 则用于在 Jetson/Ascend 上线前评估算力是否落在控制环（10–50 Hz）的预算内。

> 下一步（Module 06/05）：用 `RewritePattern` 实现规范化，再做 Conv+BN+ReLU 融合与常量折叠——
> 把"图层 IR 使能优化"的价值兑现成实际的算子数/访存下降。
