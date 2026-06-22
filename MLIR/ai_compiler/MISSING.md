# AI Compiler Project Missing Work

本文档记录当前 `ai_compiler` 项目相对“可系统学习并接近真实 AI compiler 工程”的缺失项。
后续可以按本文件中的编号逐项补齐。

## 如何使用本文件驱动 Codex 补齐

后续你可以直接复制下面的 prompt 给 Codex：

```text
请根据 /home/hpc/ghr_code/cuda_pytorch/mlir/ai_compiler/MISSING.md 补齐其中的 M0。
要求：
1. 先阅读 MISSING.md 中对应模块和相关源码；
2. 按当前项目风格实现，不做无关重构；
3. 补必要测试或验证命令；
4. 最后说明改了哪些文件、如何验证、还剩什么缺口。
```

把 `M0` 换成你要补的模块即可，例如：

```text
请根据 /home/hpc/ghr_code/cuda_pytorch/mlir/ai_compiler/MISSING.md 补齐 M2 Edge Dialect Verifier。
```

```text
请根据 /home/hpc/ghr_code/cuda_pytorch/mlir/ai_compiler/MISSING.md 先补 M8 Runtime 正确性，只处理 unsupported op 必须 fail 和相关测试。
```

```text
请根据 /home/hpc/ghr_code/cuda_pytorch/mlir/ai_compiler/MISSING.md 补 M4 Edge -> Linalg Lowering，优先实现 conv2d lowering，并补 lit 测试。
```

如果你只想让我分析方案、不改代码，可以这样说：

```text
请根据 /home/hpc/ghr_code/cuda_pytorch/mlir/ai_compiler/MISSING.md 分析 M6 Quantization Pass 的实现方案，先不要改代码。
```

## 总体结论

当前项目适合作为 MLIR AI compiler 的入门骨架，已经覆盖：

- MLIR CMake 项目结构
- ODS/TableGen 定义 dialect、op、type、attr
- 简单 graph pass 与 rewrite pattern
- 简单 Edge -> Linalg lowering
- lit/FileCheck 测试框架
- 量化、内存规划、runtime、profiler 的概念演示

但它还不是完整 AI compiler。主要问题是：文档声明超过实际实现、构建依赖本机硬编码路径、lowering/runtime/quantization/memory planner 没有形成闭环。

## M0. 构建与可复现性

### 现状

- 顶层 CMake 默认写死 MLIR 路径：`/home/ghr/code/llvm-project/install/lib/cmake/mlir`
- `llvm-lit` 路径也写死到 `/home/ghr/code/llvm-project/build/bin/llvm-lit`
- 当前 workspace 没有现成 `build/` 产物，README 中的 verified 状态不能直接复现

### 缺失

- 自动发现或文档化 `MLIR_DIR`
- 清晰的 build prerequisites
- 可复现的 clean checkout 构建流程
- CI 或最小本地验证脚本

### 补齐标准

- `cmake -S . -B build -DMLIR_DIR=...` 能在当前机器成功配置
- `ninja -C build edge-opt edge-run edge-memplan edge-quantize` 能通过
- `ninja -C build check-edge` 能跑通
- README 中不再声称无法复现的 verified 状态

### 优先级

P0。没有可复现构建，后续补功能无法稳定验证。

## M1. README 与项目状态修正

### 现状

README 声称：

- 17/17 modules done
- Quantization done/tested
- Memory planner done/tested
- Runtime + Profiler done/tested
- End-to-end driver verified
- lit 10/10 passing

实际代码中缺少或不完整：

- `src/Quantization/`
- `src/MemoryPlanner/`
- `src/Profiler/`
- 完整 Edge -> Linalg lowering
- runtime 对 conv/attention/fused op 的执行支持
- quantization pass 和 IR 改写闭环

### 缺失

- README 需要区分“已实现”、“概念演示”、“计划补齐”
- `task.json` 需要反映真实状态
- notes/interview guide 中引用不存在目录的位置需要修正

### 补齐标准

- README 状态表与源码一致
- 不存在的目录不再被标为 done
- 每个模块都有准确入口文件

### 优先级

P0。文档误导会影响学习路径。

## M2. Edge Dialect Verifier

### 现状

`EdgeOps.td` 定义了 `conv2d`、`batch_norm`、`matmul`、`attention` 等 op，但大多数 op 没有严格 verifier。

例如：

- `matmul` 没检查 K 维一致
- `conv2d` 没检查 input/weight rank、channel/group、stride/pad/dilation 合法性
- `batch_norm` 没检查 scale/bias/mean/variance 与 channel 数一致
- `attention` 没检查 query/key/value rank 和 hidden dim

### 缺失

- `hasVerifier = 1`
- C++ verifier 实现
- invalid IR 测试

### 补齐标准

- 非法 shape 能在 parse/verify 阶段报错
- 每个核心 op 至少有 valid + invalid lit 测试
- verifier 错误信息可读

### 优先级

P0。没有 verifier，IR 合法性无法保证。

## M3. Shape Inference 完整性

### 现状

已有简单 shape inference：

- `conv2d` 推 NCHW 输出
- `batch_norm` 输出等于输入
- `attention` 输出等于 query
- `matmul` 简单替换最后一维

### 缺失

- dynamic shape 更完整处理
- matmul batch broadcasting
- attention 输出 shape 正确推导
- group convolution shape/check
- shape inference 与 verifier 的边界定义

### 补齐标准

- rank-2 matmul、batched matmul 都有测试
- dynamic dim 不被错误静态化
- shape inference 不掩盖非法 IR

### 优先级

P1。

## M4. Edge -> Linalg Lowering

### 现状

`edge-lower-to-linalg` 只实现：

- `edge.constant -> arith.constant`
- `edge.relu -> linalg.generic`
- `edge.matmul -> linalg.matmul`，且仅 rank-2

未实现：

- `edge.conv2d`
- `edge.batch_norm`
- `edge.attention`
- `edge.conv_bn_relu`
- quantized tensor lowering

### 缺失

- conv2d lowering 到 `linalg.conv_2d_nchw_fchw` 或等价 generic
- batch_norm lowering
- fused conv_bn_relu lowering
- attention lowering 的教学版实现
- 完整 conversion legality

### 补齐标准

- 所有 Edge op 都能 lowering 到非 Edge dialect
- `edge-lower-to-llvm` 后不残留 `edge.*`
- conversion 测试覆盖每个 op

### 优先级

P0。lowering 是 AI compiler 主链路。

## M5. Bufferization 与 LLVM Pipeline

### 现状

项目注册了 `edge-lower-to-loops` 和 `edge-lower-to-llvm` pipeline，但依赖 `edge-lower-to-linalg` 的完整性。

### 缺失

- 端到端 pipeline 测试确认最终无 Edge op
- bufferization 失败 case 处理
- memref/call convention 说明
- LLVM dialect 输出示例

### 补齐标准

- `tests/Conversion/lower-pipeline.mlir` 覆盖 matmul、relu、conv
- pipeline 输出能通过 FileCheck 验证关键 dialect
- 对 unsupported op fail，而不是静默残留

### 优先级

P1。

## M6. Quantization Pass

### 现状

`edge-quantize` 是独立 CLI 演示：

- 使用合成 calibration 数据
- 统计 MinMax / Percentile / KL
- 对 `edge.constant` 做 per-tensor SQNR 报告
- 估算 INT8 speedup

它没有真正改写 IR。

### 缺失

- quantization dialect/type 使用闭环
- quantize/dequantize op 或等价表达
- weight quantization IR rewrite
- activation calibration 输入
- per-channel weight quantization
- mixed precision decision 写回 IR

### 补齐标准

- 新增 `edge-quantize-weights` 或 `edge-apply-quantization` pass
- 输入 f32 constant，输出 int8 constant + quant params
- lowering/runtime 能识别量化表达
- lit 测试检查 IR 中出现量化类型或量化属性

### 优先级

P1。

## M7. Memory Planner 闭环

### 现状

`edge-memplan` 能分析 tensor SSA 值生命周期并打印 offset 表。

问题：

- 结果没有写回 IR
- 没接入 bufferization
- 没接入 runtime arena
- 只统计 static ranked tensor

### 缺失

- memory planning pass 或 metadata attr
- 与 memref allocation 的关系
- runtime 使用 planned offset
- alias/in-place/update 语义
- dynamic shape fallback

### 补齐标准

- 内存规划结果能以 attr/report 双形式输出
- runtime 或 lowering 能消费 planner 结果
- 测试验证生命周期不重叠的 tensor 复用 offset

### 优先级

P1。

## M8. Runtime 正确性

### 现状

runtime 是解释器，支持：

- `edge.constant`
- `edge.relu`
- `edge.matmul`

不支持 op 时 warning 并跳过。

### 缺失

- unsupported op 应该 fail
- conv2d kernel
- batch_norm kernel
- fused conv_bn_relu kernel
- attention kernel
- shape/type runtime check
- 输出 correctness tests

### 补齐标准

- 遇到 unsupported op 返回 failure
- 每个已定义核心 op 都有 runtime kernel 或明确不支持
- `edge-run` 测试不只检查报告标题，还检查 checksum

### 优先级

P0。静默跳过 unsupported op 会误导测试结果。

## M9. Profiler 完整性

### 现状

Profiler 记录每个解释执行 op 的 wall time 和 output bytes。

### 缺失

- timeline
- memory peak
- warmup/repeat
- min/median/p95
- kernel category
- report 与 benchmark 脱钩

### 补齐标准

- `edge-run --repeat=N --warmup=M`
- 输出 total、per-op avg、min、max
- 可选 JSON/Markdown report

### 优先级

P2。

## M10. End-to-End Driver

### 现状

`scripts/edge_compile.py` 串联多个工具生成报告，但每个工具之间缺少真实 IR/metadata 闭环。

### 缺失

- 每一步输出明确 artifact
- 失败时停止
- 检查最终 IR 无 Edge op
- 检查 runtime 结果
- 支持配置 pipeline

### 补齐标准

- 一个 example 可以从 Edge IR 跑到 lowered IR、memory report、runtime report
- 任一步失败返回非零
- 生成的 reports 与实际命令一致

### 优先级

P1。

## M11. Tests

### 现状

已有 lit/FileCheck 测试，但多数测试偏 smoke test。

### 缺失

- invalid verifier tests
- lowering no-Edge-op tests
- runtime checksum tests
- quantization IR rewrite tests
- memory planner offset correctness tests
- end-to-end regression tests

### 补齐标准

- 每个模块至少包含正例、反例、边界例
- FileCheck 不只检查标题，也检查关键 IR/数值/错误

### 优先级

P0。

## M12. AI Compiler 学习覆盖缺口

即使补完上述内容，本项目仍不覆盖完整工业 AI compiler 的所有要求。以下主题需要额外学习：

- ONNX / Torch / TOSA importer
- graph canonicalization 与 pattern benefit
- layout transform：NCHW/NHWC、blocked layout
- operator legalization
- tiling/fusion/codegen
- vectorization
- GPU lowering：LLVMGPU/NVVM/ROCDL
- schedule search/autotuning
- cost model
- dynamic shape runtime
- kernel library dispatch
- real calibration dataset
- numerical accuracy validation
- deployment packaging

## 推荐补齐顺序

1. M0 构建与可复现性
2. M1 README 与状态修正
3. M2 Edge op verifier
4. M8 runtime unsupported op fail
5. M4 conv2d/relu/matmul lowering 完整化
6. M11 测试增强
7. M6 quantization pass
8. M7 memory planner 闭环
9. M10 end-to-end driver
10. M9 profiler 增强

## 后续可直接使用的任务句式

- “根据 `MISSING.md` 补 M0”
- “根据 `MISSING.md` 补 M2 verifier”
- “根据 `MISSING.md` 补 M4 conv2d lowering”
- “根据 `MISSING.md` 先修 README，不改代码”
- “根据 `MISSING.md` 给 runtime unsupported op 改成 fail”
- “根据 `MISSING.md` 给每个模块补 lit 测试”
