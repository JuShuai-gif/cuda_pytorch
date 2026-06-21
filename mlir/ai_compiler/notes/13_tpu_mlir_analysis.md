# 13 · TPU-MLIR 源码架构分析（与本项目对比）

> TPU-MLIR 是算能（Sophgo）开源的、基于 MLIR 的 TPU 编译器。它是本项目最直接的对标，
> 架构同源（MLIR），但落到真实 TPU 后端。本笔记分析其架构并与 edge_ai_compiler_pro 逐项对比。

---

## 1. 整体架构

```
ONNX/Caffe/TFLite/PyTorch
        │ (前端转换, Python)
        ▼
   top 方言 (框架无关高层算子, fp32)         ← 对应本项目 EdgeDialect
        │ calibration (校准表) + lowering
        ▼
   tpu 方言 (硬件相关, 量化/排布已定)         ← 对应本项目 Edge→Linalg 之后的"绑定后端"层
        │ codegen
        ▼
   bmodel / cvimodel (后端指令 + 权重)
        │
        ▼
   runtime (bmrt / model_runner) 在 TPU 上执行
```

## 2. Dialect

- **top**：框架无关的高层算子（Conv/MatMul/Softmax/Attention…），fp32 语义，做图优化与融合。
  ↔ 本项目 `EdgeDialect`（conv2d/matmul/relu/attention/conv_bn_relu）。
- **tpu**：硬件相关算子，已确定量化类型、数据排布、是否切分到 LMEM。
  ↔ 本项目暂以 Linalg + 标准 lowering 代替"绑定后端"层（未做真实 TPU 后端）。

## 3. Passes

- top 层：`top::ConvBnMerge`、shape inference、常量折叠、形状/排布规范化。
  ↔ 本项目 `edge-fuse-conv-bn-relu`（BN 折叠）、`edge-shape-inference`、canonicalize/cse。
- lowering：`ConvertTopToTpu`（带量化决策的 ConversionPattern）。
  ↔ 本项目 `edge-lower-to-linalg`（ConversionPattern，但目标是通用 Linalg）。

## 4. Quantization

- 流程：`run_calibration`（前向收集激活范围，产出 calibration table）→ `model_deploy` 带表 lowering 到
  量化 `tpu` 算子；支持 INT8（per-channel 权重）、INT4、混合精度、F16/BF16。
  校准支持 KL（默认）/MAX/百分位。
  ↔ 本项目 Module 07（规划中）：`!edge.qtensor`/`#edge.quant_params` 承载量化参数，校准算法对标其 KL。

## 5. Backend

- tpu 算子 → 后端指令（BM168x/CV18xx），含 LMEM（片上）/GMEM（片外）地址分配、指令调度、DMA 编排。
  ↔ 本项目用 MLIR 标准 `convert-linalg-to-loops` + `convert-to-llvm`（CPU 路径），未做 TPU ASIC 后端。

## 6. Runtime

- `bmrt`（BM runtime）/ `model_runner`：加载 bmodel，管理输入输出、stream、内存，调用后端执行。
  ↔ 本项目 `edge-run`（解释执行）/ Module 10 的 lower-to-llvm（编译执行）。

## 7. 工程决策（值得学习）

- **双方言分层**（top/tpu）：高层做框架无关优化，低层绑定硬件——干净的关注点分离。本项目沿用此理念。
- **校准与部署分离**：先产出 calibration table，再带表部署，可复现、可调参。
- **大量复用 MLIR 基础设施**：ODS/Pattern/PassManager 全官方，与本项目一致——证明"不自造框架"是工业正确路径。

## 8. 与本项目的差距（诚实评估）

| 维度       | TPU-MLIR             | edge_ai_compiler_pro          |
| ---------- | -------------------- | ----------------------------- |
| 高层方言   | top（成熟、算子全）  | EdgeDialect（核心算子，可扩展）|
| 后端       | 真实 TPU ASIC 指令   | CPU（Linalg→LLVM）/ 解释执行   |
| 量化       | 完整 PTQ + 校准表    | Module 07 规划中               |
| 排布优化   | LMEM 切分、自动调度  | 暂无（依赖 MLIR 通用变换）     |
| 成熟度     | 生产级               | 教学/面试级，架构同源          |

## 9. 调试 / 学习方法

- 读 `lib/Dialect/Top/Transforms` 看融合/折叠 pattern 的工业写法。
- 读 `lib/Conversion/TopToTpu` 看带量化决策的 ConversionPattern。
- 用 `tpuc-opt`（其 opt 工具，等价本项目 `edge-opt`）逐 pass dump IR 观察 lowering。

## 10. 对本项目的启发（路线图）

1. 引入 `tpu` 风格的"后端绑定方言"，把量化类型/排布固化（Module 07 后）。
2. 把 `edge-fuse-conv-bn-relu` 扩成 pattern 库（更多融合：conv+add、matmul+gelu）。
3. 校准与部署分离：产出 calibration table（JSON），部署时带表 lowering。
