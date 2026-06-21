# 15 · 华为 Ascend CANN 编译栈分析

> CANN（Compute Architecture for Neural Networks）是华为昇腾 NPU 的软件栈。本笔记梳理其
> 编译/运行架构（GE / TBE / ACL / Runtime），并与 edge_ai_compiler_pro 对照。

---

## 1. 整体架构

```
TensorFlow/PyTorch/ONNX/MindSpore
        │ (前端解析 + IR 转换)
        ▼
   GE (Graph Engine): ComputeGraph 图优化         ← 对应本项目 EdgeDialect + 图优化 pass
        │ 算子选择 / 融合 / 内存分配 / 图切分
        ▼
   TBE (Tensor Boost Engine): 算子实现 (基于 TVM)  ← 对应本项目 Edge→Linalg→kernel
        │ 编译为 AI Core 指令
        ▼
   om 模型 (offline model)
        │
        ▼
   ACL (AscendCL) Runtime 在 NPU(AI Core) 上执行
```

## 2. GE（Graph Engine）

- 核心数据结构 `ComputeGraph`：`Node`(≈Op) + `Anchor`/`Edge`(≈use-def)。
  ↔ 本项目 MLIR 的 Operation + Value/use-def（`edge-introspect` 可遍历）。
- 职责：图优化、算子融合、常量折叠、shape 推导、内存分配、图切分（多 NPU/多核）。
- 优化以 `GraphPass`/`FusionPass` 形式注册（`GraphFusionPass`、`BufferFusionPass`、`ConstantFoldingPass`…）。
  ↔ 本项目的 `OpRewritePattern` + PassManager（`edge-fuse-conv-bn-relu` ≈ ConvBatchnormFusionPass）。

## 3. TBE（Tensor Boost Engine）

- 算子实现层，**基于 TVM**：用 DSL（TBE DSL / TIK）描述算子计算与调度，编译为 AI Core 指令。
- 支持自定义算子（custom op）：写 TBE 算子原型（`REG_OP` 宏 DSL）+ 实现 + tiling。
  ↔ 本项目 `Edge→Linalg`（结构化算子）+ MLIR 的 tiling/vectorize，思想一致（TBE 用 TVM，我们用 Linalg）。

## 4. ACL（AscendCL）/ Runtime

- `aclmdlLoadFromFile` 加载 om；`aclmdlExecute`/`aclmdlExecuteAsync` 执行；管理 stream、event、device 内存
  （`aclrtMalloc`）、H2D/D2H 拷贝。
  ↔ 本项目 `edge-run` 的 ExecutionContext/GraphExecutor（同步、CPU）。

## 5. Graph Engine 优化流水线（重点）

- **算子融合**：UB（Unified Buffer）融合——把多个算子的中间结果留在片上 UB, 不回 GMEM（如
  Conv+BN+ReLU、Conv+Eltwise）。这是降带宽的关键，与 TensorRT 垂直融合同理。
  ↔ 本项目 `edge-fuse-conv-bn-relu`（图层融合）+ 未来的"留在片上"语义。
- **内存分配**：`MemoryAssigner`/`BlockMemAssigner` 做连续内存块分配与复用。
  ↔ 本项目 Module 09 `edge-memplan`（生命周期 + 图着色复用）。
- **格式转换**：在算子间插入 `TransData`（NCHW↔NC1HWC0 等昇腾专用 5D 格式）以匹配 AI Core。
  ↔ 本项目 `Layout` 枚举建模了这一类格式选择问题。
- **shape 推导**：每个算子注册 `INFER_FUNC`，GE 在编译期调用。
  ↔ 本项目 `ShapeInferenceOpInterface::inferShapes`（Module 04，几乎同名同构）。

## 6. 量化

- AMCT（Ascend Model Compression Toolkit）做 PTQ/QAT：校准收集分布 → INT8 量化 → 部署。
  ↔ 本项目 Module 07（规划）。

## 7. 与本项目的对照

| 维度       | CANN                          | edge_ai_compiler_pro            |
| ---------- | ----------------------------- | ------------------------------- |
| 图层 IR    | GE ComputeGraph               | EdgeDialect (MLIR)              |
| 融合       | UB 融合 / GraphFusionPass     | edge-fuse-conv-bn-relu          |
| shape 推导 | INFER_FUNC                    | ShapeInferenceOpInterface       |
| 算子实现   | TBE（基于 TVM）               | Edge→Linalg（MLIR 结构化）      |
| 内存       | MemoryAssigner                | edge-memplan（图着色）           |
| 运行时     | ACL（NPU, 异步 stream）       | edge-run（CPU, 同步）            |
| 格式       | NC1HWC0 5D + TransData        | Layout 枚举（建模）              |

## 8. 工程启发

- **片上 buffer 融合（UB fusion）**：把"减少 GMEM 往返"作为融合的首要目标——本项目融合 pass 可加上
  "中间结果是否能留在片上"的代价考量。
- **专用格式 + 自动插 TransData**：布局是一等优化对象；本项目 `Layout` 枚举是起点，可加布局传播 pass。
- **shape INFER_FUNC 注册机制** 与本项目 OpInterface 完全同构, 印证设计方向正确。

## 9. 学习/调试方法

- 读 CANN 的 `fusion_pass` 样例与 `REG_OP` 算子原型，理解工业级算子注册/融合写法。
- `msprof` 看 AI Core 利用率、内存带宽，定位是算力瓶颈还是带宽瓶颈。
