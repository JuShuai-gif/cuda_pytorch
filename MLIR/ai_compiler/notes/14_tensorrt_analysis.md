# 14 · TensorRT 架构分析

> TensorRT 是 NVIDIA 的闭源推理编译器/运行时。虽不开源，但其架构是 GPU 推理部署的事实标准，
> 也是本项目很多设计的对标对象。本笔记梳理其架构并与 edge_ai_compiler_pro 对照。

---

## 1. 整体流程

```
ONNX / API 搭网 → INetworkDefinition (layer 图)
        │ Builder + BuilderConfig (精度/tactic/workspace)
        ▼
   优化 (融合 + 精度选择 + tactic 自动调优)
        ▼
   ICudaEngine (序列化为 .plan / .engine)
        │
        ▼
   IExecutionContext.enqueueV3(stream)  在 GPU 上执行
```

## 2. Builder

- `IBuilder` + `IBuilderConfig`：输入 layer 图与约束（FP16/INT8、workspace 上限、DLA、动态 shape 的
  optimization profile），输出优化后的 engine。
- **tactic 自动调优**：对每个 layer 实测多种 kernel 实现（cuDNN/cuBLAS/cutlass/内置），选最快的——
  这是 TensorRT 的核心竞争力，本质是"基于实测的自动调度"。
  ↔ 本项目：`edge-opt` 的 pass 流水线对应 builder 的优化阶段，但本项目无 tactic 实测调优（用静态 lowering）。

## 3. Optimization（图优化）

- 常量折叠、死层消除、concat/slice 化简、维度/精度推导。
  ↔ 本项目 `edge-shape-inference` + 常量折叠 + DCE/CSE（Module 04/05）。

## 4. Fusion（融合，TensorRT 的招牌）

- **垂直融合**：Conv+Bias+Activation → 单 CBR kernel；Conv+BN 折叠。
  ↔ 本项目 `edge-fuse-conv-bn-relu`（同款 BN 折叠 + Conv+BN+ReLU 融合）。
- **水平融合**：把多个共享输入的同类 layer（如多个 1x1 conv）合并成一个大 kernel。
- **精度融合**：在融合边界插入/消除 reformat（NCHW↔NHWC↔NC4HW4↔NC8HW8）以喂满 Tensor Core。
  ↔ 本项目 `Layout` 枚举建模了布局选择问题（实际 reformat 优化待做）。
- **Flash-Attention 风格融合**：把 MHA 融成单 kernel。↔ 本项目 `edge.attention` 一等算子为此预留。

## 5. Engine

- `ICudaEngine`：优化后的、绑定到具体 GPU 架构 + 精度 + tactic 的产物，可序列化（.plan）。
  注意：engine 与 GPU 架构/TensorRT 版本强绑定（不可跨架构复用）。
  ↔ 本项目 Module 10 lower-to-llvm 的产物（LLVM IR/机器码）类比 engine，但目标是 CPU。

## 6. Runtime

- `IRuntime.deserializeCudaEngine` 加载 engine；`IExecutionContext` 持有运行期状态与 workspace；
  `enqueueV3` 在 CUDA stream 上异步调度 fused kernel 序列；支持多 context 并发、多 stream 重叠。
  ↔ 本项目 `edge-run` 的 ExecutionContext/GraphExecutor（同步、CPU、解释执行）。

## 7. Memory Management

- **workspace**：一块预分配的临时显存，所有 layer 的中间张量在其中复用（builder 规划好复用方案）。
  ↔ 本项目 Module 09 `edge-memplan`（生命周期 + 图着色复用，同思想）。
- 权重常驻显存；激活在 workspace 复用；动态 shape 用 profile 预留最大尺寸。

## 8. INT8 量化

- PTQ：`IInt8Calibrator`（Entropy/MinMax/Legacy）前向收集激活分布，选 per-tensor scale；
  builder 在量化与 fp16/fp32 间按精度-性能权衡选每层精度（隐式量化）；新版支持显式量化（Q/DQ 节点）。
  ↔ 本项目 Module 07（规划）：KL/MinMax/百分位校准对标 `IInt8EntropyCalibrator2`。

## 9. 与本项目的对照总结

| 能力       | TensorRT                  | edge_ai_compiler_pro                |
| ---------- | ------------------------- | ----------------------------------- |
| 图优化/融合 | 垂直+水平+精度融合, 闭源  | Conv+BN+ReLU 融合 + 折叠/DCE/CSE     |
| kernel 选择 | tactic 实测自动调优       | 静态 lowering（无实测调优）          |
| 后端       | CUDA/Tensor Core/DLA      | CPU（Linalg→LLVM）                  |
| 内存       | workspace 复用            | edge-memplan 图着色复用              |
| 运行时     | 异步多流 enqueueV3        | 同步解释执行 edge-run                |

## 10. 学习/调试方法

- `trtexec --verbose --dumpProfile --dumpLayerInfo`：看融合后的 layer、各层精度与耗时。
- Nsight Systems/Compute：看 kernel 时间线、Tensor Core 利用率、显存带宽。
- 启发：本项目可引入"基于实测的 kernel 选择"作为 tactic 的简化版（对 matmul 试 tile 大小）。
