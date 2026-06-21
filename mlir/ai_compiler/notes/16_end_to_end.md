# 16 · 端到端编译器：完整流水线 + 报告

> 对应代码：`scripts/edge_compile.py`、`examples/end_to_end/mlp.mlir`、产物 `reports/*.md`
> 验证：`python3 scripts/edge_compile.py` → 产出 fusion/compilation/latency/memory 四份报告

---

## 1. 中文原理讲解

端到端驱动把前面所有模块串成一条可一键运行的流水线，并产出可交付的报告：

```
model.mlir (EdgeDialect)
   │ edge-opt --edge-shape-inference --edge-fuse-conv-bn-relu      → optimized.mlir
   │ edge-opt --edge-statistics (before/after)                     → Fusion Report
   │ edge-opt --edge-lower-to-llvm                                  → Compilation Report (Edge→LLVM 方言)
   │ edge-memplan --edge-align=64                                   → Memory Report (峰值/复用)
   │ edge-run --edge-fill=1.0                                       → Latency Report (per-op 延迟 + checksum)
   ▼
reports/{fusion,compilation,latency,memory}_report.md
```

验证（示例 MLP: matmul→relu→matmul→relu）：
- **Compilation**：成功降到 LLVM 方言（2 个 `llvm.func`，309 行 `llvm.*`）。
- **Latency**：4 算子，两个 matmul 占 ~90% 延迟；输出 checksum=32768（全 1 输入手算一致）。
- **Memory**：7 个张量，朴素峰值 6144 B → 规划峰值 4608 B，节省 25%。
- **Fusion**：打印优化前后算子统计（MLP 无 conv+bn+relu，故无变化——诚实反映）。

这条流水线对标 `trtexec`（一键 build+profile）/ TPU-MLIR `model_deploy` / CANN `atc`：把"优化决策 +
后端编译 + 性能/内存评估"封装成单命令，是工程交付的标准形态。

## 2. 工业背景

真实编译器都提供"一键编译 + 报告"的驱动（trtexec / atc / tvmc / model_deploy）。报告是部署决策的依据：
能否上板（内存）、能否达控制频率（延迟）、优化是否生效（fusion）、后端是否打通（compilation）。

## 3–6. 厂商对应

- TensorRT：`trtexec --dumpProfile --dumpLayerInfo` 一键 build engine + 报告 → 本驱动的对应物。
- TVM：`tvmc compile/run`。
- TPU-MLIR：`model_transform` + `model_deploy`（带校准表）。
- Ascend CANN：`atc`（Ascend Tensor Compiler）一键把模型编译成 om + profiling。

## 7. 性能收益

驱动本身不提速，但它把"优化—编译—评估"闭环自动化，使每次改动都能快速量化收益（延迟/内存/算子数），
是性能工程可持续迭代的基础设施。

## 8. Trade-off

- 一键驱动方便，但隐藏了中间步骤；调优时仍需能逐 pass dump（`edge-opt --mlir-print-ir-after-all`）。
- 报告用合成输入（全 1）测延迟，绝对值仅供相对比较；真实延迟需用代表性输入 + warmup + 多次取均值。

## 9. 常见 Bug

1. **工具路径**：驱动需找到 `build/bin` 下的工具；未先 `ninja` 会报缺工具（脚本已检查）。
2. **不可运行算子**：`edge-run` 只支持 constant/relu/matmul；含 conv 的模型 latency 阶段会跳过/告警
   （应先 lower 或用可运行子集）。
3. **报告解析脆弱**：从工具 stdout 提取统计要稳健（脚本用 `module` 行作 IR 起始分隔）。

## 10. 调试方法

- 逐阶段单独运行脚本里的命令，定位是哪一步失败。
- `reports/optimized.mlir` 保留了优化后 IR，可直接喂回 `edge-opt`/`edge-run` 复查。

## 11. Profiling 方法

- Latency Report 即 per-op 延迟分解；Memory Report 即峰值/复用；Fusion Report 即算子数变化。
- 三份报告合起来回答："改动后更快了吗？更省内存了吗？融合生效了吗？"

## 12. 在机器人 / VLA 中的应用

把 VLA 模型喂给本驱动, 一键得到"延迟分解 + 峰值内存 + 是否打通后端"的部署前体检报告, 据此判断
能否落进控制环预算与边缘内存。这是从"编译器"走向"可交付部署工具链"的关键一步, 也是面试中展示
"端到端工程能力"的最佳载体。

> 至此 17 个模块全部打通：从 EdgeDialect 图层 IR, 经形状推断/融合/lowering/内存规划/运行时/profiling,
> 到一键端到端驱动与报告, 并与 TensorRT/TVM/TPU-MLIR/Ascend CANN 逐项对标。
