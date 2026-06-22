# 05 · 图优化：Conv+BN+ReLU 融合 / 常量折叠 / DCE / CSE

> 对应代码：`src/Transforms/ConvBnReluFusion.cpp`、`ConstantOp::fold` +
> `EdgeDialect::materializeConstant`
> 验证：`ninja -C build check-edge`（conv-bn-relu-fusion 测试已通过）

---

## 1. 中文原理讲解

图优化是推理编译器**价值最高**的一层。本模块在 EdgeDialect 图层实现/启用四类优化：

### (1) Conv+BN+ReLU 融合（核心，已实现并验证）
匹配 `relu(batch_norm(conv2d(x, w, b)))` 单使用链；当 `weight` 与 BN 参数（scale/bias/mean/var）
均为 `edge.constant` 时，把 BN 的仿射变换**数学折叠**进卷积权重/偏置（推理期 BN 是确定的仿射变换）：

```
factor[c]   = bn_scale[c] / sqrt(bn_var[c] + eps)
new_w[c,:]  = w[c,:] * factor[c]
new_bias[c] = (b[c] - bn_mean[c]) * factor[c] + bn_bias[c]
```

随后用单个 `edge.conv_bn_relu(x, new_w_const, new_bias_const)` 替换整条链。原 conv/bn/relu 与旧常量
因都是 `Pure`，被贪心驱动的死代码消除自动清除。验证结果（scale=[2,3], var=0, eps=1 →
factor=[2,3], new_w=[2,3], new_bias=[0.5,1.0]）与手算一致。

### (2) 常量折叠（Constant Folding）
`edge.constant` 标记 `ConstantLike` + `hasFolder`，`fold()` 返回其 value 属性；方言实现
`materializeConstant` 把折叠结果物化为新 `edge.constant`。这让 `--canonicalize` 能把常量子图算掉。

### (3) DCE（死代码消除）
所有图层算子都是 `Pure`（无副作用），因此无用结果会被贪心驱动 / `--cse` / symbol-dce 自动删除。
融合后原算子正是靠这一点被清掉。

### (4) CSE（公共子表达式消除）
相同输入 + 相同属性的 `Pure` 算子会被内建 `--cse` 合并。图层算子的 `Pure` 标注是 CSE 生效的前提。

> 实现方式：自定义优化（融合）用 `OpRewritePattern` + `applyPatternsGreedily`（见 Module 06）；
> 通用优化（fold/DCE/CSE）直接复用 MLIR 内建 pass，不重复造轮子。

## 2. 工业背景

“算子融合 + BN 折叠 + 常量折叠”是**每个**推理编译器的标配，因为它们直接削减 kernel 数量、
访存量和运行期算子，且不损精度（推理期 BN 是常量仿射）。这是部署优化里性价比最高的一步。

## 3. TensorRT 对应模块

- ConvBnReluFusion ≈ TensorRT builder 的 **layer fusion**（Conv+BN+Activation 融合成一个 fused
  layer + activation），BN 折叠进 conv 的 kernel/bias。
- 常量折叠 ≈ builder 的常量预计算（weights pre-scaling）。

## 4. TVM 对应模块

- 融合 ≈ Relay `FuseOps`（按 pattern 把子图打包）+ `SimplifyInference`（把 BN 展开/折叠）。
- 常量折叠 ≈ `FoldConstant`；DCE ≈ `DeadCodeElimination`；CSE ≈ `EliminateCommonSubexpr`。

## 5. TPU-MLIR 对应模块

- ConvBnReluFusion ≈ TPU-MLIR `top` 层的 `top::ConvBnMergePattern` / BatchNorm→Scale→Conv 折叠。
- 常量折叠/DCE/CSE ≈ 其调用的 MLIR 内建规范化 + 自定义 pattern。

## 6. Ascend CANN 对应模块

- 融合 ≈ GE 的 `ConvBatchnormFusionPass` / `BufferFusionPass`（UB 融合, 减少搬运）。
- 常量折叠 ≈ GE 的 `ConstantFoldingPass`。

## 7. 性能收益

- BN 折叠后：去掉一个全张量的 BN 算子 + 一次 kernel 启动 + 一次全张量读写；ReLU 也并入。
  典型 CNN 主干上可省 10%–30% 的卷积段延迟，且**零精度损失**。
- 常量折叠把预处理/权重缩放搬到编译期，运行期算子更少。

## 8. Trade-off

- 折叠要求 weight/BN 参数是常量；动态权重（少见）无法折叠 → pattern 直接 `return failure()` 跳过。
- 融合成"大算子"减少了调度灵活性：后端必须有对应的 fused kernel，否则又得拆开（见 Module 10 lowering）。
- 过度融合可能增大单 kernel 的寄存器/共享内存压力，需结合后端代价模型（TensorRT 用 tactic 选择）。

## 9. 常见 Bug（本模块真实注意点）

1. **数值方向错误**：`new_bias = (b - mean)*factor + bn_bias`，符号/顺序写反会静默产生错误结果——
   必须有数值验证测试（本模块测试用 var=0,eps=1 让 factor 可手算）。
2. **单使用检查**：`bn->hasOneUse()` / `conv->hasOneUse()` 必须检查，否则把被多处使用的中间结果
   融掉会改变语义。
3. **元素类型/排布假设**：当前折叠假设 f32 + NCHW + per-output-channel；其他 dtype/layout 要先
   规范化或在 pattern 里判定后跳过。
4. **DCE 依赖 Pure**：若算子忘标 `Pure`，融合后旧算子不会被删，IR 里残留死算子。

## 10. 调试方法

- `--edge-fuse-conv-bn-relu --edge-ir-printer`：看融合前后 IR。
- 数值核对：用 var=0/eps=1 等可手算的输入，比对折叠出的常量。
- `--debug` / `--debug-only=greedy-rewriter`：观察 pattern 的匹配/应用/回滚。
- 融合没发生时：逐条放开 match 条件（先去掉 hasOneUse、再去掉常量检查）定位卡在哪一步。

## 11. Profiling 方法

- 融合前后跑 `--edge-statistics` 对比算子数与 MAC（MAC 不变, 但算子数下降、访存下降）。
- 端到端延迟在 Module 12 Profiler 量；融合收益主要体现在访存与 kernel 启动次数。

## 12. 在机器人 / VLA 中的应用

VLA 的视觉主干（ResNet/ViT patch embed）含大量 Conv+BN+ReLU；融合直接压低视觉编码延迟，
为控制环（10–50 Hz）腾出预算。常量折叠把相机预处理（归一化的 scale/mean）折进首层卷积，
省掉运行期预处理算子——这对多相机、高帧率的机器人管线尤其重要。

> 下一步（Module 07）：在融合后的图上做 PTQ 量化（校准 + INT8），把延迟再降一档。
