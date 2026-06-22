# 10 · Lowering 流水线：Edge → Linalg → MemRef → Affine/SCF → LLVM

> 对应代码：`src/Conversion/EdgeToLinalg/EdgeToLinalg.cpp`（`edge-lower-to-linalg`）
> 验证：`ninja -C build check-edge`（edge-to-linalg 测试已通过）

---

## 1. 中文原理讲解

MLIR 的精髓是**渐进式 lowering**：每一步只降低一个抽象层级，且独立可测。本编译器的后端路径：

```
EdgeDialect ──(edge-lower-to-linalg, 本模块)──> Linalg(on tensors)
            ──(one-shot-bufferize)──> Linalg(on memref) + MemRef
            ──(convert-linalg-to-loops)──> Affine/SCF + MemRef
            ──(convert-scf-to-cf, convert-to-llvm)──> LLVM Dialect ──> LLVM IR
```

### 本模块实现：Edge → Linalg（已验证）
用 **dialect-conversion 框架**（`OpConversionPattern` + `ConversionTarget` +
`applyPartialConversion`）把图层算子降级到 Linalg。Linalg 采用 **destination-passing style
(DPS)**：结果张量由 `outs` 提供的 init 张量承载。

- `edge.constant` → `arith.constant`（dense 属性直接复用）
- `edge.relu`     → `tensor.empty` + `linalg.generic`（body: `arith.maximumf(x, 0)`）
- `edge.matmul`   → `tensor.empty` + `linalg.fill(0)` + `linalg.matmul`

`ConversionTarget` 把这些算子标记为非法（`addIllegalOp`），把 linalg/tensor/arith/func 标记为
合法，驱动器据此把非法算子重写为合法算子。**关键**：pattern 里用 `adaptor.getXxx()` 取已转换的
操作数，而非原始操作数。

### 为什么经 Linalg？
Linalg 是 MLIR 的"结构化算子"层：它把循环结构、并行性、访存模式用 indexing map + iterator
type 显式表达，从而能复用 MLIR 海量的现成变换（tiling、fusion、vectorization、bufferization、
lowering 到 loops/LLVM）。我们只需写"Edge→Linalg"这一段，后面全部复用 MLIR 标准 pass，不重复造轮子。

### 下游（标准 MLIR pass，配合 Module 08 bufferization）
Linalg-on-tensors 经 one-shot-bufferize 变成 Linalg-on-memref（Module 08），再
`convert-linalg-to-loops` 得到 Affine/SCF + MemRef，最后 `convert-to-llvm` 到 LLVM 方言，
经 `mlir-translate` 出 LLVM IR。

## 2. 工业背景

“高层图 → 结构化中层 → 循环/向量 → 目标 IR”是所有现代张量编译器的通用骨架。MLIR 把中层
标准化为 Linalg，使不同前端（ONNX/Torch/Edge）共享同一套后端，这正是 IREE 的核心思想。

## 3. TensorRT 对应模块

TensorRT 的 builder 把 layer 图直接 lowering 到 tactic（具体 CUDA/cuDNN/cuBLAS kernel），相当于
跳过通用中层、直接选 kernel。我们的 Linalg 层对应"未绑定具体 kernel 的结构化表达"，可再下降到
LLVM/GPU 或对接厂商库。

## 4. TVM 对应模块

- Edge→Linalg ≈ Relay/Relax → TE（Tensor Expression）的 lowering。
- Linalg 的 indexing map/iterator ≈ TE 的 compute + schedule；tiling/vectorize ≈ TVM schedule 原语。

## 5. TPU-MLIR 对应模块

- Edge→Linalg ≈ TPU-MLIR 的 `top` → `tpu` lowering（也用 ConversionPattern）。
- 之后 `tpu` 再 lowering 到后端 kernel ≈ 我们的 Linalg→loops→LLVM（只是目标不同）。

## 6. Ascend CANN 对应模块

- Edge→Linalg ≈ GE 图 → TBE 算子的下降；TBE 本身基于 TVM TE。
- 结构化中层的角色 ≈ CANN 里 AKG（基于 polyhedral 的自动调度）的输入。

## 7. 性能收益

lowering 本身不提速，但选对中层（Linalg）能**免费**获得 MLIR 的 tiling/vectorization/fusion，
这些才是 kernel 级提速的来源。DPS 风格还便于后续 in-place bufferization，减少内存分配。

## 8. Trade-off

- 经通用 Linalg 通用性强，但**可能不如直接对接厂商库（cuDNN/cuBLAS/AscendCL）极致**；生产里常
  混合：能融合的走 Linalg codegen，标准卷积/GEMM 直接调库。
- DPS 需要显式 init 张量（`tensor.empty`+`fill`），代码更啰嗦，但换来可 bufferize、可 in-place。
- partial conversion 只降已实现的算子，未实现的算子需保证后续有 pattern，否则后端报非法算子。

## 9. 常见 Bug（本模块真实注意点）

1. **用了原始操作数而非 `adaptor`**：dialect conversion 中操作数可能已被替换，必须用 `adaptor.getXxx()`。
2. **结果非静态 shape**：Linalg named op 需要静态 shape 才能建 `tensor.empty`；务必先跑
   `edge-shape-inference`（Module 04）。本模块 pattern 对动态 shape `return failure()`。
3. **DPS init 忘了 fill**：matmul/conv 的 init 必须先 `linalg.fill(0)`，否则累加到未初始化内存 → 错误结果。
4. **ConversionTarget 漏标合法方言**：忘了 `addLegalDialect<tensor>` 会导致新建的 `tensor.empty` 被判非法。

## 10. 调试方法

- `--edge-lower-to-linalg --mlir-print-ir-after-all`：看每步 IR。
- `--debug-only=dialect-conversion`：观察 legality 判定与 pattern 选择/回滚。
- 验证语义：lower 后用 `mlir-opt --convert-linalg-to-loops` 看循环是否合理。
- 全链路：`edge-opt ... | mlir-opt --one-shot-bufferize --convert-linalg-to-loops ... --convert-to-llvm | mlir-translate --mlir-to-llvmir`。

## 11. Profiling 方法

- `--mlir-timing` 看各 lowering pass 耗时。
- lower 到 LLVM 后用 `mlir-runner` / 链接执行做端到端延迟（Module 11/12）。

## 12. 在机器人 / VLA 中的应用

把 VLA 策略网络 lowering 到 Linalg 后，可针对机器人 SoC 做 tiling/vectorization 或对接 NPU 库，
生成低延迟、可静态分析内存的部署包。`edge.attention` 未来可选择"整体融合 kernel"或"分解为
matmul+softmax+matmul 走 Linalg"两条路，按硬件择优——这是控制环延迟优化的关键决策点。

> 下一步（Module 08/09）：one-shot-bufferize 把 tensor 降到 memref, 再做生命周期分析与
> graph-coloring 内存复用, 打通到 LLVM 并量化峰值内存。
