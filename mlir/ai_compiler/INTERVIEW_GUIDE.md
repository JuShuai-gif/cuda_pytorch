# Interview Guide

How to use this repository to prepare for **AI compiler / AI infra / robotics-runtime**
interviews. Each topic links to where it is implemented (or planned) in the codebase,
plus the questions you should be ready to answer.

---

## 0. The 60-second pitch

> "I built an MLIR-based graph compiler with a custom `edge` dialect that mirrors how
> TPU-MLIR and TensorRT are structured: a high-level typed op set where I do
> Conv+BN+ReLU fusion, constant folding and PTQ quantization, then a progressive
> lowering pipeline (Edge → Linalg → MemRef → Affine → LLVM), a graph-coloring memory
> planner, a small runtime, and a profiler. It's built entirely on the official MLIR
> API — ODS/TableGen, RewritePattern/ConversionPattern, PassManager, MlirOptMain —
> against a real `-fno-rtti` LLVM 23 install."

---

## 1. MLIR fundamentals (Module 01, 03)

**Where:** `include/Edge/*.td`, `src/EdgeDialect/`, `tools/edge-opt`.

Be ready to explain:
- Operation / Region / Block / Value / Type / Attribute and the SSA + nesting model.
- Why MLIR is multi-level and how dialects coexist (`func`, `arith`, `edge`, `linalg`).
- How `MLIRContext`, `DialectRegistry`, and dialect `initialize()` register ops/types.
- Trace it live: `edge-opt tests/Dialect/edge-ops.mlir --allow-unregistered-dialect`.

## 2. ODS / TableGen (Module 02, 03)

**Where:** `include/Edge/EdgeOps.td`, `EdgeTypes.td`, `EdgeAttrs.td` + generated `*.inc`.

- What `-gen-op-decls/defs`, `-gen-typedef-*`, `-gen-attrdef-*`, `-gen-enum-*` produce.
- `assemblyFormat` vs custom parser/printer; why `double` params fail declarative
  parsing (no `FieldParser<double>`) and the `FloatAttr` workaround — *a real bug we
  hit and fixed; great to discuss.*
- Traits: `Pure`, `SameOperandsAndResultType`, `ConstantLike`, and what each generates.

## 3. Passes & the pass manager (Module 04)

**Where:** `src/Transforms/` (ShapeInference, Statistics, IRPrinter).

- Analysis vs transform passes; `InferTypeOpInterface` for shape inference.
- Pass pipelines, nesting (`module` → `func`), the `--pass-pipeline` syntax.
- Pass invalidation of analyses; why ordering matters (fold before fuse).

## 4. Graph optimization & rewrites (Modules 05, 06)

**Where:** `src/Transforms/ConvBnReluFusion.cpp`, canonicalization patterns.

- **Conv+BN+ReLU fusion**: fold BN's affine transform into conv weights/bias at
  inference time → one kernel, less memory traffic. *Same optimization TensorRT does
  in its builder and TPU-MLIR does as conv+bn folding.*
- Constant folding + `materializeConstant`; DCE; CSE; canonicalization fixpoint.
- `RewritePattern` vs `ConversionPattern` (the latter tracks type conversion + legality).

## 5. Quantization (Module 07)

**Where:** `src/Quantization/`, `!edge.qtensor`, `#edge.quant_params`.

- PTQ flow: calibrate (collect activation ranges) → choose scale/zero-point → insert
  quantize/dequantize → propagate.
- MinMax vs **KL-divergence** (TensorRT's `IInt8EntropyCalibrator`) vs percentile.
- Symmetric vs asymmetric, per-tensor vs per-channel; mixed precision trade-offs.

## 6. Lowering & code generation (Modules 08, 10)

**Where:** `src/Conversion/EdgeToLinalg/`, bufferization pipeline.

- Why lower through Linalg (named ops → generic → loops); tensor vs memref domains.
- One-shot bufferization, buffer aliasing, in-place updates.
- Edge → `linalg.conv_2d_nchw_fchw` / `linalg.matmul`; attention decomposition.

## 7. Memory planning (Module 09)

**Where:** `src/MemoryPlanner/`.

- Tensor **lifetime analysis** → interference graph → **graph-coloring** allocation
  to minimize peak memory (the same problem register allocators solve).
- Why this dominates on-device deployment (DRAM/SRAM budgets on edge SoCs).

## 8. Runtime & scheduling (Modules 11, 12)

**Where:** `src/Runtime/`, `src/Profiler/`.

- `ExecutionContext`, topological scheduling, async/overlap, stream-like execution.
- Latency vs throughput; profiling: per-op latency breakdown, timeline, traces.

## 9. Vendor mapping (Modules 13–15) — the differentiator

For each pass/stage, the `notes/` connect it to:
- **TensorRT**: builder/optimizer, layer & precision fusion, engine + serialized plan,
  workspace/memory management, INT8 calibration.
- **TVM**: Relay/Relax graph IR, TE/TIR scheduling, AutoTVM/Ansor.
- **TPU-MLIR**: `top`/`tpu` dialects, calibration table, lowering to TPU kernels.
- **Ascend CANN**: GE (graph engine), TBE (kernel), ACL runtime, fusion passes.

Interviewers love "how would NVIDIA/Huawei do this differently?" — the notes answer it.

## 10. Robotics / VLA (Module 17) — for robotics-runtime roles

**Where:** `notes/17_*`, `examples/vla/`.

- VLA policy inference latency budget (control loop @ 10–50 Hz); why `attention`
  fusion and quantization matter for on-robot inference.
- Multi-camera pipeline scheduling; action-chunking latency; deterministic runtime.

---

## Likely live-coding asks (and where to practice them here)

| Ask | Practice in |
|-----|-------------|
| "Add an op to a dialect" | `include/Edge/EdgeOps.td` + `EdgeOps.cpp` |
| "Write a fusion pattern" | `src/Transforms/ConvBnReluFusion.cpp` (Module 05) |
| "Lower op X to Linalg" | `src/Conversion/EdgeToLinalg/` (Module 10) |
| "Fold a constant" | `ConstantOp::fold` + `materializeConstant` |
| "Find peak memory of a graph" | `src/MemoryPlanner/` (Module 09) |
