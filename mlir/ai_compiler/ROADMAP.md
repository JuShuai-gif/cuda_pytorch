# Roadmap

The project is built in phases. Each phase produces code that **compiles, has a
FileCheck/GoogleTest test, and a Chinese engineering note** in `notes/`. The
authoritative per-module deliverable list and status live in [`task.json`](task.json).

Legend: ✅ done · 🚧 in progress · ⬜ planned

## Phase 1 — Foundation (the verified spine) ✅

- ✅ Repo skeleton + `task.json` master plan
- ✅ CMake build system (C++20, Ninja, reuses local MLIR install, `-fno-rtti` ABI
  match, opt-in ASan, lld)
- ✅ **Module 03 — `EdgeDialect`** via ODS/TableGen: `conv2d`, `batch_norm`, `relu`,
  `matmul`, `attention`, `conv_bn_relu`, `constant`; custom type `!edge.qtensor`;
  custom attribute `#edge.quant_params`; `Layout` enum. *Roundtrip-verified.*
- ✅ `edge-opt` driver (MlirOptMain, all dialects + passes)
- ✅ lit + FileCheck regression harness (`ninja check-edge`)
- 🚧 Top docs (README / ARCHITECTURE / ROADMAP / INTERVIEW_GUIDE) + notes 01–03

## Phase 2 — Analysis & transforms

- ⬜ **Module 01 — mlir_fundamentals**: `edge-introspect` tool (walk regions/blocks/
  ops, print SSA use-def, op statistics) + note.
- ⬜ **Module 02 — tablegen_ods**: generated-code walkthrough doc + note (how ODS
  becomes `.inc`, builders, verifiers, accessors).
- ⬜ **Module 04 — pass_manager**: `ShapeInferencePass` (InferTypeOpInterface),
  `StatisticsPass` (analysis), `IRPrinterPass`; pass registration + pipeline.
- ⬜ **Module 06 — rewrite_patterns**: canonicalization patterns, the rewrite engine,
  `ConversionPattern` basics.
- ⬜ **Module 05 — graph_optimization**: `ConvBnReluFusion`, constant folding, DCE,
  CSE, graph simplification; emit an optimization report.

## Phase 3 — Quantization

- ⬜ **Module 07 — quantization**: PTQ pipeline, calibration dataset loader,
  MinMax / KL-divergence / percentile calibrators, mixed-precision simulation;
  accuracy / latency / quantization reports. Reuses the upstream `quant` dialect +
  `!edge.qtensor`.

## Phase 4 — Lowering, memory, runtime

- ⬜ **Module 10 — lowering_pipeline**: `EdgeToLinalg` → bufferize → MemRef → Affine
  → LLVM; per-stage IR dumps + lowering visualization. `edge-compile` tool.
- ⬜ **Module 08 — bufferization**: tensor→memref, buffer allocation, in-place reuse.
- ⬜ **Module 09 — memory_planner**: tensor lifetime analysis, buffer reuse, graph-
  coloring allocation; peak-memory + memory-optimization reports.
- ⬜ **Module 11 — runtime_engine**: `ExecutionContext`, `GraphExecutor`,
  `OperatorScheduler`, async execution; `edge-run` tool.
- ⬜ **Module 12 — profiler**: latency / memory breakdown, timeline, execution trace.

## Phase 5 — Vendor analysis & end-to-end

- ⬜ **Module 13 — tpu_mlir_analysis**: architecture/dialect/passes/quant/backend/
  runtime, compared to this implementation.
- ⬜ **Module 14 — tensorrt_analysis**: builder, fusion, runtime, engine, memory mgmt.
- ⬜ **Module 15 — ascend_cann_analysis**: GE / TBE / ACL / runtime / graph engine.
- ⬜ **Module 16 — end_to_end_compiler**: full PyTorch→ONNX→MLIR→opt→quant→LLVM→run
  driver + fusion/compilation/latency/memory reports.
- ⬜ **Module 17 — robot_vla_deployment**: TensorRT deployment, robot runtime, VLA
  policy inference, action-latency optimization, multi-camera pipeline, scheduling.

## Cross-cutting (continuous)

- ⬜ Benchmarks: compilation time, optimization time, runtime latency, throughput,
  memory usage — emitted as Markdown under `benchmarks/`.
- ⬜ Tests: lit + FileCheck for IR transforms, GoogleTest for C++ runtime/planner.
- ⬜ Notes: one 12-section Chinese note per module (see template in `task.json`).
