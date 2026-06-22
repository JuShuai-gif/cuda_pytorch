# edge_ai_compiler_pro

A production-oriented, MLIR-based AI compiler stack targeting **edge / robotics / VLA**
deployment. Built entirely on the **official MLIR / LLVM API** (ODS, TableGen,
RewritePattern, ConversionPattern, PassManager, MlirOptMain) — no custom IR or
compiler framework is reinvented.

> This repository is the next evolution of the `learn_mlir` demo: it reuses the
> same local MLIR installation and CMake conventions, but grows into a real
> graph-level compiler with a custom dialect, optimization passes, a lowering
> pipeline, quantization, a runtime, and a profiler.

> 🟢 **零基础从这里开始**：如果你没学过编译原理，请先读
> **[`docs/TUTORIAL.md`](docs/TUTORIAL.md)** —— 一份用生活化比喻、从"编译器是什么"讲起、
> 每个概念都配可运行命令的中文入门教程。

---

## Status

| Layer | State |
|-------|-------|
| Build system (CMake + Ninja, C++20, reuses local MLIR install) | **Done, verified** |
| **All 17 modules (01–17)** implemented | **Done** — see [`task.json`](task.json) (17/17) |
| `EdgeDialect` (M03): Conv2D / BatchNorm / Relu / Matmul / Attention / ConvBnRelu / Constant + custom type `!edge.qtensor` + attr `#edge.quant_params` + `Layout` enum | **Done, roundtrip-verified** |
| Passes (M04/05/06): `edge-shape-inference`, `edge-fuse-conv-bn-relu` (BN folding), `edge-statistics`, `edge-ir-printer`; canonicalize/DCE/CSE | **Done, tested** |
| Quantization (M07): MinMax / KL / Percentile calibration + INT8 sim (full/body SQNR) + mixed precision | **Done, tested** |
| Lowering (M08/10): `edge-lower-to-linalg`, `edge-lower-to-loops`, `edge-lower-to-llvm` (Edge → Linalg → bufferize → LLVM) | **Done, tested** |
| Memory planner (M09): lifetime + graph-coloring reuse (`edge-memplan`) | **Done, tested** |
| Runtime + Profiler (M11/12): `edge-run` (ExecutionContext/GraphExecutor/Scheduler + latency breakdown) | **Done, tested** |
| End-to-end driver (M16): `scripts/edge_compile.py` → fusion/compilation/latency/memory reports | **Done, verified** |
| Vendor analysis notes (M13/14/15) + Robot/VLA (M17) | **Done** |
| Tools: `edge-opt`, `edge-introspect`, `edge-memplan`, `edge-run`, `edge-quantize` | **Done** |
| lit + FileCheck regression harness | **Done — 10/10 passing** |

Verified roundtrip:

```mlir
%0 = edge.conv2d %input, %weight {strides = array<i64: 2, 2>, ...}
     : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32>
%1 = edge.batch_norm %0, %scale, %bias, %mean, %var : ...
%2 = edge.relu %1 : tensor<1x64x112x112xf32>
// custom type / attr also roundtrip:
//   !edge.qtensor<tensor<1x3x4x4xi8>, 7.812500e-03 : f64, 0>
//   #edge.quant_params<2.500000e-02 : f64, 128>
```

---

## The Compiler Pipeline

```
PyTorch → ONNX → ONNX Importer → MLIR (ONNX/TOSA) → EdgeDialect
  → Optimization Passes → Linalg → Bufferization → MemRef
  → Affine / SCF → LLVM Dialect → LLVM IR → Backend Runtime → Execution → Profiling
```

`EdgeDialect` is the **graph-level IR**, analogous to TPU-MLIR's `top`/`tpu`
dialects or torch-mlir's `torch` dialect: high-level fusion, constant folding,
and quantization happen here before progressive lowering to Linalg and LLVM.

---

## Repository Layout

```
ai_compiler/
├── include/Edge/      # TableGen (.td) + public headers for the Edge dialect
├── src/               # C++ implementation
│   ├── EdgeDialect/    # dialect, ops, types, attributes  (Module 03 — done)
│   ├── Transforms/     # passes: shape inference, fusion, DCE/CSE  (Modules 04–06)
│   ├── Conversion/     # Edge → Linalg → MemRef → LLVM lowering  (Module 10)
│   ├── Quantization/   # PTQ / calibration / mixed precision  (Module 07)
│   ├── MemoryPlanner/  # lifetime analysis + graph-coloring allocation  (Module 09)
│   ├── Runtime/        # execution context, graph executor, scheduler  (Module 11)
│   └── Profiler/       # latency / memory / timeline  (Module 12)
├── tools/             # edge-opt (done), edge-introspect, edge-quantize, edge-compile ...
├── tests/             # lit + FileCheck + GoogleTest
├── notes/             # 中文工程笔记 (12-section template per module)
├── docs/              # generated dialect docs + cross-vendor comparisons
├── benchmarks/        # compile time / latency / throughput / memory reports
├── examples/          # end-to-end examples (incl. VLA policy inference)
├── task.json          # machine-readable master plan (17 modules)
├── ARCHITECTURE.md  ROADMAP.md  INTERVIEW_GUIDE.md
```

---

## Build

Prerequisites: a local LLVM/MLIR install (this repo targets the one at
`/home/ghr/code/llvm-project/install`, **LLVM 23.0.0git, Release+Asserts, no-RTTI,
no-ASan, static libs, clang-18**).

```bash
cmake -G Ninja -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=/home/ghr/code/llvm-project/install/lib/cmake/mlir
ninja -C build edge-opt        # build the optimizer driver
ninja -C build check-edge      # run the lit + FileCheck regression suite
```

Notes:
- The build reuses the official MLIR CMake macros (`add_mlir_dialect_library`,
  `mlir_tablegen`, `MlirOptMain`) and `HandleLLVMOptions` (propagates `-fno-rtti`
  to match the install ABI).
- AddressSanitizer is an **opt-in** option (`-DEDGE_ENABLE_ASAN=ON`); it is OFF by
  default because the local MLIR libraries were not built with ASan.
- `lit`/`FileCheck` are taken from the LLVM **build** tree
  (`/home/ghr/code/llvm-project/build/bin`) since the install omits test tools.

---

## Why this project exists

Real inference compilers (TensorRT, TVM, TPU-MLIR, Ascend CANN) all share the same
skeleton: a graph-level IR, a fusion/quantization optimizer, a lowering pipeline to
a low-level kernel IR, a memory planner, a runtime, and a profiler. This repo builds
that skeleton on MLIR so each concept maps 1:1 onto what those vendors ship — see the
per-module notes in [`notes/`](notes/), which connect every pass to its TensorRT /
TVM / TPU-MLIR / Ascend counterpart and to robotics/VLA latency.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the design and [`INTERVIEW_GUIDE.md`](INTERVIEW_GUIDE.md)
for interview preparation mapped to this codebase.
