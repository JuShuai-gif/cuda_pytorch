# Architecture

This document describes the design of `edge_ai_compiler_pro`: the IR layering, the
dialect design, the lowering strategy, and the engineering constraints imposed by
the target MLIR installation.

## 1. Design principles

1. **Reuse, don't reinvent.** Everything is built on the official MLIR/LLVM API —
   ODS + TableGen for op definitions, `RewritePattern`/`ConversionPattern` for
   transforms, `PassManager` for orchestration, `MlirOptMain` for the driver. No
   bespoke IR, no bespoke pass framework.
2. **Graph-level IR first.** A custom `EdgeDialect` is the high-level entry point
   where the value-adding optimizations (fusion, constant folding, quantization)
   live, mirroring how TPU-MLIR (`top`/`tpu`) and torch-mlir (`torch`) structure
   their stacks.
3. **Progressive lowering.** Each step lowers exactly one level of abstraction and
   is independently testable with FileCheck. This is the MLIR philosophy and the
   reason production compilers can be debugged at all.
4. **Production constraints baked in.** Built against a real `-fno-rtti`,
   `-fno-exceptions`, Release+Asserts MLIR; the build mirrors those flags via
   `HandleLLVMOptions` to stay ABI-compatible with the prebuilt static libraries.

## 2. IR layering

```
                         ┌─────────────────────────────────────────────┐
  Frontend               │  PyTorch → ONNX → (ONNX/TOSA importer)        │
                         └───────────────────────┬─────────────────────┘
                                                 │
  Graph-level IR  ┌──────────────────────────────▼─────────────────────┐
  (this project)  │  EdgeDialect                                        │
                  │   ops:  conv2d, batch_norm, relu, matmul,           │
                  │         attention, conv_bn_relu, constant           │
                  │   type: !edge.qtensor   attr: #edge.quant_params     │
                  │   passes: shape-inference, fusion, const-fold,      │
                  │           DCE, CSE, quantization                    │
                  └──────────────────────────────┬─────────────────────┘
                                                 │ EdgeToLinalg (ConversionPattern)
  Compute IR      ┌──────────────────────────────▼─────────────────────┐
                  │  Linalg on tensors → Bufferization → MemRef         │
                  └──────────────────────────────┬─────────────────────┘
                                                 │ lower to loops
  Loop/Target IR  ┌──────────────────────────────▼─────────────────────┐
                  │  Affine / SCF → LLVM Dialect → LLVM IR              │
                  └──────────────────────────────┬─────────────────────┘
                                                 │
  Runtime         ┌──────────────────────────────▼─────────────────────┐
                  │  ExecutionContext + GraphExecutor + Profiler        │
                  └─────────────────────────────────────────────────────┘
```

### Why a custom dialect instead of starting at Linalg/TOSA?

Fusion and quantization decisions need *semantic* operations (`conv`, `batch_norm`,
`attention`) — not their already-decomposed Linalg form. Once a `conv` is expanded
into `linalg.generic` + loops, recovering "this is a convolution that can fuse with
the next BN" is expensive pattern-matching. Keeping a typed, named, high-level op set
(the `edge` dialect) makes these passes simple, fast, and verifiable — exactly the
reason TensorRT works on a graph of layers and TPU-MLIR keeps a `top` dialect.

## 3. Dialect design (`EdgeDialect`)

- **Operands/results use builtin `RankedTensorType`** so lowering to Linalg/TOSA is
  frictionless. Quantization is expressed with a dedicated `!edge.qtensor` type and
  `#edge.quant_params` attribute rather than overloading the builtin tensor.
- **`edge.constant`** is `ConstantLike` + foldable; the dialect implements
  `materializeConstant`, enabling constant folding and canonicalization to create
  fresh constants (Module 05).
- **`edge.conv_bn_relu`** is the fusion target produced by `ConvBnReluFusion`
  (Module 05). Defining it as a first-class op (vs. an attribute flag) keeps the
  cost model and lowering explicit.
- **`Layout` enum (NCHW/NHWC)** models the format-selection problem that TensorRT
  solves automatically (NCHW ↔ NHWC ↔ NC4HW4) to feed Tensor Cores efficiently.

ODS generation is wired manually in `include/Edge/CMakeLists.txt` (rather than the
all-in-one `add_mlir_dialect`) so that op, dialect, type, **attribute**, and **enum**
generators are all invoked — the same control large projects (torch-mlir, IREE) use.

## 4. Lowering strategy (Module 10)

`EdgeToLinalg` uses the dialect-conversion framework:
- `edge.matmul`  → `linalg.matmul`
- `edge.conv2d`  → `linalg.conv_2d_nchw_fchw`
- `edge.relu`    → `linalg.generic` (max with 0) / `tosa` then `linalg`
- `edge.attention` → decomposed (matmul + softmax + matmul) or kept fused for a
  FlashAttention-style backend kernel.

After Linalg, standard MLIR pipelines handle bufferization → MemRef → loops → LLVM.
Each stage dumps IR (`--mlir-print-ir-after-all`) for the lowering visualization
deliverable.

## 5. Build/ABI constraints (verified)

| Constraint | Value | Why it matters |
|------------|-------|----------------|
| LLVM/MLIR | 23.0.0git, Release+Asserts | newest ODS/Conversion API |
| RTTI | OFF | code must use `llvm::isa/cast/dyn_cast`, never `dynamic_cast`; build mirrors `-fno-rtti` for ABI match |
| Exceptions | OFF | no `try/catch` in core |
| ASan | not in libs | `EDGE_ENABLE_ASAN` default OFF to avoid link mismatch |
| Libraries | static `.a` | link via `MLIR_DIALECT_LIBS`/`MLIR_CONVERSION_LIBS` globals |
| Test tools | only in LLVM build tree | lit/FileCheck sourced from `…/build/bin` |

## 6. Module dependency graph

```
03 EdgeDialect ──┬─> 04 PassManager ─┬─> 05 GraphOpt ──┐
                 │                    └─> 06 Rewrite ───┤
                 ├─> 07 Quantization ─────────────────┤
                 └─> 10 Lowering ─> 08 Bufferization ─> 09 MemoryPlanner ─> 11 Runtime ─> 12 Profiler
                                                                                          │
01 Fundamentals, 02 ODS  (foundational notes/tooling)                                     ▼
13 TPU-MLIR, 14 TensorRT, 15 Ascend (analysis)  ───────────────────────> 16 End-to-end ─> 17 Robot/VLA
```

See [`task.json`](task.json) for the authoritative per-module deliverables and status.
