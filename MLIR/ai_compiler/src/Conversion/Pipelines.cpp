//===- Pipelines.cpp - 端到端 lowering 流水线注册 ----------------*- C++ -*-===//
//
// 注册两条流水线 (用 parsePassPipeline 拼装已注册的 pass, 复用 MLIR 标准 lowering):
//   edge-lower-to-loops : Edge -> Linalg -> bufferize -> scf.for + memref
//   edge-lower-to-llvm  : Edge -> ... -> LLVM 方言
//
// 这些配方已在命令行验证可零错误地把 Edge IR 一路降到 LLVM 方言.
//
//===----------------------------------------------------------------------===//

#include "Edge/Passes.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;

namespace mlir {
namespace edge {

// Edge -> Linalg -> bufferize -> 循环(scf)+memref
static const char *kLowerToLoops =
    "func.func(edge-shape-inference),"
    "func.func(edge-lower-to-linalg),"
    "one-shot-bufferize{bufferize-function-boundaries=true},"
    "func.func(convert-linalg-to-loops)";

// Edge -> ... -> LLVM 方言 (完整后端路径)
static const char *kLowerToLLVM =
    "func.func(edge-shape-inference),"
    "func.func(edge-lower-to-linalg),"
    "one-shot-bufferize{bufferize-function-boundaries=true},"
    "buffer-results-to-out-params,"
    "func.func(convert-linalg-to-loops),"
    "convert-scf-to-cf,"
    "expand-strided-metadata,"
    "finalize-memref-to-llvm,"
    "convert-cf-to-llvm,"
    "convert-arith-to-llvm,"
    "convert-func-to-llvm,"
    "reconcile-unrealized-casts";

void registerEdgePipelines() {
  PassPipelineRegistration<>(
      "edge-lower-to-loops",
      "Edge -> Linalg -> bufferize -> scf.for + memref",
      [](OpPassManager &pm) {
        if (failed(parsePassPipeline(kLowerToLoops, pm)))
          llvm::report_fatal_error("invalid edge-lower-to-loops pipeline");
      });

  PassPipelineRegistration<>(
      "edge-lower-to-llvm", "Edge -> ... -> LLVM dialect",
      [](OpPassManager &pm) {
        if (failed(parsePassPipeline(kLowerToLLVM, pm)))
          llvm::report_fatal_error("invalid edge-lower-to-llvm pipeline");
      });
}

} // namespace edge
} // namespace mlir
