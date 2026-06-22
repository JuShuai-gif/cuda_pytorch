//===- edge-opt.cpp - edge_ai_compiler_pro 的 opt 驱动 -----------*- C++ -*-===//
//
// 基于官方 MlirOptMain. 注册全部内建方言与 pass, 再额外注册 EdgeDialect,
// 因此 edge-opt 既能跑 Edge 方言 IR, 也能复用 canonicalize/cse/convert-* 等
// 内建 pass. 这与 mlir-opt 完全同源, 不自造任何驱动框架.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "Edge/EdgeDialect.h"
#include "Edge/Passes.h"

int main(int argc, char **argv) {
  // 注册全部内建 pass (canonicalize / cse / convert-* 等)
  mlir::registerAllPasses();
  // 注册本项目的 Edge passes (edge-shape-inference / edge-statistics / ...)
  mlir::edge::registerEdgePasses();
  // 注册端到端 lowering 流水线 (edge-lower-to-loops / edge-lower-to-llvm)
  mlir::edge::registerEdgePipelines();

  mlir::DialectRegistry registry;
  // 注册全部内建方言 (func/arith/tensor/linalg/memref/affine/scf/llvm ...)
  mlir::registerAllDialects(registry);
  // 注册本项目的 Edge 方言
  registry.insert<mlir::edge::EdgeDialect>();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "edge-ai-compiler optimizer driver\n", registry));
}
