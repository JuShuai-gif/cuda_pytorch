//===- Passes.h - Edge 方言 Pass 声明与注册 ----------------------*- C++ -*-===//
//
// 暴露 createXxxPass() 工厂与 registerEdgePasses() 注册函数.
//
//===----------------------------------------------------------------------===//

#ifndef EDGE_PASSES_H
#define EDGE_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace func {
class FuncOp;
} // namespace func

namespace edge {

// Pass 工厂 (在各 .cpp 中实现)
std::unique_ptr<mlir::Pass> createShapeInferencePass();
std::unique_ptr<mlir::Pass> createStatisticsPass();
std::unique_ptr<mlir::Pass> createIRPrinterPass();
std::unique_ptr<mlir::Pass> createFuseConvBnReluPass();
std::unique_ptr<mlir::Pass> createLowerEdgeToLinalgPass();

// 注册端到端 lowering 流水线 (edge-lower-to-loops / edge-lower-to-llvm)
void registerEdgePipelines();

// 生成的 pass 声明 (option 结构体等)
#define GEN_PASS_DECL
#include "Edge/Passes.h.inc"

// 生成的注册函数: registerEdgePasses() 以及各 registerXxx()
#define GEN_PASS_REGISTRATION
#include "Edge/Passes.h.inc"

} // namespace edge
} // namespace mlir

#endif // EDGE_PASSES_H
