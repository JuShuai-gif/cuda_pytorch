//===- ShapeInference.cpp - 形状推断 Pass ------------------------*- C++ -*-===//
//
// 基于 ShapeInferenceOpInterface, 在 func 内做定点迭代: 只要某算子的操作数已是
// ranked, 就调用其 inferShapes() 把结果类型里的动态维细化为静态维, 直到不再变化.
//
//===----------------------------------------------------------------------===//

#include "Edge/EdgeOps.h"
#include "Edge/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::edge;

namespace mlir {
namespace edge {

#define GEN_PASS_DEF_SHAPEINFERENCE
#include "Edge/Passes.h.inc"

namespace {

// 判断一个张量类型是否"可用于推断"(已 ranked). 非张量操作数视为已就绪.
static bool operandsAreRanked(Operation *op) {
  return llvm::all_of(op->getOperands(), [](Value v) {
    if (auto t = llvm::dyn_cast<TensorType>(v.getType()))
      return t.hasRank();
    return true;
  });
}

struct ShapeInferencePass
    : impl::ShapeInferenceBase<ShapeInferencePass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();

    bool changed = true;
    int guard = 0;
    // 定点迭代; guard 防止意外死循环
    while (changed && guard++ < 1000) {
      changed = false;
      func.walk([&](ShapeInferenceOpInterface shapeOp) {
        Operation *op = shapeOp.getOperation();
        if (!operandsAreRanked(op))
          return;
        llvm::SmallVector<Type> before(op->getResultTypes().begin(),
                                       op->getResultTypes().end());
        shapeOp.inferShapes();
        for (auto [i, res] : llvm::enumerate(op->getResults())) {
          if (res.getType() != before[i])
            changed = true;
        }
      });
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}

} // namespace edge
} // namespace mlir
