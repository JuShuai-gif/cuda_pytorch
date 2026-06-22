//===- Statistics.cpp - 计算图统计 Pass --------------------------*- C++ -*-===//
//
// 遍历计算图, 统计各算子数量, 并对 conv/matmul 估算乘加次数 (MAC),
// 输出 Markdown 风格报告. 这是部署前评估模型计算量的基础分析工具.
//
//===----------------------------------------------------------------------===//

#include "Edge/EdgeOps.h"
#include "Edge/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::edge;

namespace mlir {
namespace edge {

#define GEN_PASS_DEF_STATISTICS
#include "Edge/Passes.h.inc"

namespace {

// 取张量静态元素个数; 含动态维则返回 0 (表示未知).
static int64_t numStaticElements(Type t) {
  auto rt = llvm::dyn_cast<RankedTensorType>(t);
  if (!rt || !rt.hasStaticShape())
    return 0;
  int64_t n = 1;
  for (int64_t d : rt.getShape())
    n *= d;
  return n;
}

// 矩阵乘 MAC = (batch...) * M * N * K
static int64_t matmulMACs(MatmulOp op) {
  auto lhs = llvm::dyn_cast<RankedTensorType>(op.getLhs().getType());
  auto out = llvm::dyn_cast<RankedTensorType>(op.getOutput().getType());
  if (!lhs || !out || !lhs.hasStaticShape() || !out.hasStaticShape())
    return 0;
  int64_t k = lhs.getShape().back();
  return numStaticElements(out) * k;
}

// 卷积 MAC = N*Cout*Hout*Wout * Cin*kH*kW
template <typename ConvOpT>
static int64_t convMACs(ConvOpT op) {
  auto out = llvm::dyn_cast<RankedTensorType>(op.getOutput().getType());
  auto w = llvm::dyn_cast<RankedTensorType>(op.getWeight().getType());
  if (!out || !w || !out.hasStaticShape() || !w.hasStaticShape())
    return 0;
  int64_t perOutElem = w.getDimSize(1) * w.getDimSize(2) * w.getDimSize(3);
  return numStaticElements(out) * perOutElem;
}

struct StatisticsPass : impl::StatisticsBase<StatisticsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    llvm::MapVector<llvm::StringRef, int64_t> opCounts;
    int64_t totalOps = 0;
    int64_t totalMACs = 0;

    module.walk([&](Operation *op) {
      // 跳过容器型 Op (module/func 本身)
      if (llvm::isa<ModuleOp>(op))
        return;
      llvm::StringRef name = op->getName().getStringRef();
      opCounts[name]++;
      totalOps++;

      if (auto mm = llvm::dyn_cast<MatmulOp>(op))
        totalMACs += matmulMACs(mm);
      else if (auto cv = llvm::dyn_cast<Conv2DOp>(op))
        totalMACs += convMACs(cv);
      else if (auto cbr = llvm::dyn_cast<ConvBnReluOp>(op))
        totalMACs += convMACs(cbr);
    });

    // 输出 Markdown 报告
    llvm::outs() << "# Edge Graph Statistics Report\n\n";
    llvm::outs() << "- Total operations: " << totalOps << "\n";
    llvm::outs() << "- Estimated MACs: " << totalMACs << "  (~"
                 << (totalMACs / 1'000'000) << " MMACs)\n\n";
    llvm::outs() << "| Operation | Count |\n|-----------|-------|\n";
    for (auto &kv : opCounts)
      llvm::outs() << "| " << kv.first << " | " << kv.second << " |\n";
    llvm::outs() << "\n";
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createStatisticsPass() {
  return std::make_unique<StatisticsPass>();
}

} // namespace edge
} // namespace mlir
