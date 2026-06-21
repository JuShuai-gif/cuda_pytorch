//===- IRPrinter.cpp - 带标签打印 IR 的调试 Pass -----------------*- C++ -*-===//
//
// 在 pass 流水线任意位置插入, 打印当前 IR 并带一个标签横幅. 不修改 IR.
// 比 --mlir-print-ir-after-all 更可控 (只在你放置它的地方打印).
//
//===----------------------------------------------------------------------===//

#include "Edge/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::edge;

namespace mlir {
namespace edge {

#define GEN_PASS_DEF_IRPRINTER
#include "Edge/Passes.h.inc"

namespace {

struct IRPrinterPass : impl::IRPrinterBase<IRPrinterPass> {
  using impl::IRPrinterBase<IRPrinterPass>::IRPrinterBase;

  void runOnOperation() override {
    llvm::outs() << "// ===== [" << label << "] =====\n";
    getOperation()->print(llvm::outs());
    llvm::outs() << "\n";
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createIRPrinterPass() {
  return std::make_unique<IRPrinterPass>();
}

} // namespace edge
} // namespace mlir
