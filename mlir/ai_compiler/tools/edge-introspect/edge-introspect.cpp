//===- edge-introspect.cpp - IR 遍历/打印/分析工具 ---------------*- C++ -*-===//
//
// Module 01 配套工具: 加载一个 .mlir 文件, 演示 MLIR 的核心数据结构遍历:
//   Operation / Region / Block / Value / Type / Attribute / use-def(SSA).
//
//   --tree   打印嵌套结构树 (Op 含 Region, Region 含 Block, Block 含 Op)
//   --uses   打印每个结果 Value 的使用计数 (use-def 链)
//   --stats  打印按算子名的统计
//
//===----------------------------------------------------------------------===//

#include "Edge/EdgeDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                         llvm::cl::desc("<input .mlir>"),
                                         llvm::cl::init("-"));
llvm::cl::opt<bool> showTree("edge-tree", llvm::cl::desc("打印嵌套结构树"),
                             llvm::cl::init(true));
llvm::cl::opt<bool> showUses("edge-uses",
                             llvm::cl::desc("打印结果 Value 的使用计数"),
                             llvm::cl::init(false));
llvm::cl::opt<bool> showStats("edge-stats", llvm::cl::desc("打印算子统计"),
                              llvm::cl::init(true));
} // namespace

// 递归打印一个 Op 的结构 (Operation -> Region -> Block -> Operation)
static void printOpTree(Operation *op, unsigned depth) {
  std::string indent(depth * 2, ' ');
  llvm::outs() << indent << op->getName().getStringRef() << "  [operands="
               << op->getNumOperands() << ", results=" << op->getNumResults()
               << ", regions=" << op->getNumRegions()
               << ", attrs=" << op->getAttrs().size() << "]\n";

  if (showUses) {
    for (auto [i, res] : llvm::enumerate(op->getResults())) {
      llvm::outs() << indent << "  result #" << i << " : " << res.getType()
                   << "  (uses=" << std::distance(res.use_begin(), res.use_end())
                   << ")\n";
    }
  }

  for (Region &region : op->getRegions()) {
    if (region.empty())
      continue;
    llvm::outs() << indent << "  Region (" << region.getBlocks().size()
                 << " blocks):\n";
    for (Block &block : region) {
      llvm::outs() << indent << "    Block (args=" << block.getNumArguments()
                   << ", ops=" << block.getOperations().size() << "):\n";
      for (Operation &inner : block)
        printOpTree(&inner, depth + 3);
    }
  }
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "edge-introspect: MLIR IR 遍历/分析工具\n");

  MLIRContext ctx;
  DialectRegistry registry;
  registry.insert<func::FuncDialect, arith::ArithDialect, tensor::TensorDialect,
                  edge::EdgeDialect>();
  ctx.appendDialectRegistry(registry);
  ctx.loadAllAvailableDialects();

  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(inputFilename, &ctx);
  if (!module) {
    llvm::errs() << "error: failed to parse " << inputFilename << "\n";
    return 1;
  }

  if (showTree) {
    llvm::outs() << "===== IR Structure Tree =====\n";
    printOpTree(module->getOperation(), 0);
    llvm::outs() << "\n";
  }

  if (showStats) {
    llvm::MapVector<llvm::StringRef, int64_t> counts;
    int64_t total = 0;
    module->walk([&](Operation *op) {
      counts[op->getName().getStringRef()]++;
      ++total;
    });
    llvm::outs() << "===== Op Statistics (" << total << " ops) =====\n";
    for (auto &kv : counts)
      llvm::outs() << "  " << kv.first << " : " << kv.second << "\n";
  }

  return 0;
}
