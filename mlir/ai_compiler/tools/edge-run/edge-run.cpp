//===- edge-run.cpp - Edge 图执行 + Profiling CLI ----------------*- C++ -*-===//
//
// 加载 Edge IR, 用 0/1 填充输入, 解释执行 (constant/relu/matmul), 打印 Profiling
// 报告与输出张量校验和. Module 11(Runtime) + 12(Profiler) 的命令行入口.
//
//===----------------------------------------------------------------------===//

#include "Edge/EdgeDialect.h"
#include "Edge/Runtime.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::edge::runtime;

namespace {
llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                         llvm::cl::desc("<input .mlir>"),
                                         llvm::cl::init("-"));
llvm::cl::opt<float> fillValue("edge-fill",
                               llvm::cl::desc("输入张量填充值"),
                               llvm::cl::init(1.0f));
} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "edge-run: Edge 图执行 + 性能分析\n");

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

  func::FuncOp func;
  module->walk([&](func::FuncOp f) {
    if (!func)
      func = f;
  });
  if (!func) {
    llvm::errs() << "error: no func.func found\n";
    return 1;
  }

  ExecutionContext ectx;
  // 用 fillValue 填充函数入参
  if (!func.getBody().empty()) {
    for (BlockArgument arg : func.getBody().front().getArguments()) {
      auto rt = llvm::dyn_cast<RankedTensorType>(arg.getType());
      if (!rt || !rt.hasStaticShape()) {
        llvm::errs() << "error: input must be static ranked tensor\n";
        return 1;
      }
      Tensor t;
      for (int64_t d : rt.getShape())
        t.shape.push_back(d);
      t.data.assign(t.numElements(), fillValue);
      ectx.set(arg, std::move(t));
    }
  }

  Profiler prof;
  GraphExecutor exec(prof);
  if (failed(exec.run(func, ectx))) {
    llvm::errs() << "error: execution failed\n";
    return 1;
  }

  prof.report(llvm::outs());

  // 输出张量校验和
  func.walk([&](func::ReturnOp ret) {
    llvm::outs() << "## Outputs\n";
    for (Value v : ret.getOperands()) {
      if (!ectx.has(v))
        continue;
      Tensor &t = ectx.get(v);
      double sum = 0;
      for (float x : t.data)
        sum += x;
      llvm::outs() << "- elements=" << t.numElements()
                   << llvm::format(", checksum=%.4f", sum) << "\n";
    }
  });

  return 0;
}
