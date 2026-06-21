//===- Runtime.cpp - Edge 运行时实现 ----------------------------*- C++ -*-===//

#include "Edge/Runtime.h"
#include "Edge/EdgeOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Format.h"

#include <algorithm>
#include <chrono>

using namespace mlir;
using namespace mlir::edge;
using namespace mlir::edge::runtime;

//===----------------------------------------------------------------------===//
// Profiler
//===----------------------------------------------------------------------===//
void Profiler::report(llvm::raw_ostream &os) const {
  double total = totalMs();
  os << "# Edge Runtime Profiling Report\n\n";
  os << "- Ops executed: " << records.size() << "\n";
  os << llvm::format("- Total latency: %.4f ms\n\n", total);
  os << "| op | latency(ms) | %% | out bytes |\n";
  os << "|----|-------------|----|-----------|\n";
  for (auto &r : records) {
    double pct = total > 0 ? (r.ms * 100.0 / total) : 0.0;
    os << "| " << r.name << " | " << llvm::format("%.4f", r.ms) << " | "
       << llvm::format("%.1f", pct) << " | " << r.bytesOut << " |\n";
  }
  os << "\n";
}

//===----------------------------------------------------------------------===//
// OperatorScheduler: 程序顺序的拓扑序 (def-before-use 已由 SSA 保证)
//===----------------------------------------------------------------------===//
std::vector<Operation *> OperatorScheduler::schedule(func::FuncOp func) {
  std::vector<Operation *> order;
  func.walk([&](Operation *op) {
    // 只调度函数体内的计算算子, 跳过 func/module 容器与 return
    if (llvm::isa<func::FuncOp>(op) || llvm::isa<func::ReturnOp>(op))
      return;
    order.push_back(op);
  });
  return order;
}

//===----------------------------------------------------------------------===//
// 张量辅助
//===----------------------------------------------------------------------===//
static Tensor makeTensor(RankedTensorType ty) {
  Tensor t;
  for (int64_t d : ty.getShape())
    t.shape.push_back(d);
  t.data.assign(t.numElements(), 0.0f);
  return t;
}

//===----------------------------------------------------------------------===//
// 各算子的 kernel
//===----------------------------------------------------------------------===//
static LogicalResult runConstant(ConstantOp op, ExecutionContext &ctx) {
  auto ty = llvm::dyn_cast<RankedTensorType>(op.getType());
  auto dense = llvm::dyn_cast<DenseElementsAttr>(op.getValue());
  if (!ty || !dense)
    return failure();
  Tensor t = makeTensor(ty);
  int64_t i = 0;
  for (float f : dense.getValues<float>()) {
    if (i < (int64_t)t.data.size())
      t.data[i] = f;
    ++i;
  }
  // splat (dense<1.0>) 只有一个值, 广播填充
  if (i == 1)
    std::fill(t.data.begin(), t.data.end(), t.data[0]);
  ctx.set(op.getResult(), std::move(t));
  return success();
}

static LogicalResult runRelu(ReluOp op, ExecutionContext &ctx) {
  if (!ctx.has(op.getInput()))
    return failure();
  Tensor &in = ctx.get(op.getInput());
  Tensor out = in;
  for (float &x : out.data)
    x = std::max(0.0f, x);
  ctx.set(op.getResult(), std::move(out));
  return success();
}

static LogicalResult runMatmul(MatmulOp op, ExecutionContext &ctx) {
  if (!ctx.has(op.getLhs()) || !ctx.has(op.getRhs()))
    return failure();
  Tensor &a = ctx.get(op.getLhs());
  Tensor &b = ctx.get(op.getRhs());
  if (a.shape.size() != 2 || b.shape.size() != 2 || a.shape[1] != b.shape[0])
    return failure();
  int64_t M = a.shape[0], K = a.shape[1], N = b.shape[1];
  Tensor out;
  out.shape = {M, N};
  out.data.assign(M * N, 0.0f);
  for (int64_t m = 0; m < M; ++m)
    for (int64_t k = 0; k < K; ++k) {
      float av = a.data[m * K + k];
      for (int64_t n = 0; n < N; ++n)
        out.data[m * N + n] += av * b.data[k * N + n];
    }
  ctx.set(op.getResult(), std::move(out));
  return success();
}

//===----------------------------------------------------------------------===//
// GraphExecutor
//===----------------------------------------------------------------------===//
LogicalResult GraphExecutor::run(func::FuncOp func, ExecutionContext &ctx) {
  OperatorScheduler scheduler;
  for (Operation *op : scheduler.schedule(func)) {
    auto t0 = std::chrono::high_resolution_clock::now();
    LogicalResult res = success();

    if (auto c = llvm::dyn_cast<ConstantOp>(op))
      res = runConstant(c, ctx);
    else if (auto r = llvm::dyn_cast<ReluOp>(op))
      res = runRelu(r, ctx);
    else if (auto mm = llvm::dyn_cast<MatmulOp>(op))
      res = runMatmul(mm, ctx);
    else {
      llvm::errs() << "warning: unsupported op '" << op->getName()
                   << "', skipped\n";
      continue;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    int64_t bytesOut = 0;
    if (op->getNumResults() > 0 && ctx.has(op->getResult(0)))
      bytesOut = ctx.get(op->getResult(0)).numElements() * 4;
    prof.add(op->getName().getStringRef(), ms, bytesOut);

    if (failed(res)) {
      llvm::errs() << "error: failed to execute '" << op->getName() << "'\n";
      return failure();
    }
  }
  return success();
}
