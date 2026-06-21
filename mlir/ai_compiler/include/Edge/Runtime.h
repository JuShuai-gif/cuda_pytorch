//===- Runtime.h - Edge 运行时与 Profiler -----------------------*- C++ -*-===//
//
// 一个最小但真实的图执行运行时: 在 float buffer 上解释执行 Edge 子集
// (constant / relu / matmul), 含执行上下文、算子调度器与 Profiler.
//
//===----------------------------------------------------------------------===//

#ifndef EDGE_RUNTIME_H
#define EDGE_RUNTIME_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <vector>

namespace mlir {
namespace edge {
namespace runtime {

// 简单的稠密 float 张量 (行主序)
struct Tensor {
  std::vector<int64_t> shape;
  std::vector<float> data;
  int64_t numElements() const {
    int64_t n = 1;
    for (int64_t d : shape)
      n *= d;
    return n;
  }
};

// Profiler: 记录每个算子的执行延迟与输出字节, 输出 Markdown breakdown.
class Profiler {
public:
  struct Record {
    std::string name;
    double ms;
    int64_t bytesOut;
  };
  void add(llvm::StringRef name, double ms, int64_t bytesOut) {
    records.push_back({name.str(), ms, bytesOut});
  }
  double totalMs() const {
    double t = 0;
    for (auto &r : records)
      t += r.ms;
    return t;
  }
  void report(llvm::raw_ostream &os) const;

private:
  std::vector<Record> records;
};

// 执行上下文: SSA 值 -> Tensor.
class ExecutionContext {
public:
  void set(Value v, Tensor t) { store[v] = std::move(t); }
  bool has(Value v) const { return store.count(v); }
  Tensor &get(Value v) { return store[v]; }

private:
  llvm::DenseMap<Value, Tensor> store;
};

// 算子调度器: 返回执行顺序 (当前为程序顺序的拓扑序; 预留异步/并行扩展点).
class OperatorScheduler {
public:
  std::vector<Operation *> schedule(func::FuncOp func);
};

// 图执行器: 解释执行 (constant/relu/matmul). 输入需预置于 ctx.
class GraphExecutor {
public:
  explicit GraphExecutor(Profiler &p) : prof(p) {}
  LogicalResult run(func::FuncOp func, ExecutionContext &ctx);

private:
  Profiler &prof;
};

} // namespace runtime
} // namespace edge
} // namespace mlir

#endif // EDGE_RUNTIME_H
