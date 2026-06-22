//===- edge-memplan.cpp - 张量生命周期分析 + 内存复用规划 --------*- C++ -*-===//
//
// Module 09 配套工具: 在 Edge IR (张量 SSA 值) 上做内存规划:
//   1) 生命周期分析: 每个张量值的 [birth, death] (按程序顺序索引)
//   2) 大小计算: numElements * 元素字节宽
//   3) 贪心 by-size first-fit 分配 (本质是区间图着色): 生命周期不重叠的张量共享地址
//   4) 报告: 朴素峰值(无复用) vs 规划峰值(有复用) 与节省比例
//
// 这与 TFLite GreedyMemoryPlanner / XLA buffer assignment 同思路.
//
//===----------------------------------------------------------------------===//

#include "Edge/EdgeDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

using namespace mlir;

namespace {
llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                         llvm::cl::desc("<input .mlir>"),
                                         llvm::cl::init("-"));
llvm::cl::opt<int64_t> alignment("edge-align",
                                 llvm::cl::desc("内存对齐字节数"),
                                 llvm::cl::init(64));

struct TensorLive {
  unsigned id;
  std::string shape;
  int64_t bytes;
  int64_t birth;
  int64_t death;
  int64_t offset = -1; // 分配后的偏移
};

int64_t roundUp(int64_t v, int64_t a) { return a <= 1 ? v : ((v + a - 1) / a) * a; }

int64_t tensorBytes(RankedTensorType t) {
  if (!t.hasStaticShape())
    return 0;
  int64_t n = 1;
  for (int64_t d : t.getShape())
    n *= d;
  Type elt = t.getElementType();
  int64_t bits = elt.isIntOrFloat() ? elt.getIntOrFloatBitWidth() : 32;
  return n * ((bits + 7) / 8);
}
} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "edge-memplan: 张量生命周期与内存复用规划\n");

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

  // 1) 给函数体内每个 Op 编程序号
  llvm::DenseMap<Operation *, int64_t> opIndex;
  int64_t idx = 0;
  func.walk([&](Operation *op) { opIndex[op] = idx++; });
  int64_t lastIndex = idx > 0 ? idx - 1 : 0;

  // 2) 收集张量值并计算生命周期
  llvm::SmallVector<TensorLive> lives;
  auto recordValue = [&](Value v, int64_t birth) {
    auto rt = llvm::dyn_cast<RankedTensorType>(v.getType());
    if (!rt || !rt.hasStaticShape())
      return;
    int64_t death = birth;
    for (Operation *user : v.getUsers()) {
      auto it = opIndex.find(user);
      if (it != opIndex.end())
        death = std::max(death, it->second);
      // 被 return 使用 -> 活到结尾
      if (llvm::isa<func::ReturnOp>(user))
        death = lastIndex;
    }
    std::string shapeStr;
    llvm::raw_string_ostream os(shapeStr);
    os << rt;
    os.flush();
    lives.push_back({(unsigned)lives.size(), shapeStr, tensorBytes(rt), birth,
                     death, -1});
  };

  // 函数入参 (block args) birth=0
  if (!func.getBody().empty())
    for (BlockArgument arg : func.getBody().front().getArguments())
      recordValue(arg, 0);
  // 每个算子结果 birth = 算子序号
  func.walk([&](Operation *op) {
    for (Value res : op->getResults())
      recordValue(res, opIndex[op]);
  });

  // 3) 贪心 by-size first-fit 分配 (区间图着色)
  llvm::SmallVector<unsigned> order;
  for (auto &t : lives)
    order.push_back(t.id);
  std::sort(order.begin(), order.end(), [&](unsigned a, unsigned b) {
    return lives[a].bytes > lives[b].bytes;
  });

  auto overlap = [](const TensorLive &a, const TensorLive &b) {
    return !(a.death < b.birth || b.death < a.birth);
  };

  int64_t plannedPeak = 0;
  for (unsigned i : order) {
    TensorLive &t = lives[i];
    // 收集与 t 生命周期重叠且已分配的区间
    llvm::SmallVector<std::pair<int64_t, int64_t>> occupied; // [off, off+bytes)
    for (auto &o : lives)
      if (o.offset >= 0 && overlap(t, o))
        occupied.push_back({o.offset, o.offset + o.bytes});
    std::sort(occupied.begin(), occupied.end());
    // first-fit: 找最低的、能放下 t.bytes 的 offset
    int64_t candidate = 0;
    for (auto &iv : occupied) {
      if (candidate + t.bytes <= iv.first)
        break; // 在 iv 之前的空隙能放下
      candidate = std::max(candidate, roundUp(iv.second, alignment));
    }
    t.offset = candidate;
    plannedPeak = std::max(plannedPeak, t.offset + t.bytes);
  }

  // 4) 报告
  int64_t naiveTotal = 0;
  for (auto &t : lives)
    naiveTotal += t.bytes;

  llvm::outs() << "# Edge Memory Planning Report\n\n";
  llvm::outs() << "- Tensors: " << lives.size() << "\n";
  llvm::outs() << "- Naive peak (no reuse): " << naiveTotal << " bytes\n";
  llvm::outs() << "- Planned peak (reuse) : " << plannedPeak << " bytes\n";
  if (naiveTotal > 0)
    llvm::outs() << "- Saving: "
                 << (100 - (plannedPeak * 100 / naiveTotal)) << "%\n";
  llvm::outs() << "\n| id | shape | bytes | live[birth,death] | offset |\n";
  llvm::outs() << "|----|-------|-------|-------------------|--------|\n";
  for (auto &t : lives)
    llvm::outs() << "| " << t.id << " | " << t.shape << " | " << t.bytes
                 << " | [" << t.birth << "," << t.death << "] | " << t.offset
                 << " |\n";

  return 0;
}
