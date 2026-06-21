//===- EdgeDialect.cpp - Edge 方言实现 ---------------------------*- C++ -*-===//
//
// 方言的 initialize() 注册算子/类型/属性, 并实现常量物化钩子.
//
//===----------------------------------------------------------------------===//

#include "Edge/EdgeDialect.h"
#include "Edge/EdgeAttrs.h"
#include "Edge/EdgeOps.h"
#include "Edge/EdgeTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::edge;

// 方言定义 (含 EdgeDialect::EdgeDialect 构造) 由 TableGen 生成
#include "Edge/EdgeDialect.cpp.inc"

void EdgeDialect::initialize() {
  // 注册全部算子
  addOperations<
#define GET_OP_LIST
#include "Edge/EdgeOps.cpp.inc"
      >();
  // 注册自定义类型与属性 (实现见 EdgeTypes.cpp / EdgeAttrs.cpp)
  registerTypes();
  registerAttributes();
}

// 常量物化: 当 canonicalizer/常量折叠需要把一个 Attribute 变成常量算子时调用.
// 这是支持常量折叠 (Module 05) 的关键钩子.
Operation *EdgeDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  if (auto elements = llvm::dyn_cast<ElementsAttr>(value))
    return builder.create<ConstantOp>(loc, type, elements);
  return nullptr;
}
