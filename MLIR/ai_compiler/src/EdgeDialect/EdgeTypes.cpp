//===- EdgeTypes.cpp - Edge 方言自定义类型实现 -------------------*- C++ -*-===//
//
// 包含 TableGen 生成的类型定义, 并实现 registerTypes().
//
//===----------------------------------------------------------------------===//

#include "Edge/EdgeTypes.h"
#include "Edge/EdgeDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::edge;

// 类型的 get/parse/print 等定义由 TableGen 生成
#define GET_TYPEDEF_CLASSES
#include "Edge/EdgeTypes.cpp.inc"

void EdgeDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Edge/EdgeTypes.cpp.inc"
      >();
}
