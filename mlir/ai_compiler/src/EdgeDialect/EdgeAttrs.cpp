//===- EdgeAttrs.cpp - Edge 方言自定义属性/枚举实现 --------------*- C++ -*-===//
//
// 包含 TableGen 生成的枚举与属性定义, 并实现 registerAttributes().
//
//===----------------------------------------------------------------------===//

#include "Edge/EdgeAttrs.h"
#include "Edge/EdgeDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::edge;

// 枚举 (Layout) 的 stringify/symbolize 定义
#include "Edge/EdgeEnums.cpp.inc"

// 属性的 get/parse/print 定义
#define GET_ATTRDEF_CLASSES
#include "Edge/EdgeAttrs.cpp.inc"

void EdgeDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Edge/EdgeAttrs.cpp.inc"
      >();
}
