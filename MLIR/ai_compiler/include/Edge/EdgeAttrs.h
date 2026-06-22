//===- EdgeAttrs.h - Edge 方言自定义属性声明 ---------------------*- C++ -*-===//
//
// 包含 TableGen 生成的自定义属性 (QuantParamsAttr) 与枚举属性声明.
//
//===----------------------------------------------------------------------===//

#ifndef EDGE_EDGEATTRS_H
#define EDGE_EDGEATTRS_H

#include "Edge/EdgeDialect.h"
#include "Edge/EdgeEnums.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#define GET_ATTRDEF_CLASSES
#include "Edge/EdgeAttrs.h.inc"

#endif // EDGE_EDGEATTRS_H
