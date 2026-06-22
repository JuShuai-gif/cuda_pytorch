//===- EdgeTypes.h - Edge 方言自定义类型声明 ---------------------*- C++ -*-===//
//
// 包含 TableGen 生成的自定义类型 (QuantTensorType) 声明.
//
//===----------------------------------------------------------------------===//

#ifndef EDGE_EDGETYPES_H
#define EDGE_EDGETYPES_H

#include "Edge/EdgeDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "Edge/EdgeTypes.h.inc"

#endif // EDGE_EDGETYPES_H
