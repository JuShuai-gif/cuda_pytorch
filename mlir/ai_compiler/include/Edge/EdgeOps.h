//===- EdgeOps.h - Edge 方言算子声明 -----------------------------*- C++ -*-===//
//
// 包含 TableGen 生成的算子类 (Conv2DOp / ReluOp / MatmulOp / AttentionOp ...).
//
//===----------------------------------------------------------------------===//

#ifndef EDGE_EDGEOPS_H
#define EDGE_EDGEOPS_H

#include "Edge/EdgeAttrs.h"
#include "Edge/EdgeDialect.h"
#include "Edge/EdgeTypes.h"
#include "Edge/ShapeInferenceOpInterface.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "Edge/EdgeOps.h.inc"

#endif // EDGE_EDGEOPS_H
