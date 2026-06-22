//===- EdgeTypes.td - Edge dialect 自定义类型 --------------*- tablegen -*-===//
//
// 自定义类型: QuantTensorType (量化张量类型).
// 工业动机: 量化推理 (INT8/INT4) 是边缘部署的核心. TensorRT 的 IInt8Calibrator,
// TPU-MLIR 的 quant 类型, 都需要在 IR 中显式携带 scale / zeroPoint. 这里用一个
// 自定义类型把"已量化张量"和普通浮点张量在类型系统层面区分开, 便于 pass 推理.
//
//===----------------------------------------------------------------------===//

#ifndef EDGE_TYPES
#define EDGE_TYPES

include "mlir/IR/AttrTypeBase.td"
include "Edge/EdgeDialect.td"

// Edge 类型基类
class Edge_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Edge_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// QuantTensorType: 量化张量
//   - tensorType: 底层 RankedTensorType (提供 shape 与存储元素类型, 如 i8)
//   - scale:      反量化比例 (real = scale * (q - zeroPoint))
//   - zeroPoint:  零点
//===----------------------------------------------------------------------===//
def Edge_QuantTensorType : Edge_Type<"QuantTensor", "qtensor"> {
  let summary = "携带 scale/zeroPoint 的量化张量类型";
  let description = [{
    `!edge.qtensor<tensor<1x3x224x224xi8>, 0.0078125, 0>` 表示一个 INT8 量化张量,
    其反量化公式为 real = scale * (q - zeroPoint).
  }];
  let parameters = (ins
    "::mlir::RankedTensorType":$tensorType,
    "::mlir::FloatAttr":$scale,
    "int64_t":$zeroPoint
  );
  let assemblyFormat = "`<` $tensorType `,` $scale `,` $zeroPoint `>`";
}

#endif // EDGE_TYPES
