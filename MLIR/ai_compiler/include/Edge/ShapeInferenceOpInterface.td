//===- ShapeInferenceOpInterface.td - 形状推断接口 ---------*- tablegen -*-===//
//
// 自定义 OpInterface: 携带该接口的算子实现 inferShapes(), 根据已知的操作数形状
// 推断/细化结果类型 (把动态维 `?` 细化为静态维). ShapeInferencePass 据此工作.
// 这是 MLIR Toy 教程 Ch.4 的工业化版本, 也是 TPU-MLIR/ONNX-MLIR 形状推断的范式.
//
//===----------------------------------------------------------------------===//

#ifndef EDGE_SHAPE_INFERENCE_OP_INTERFACE
#define EDGE_SHAPE_INFERENCE_OP_INTERFACE

include "mlir/IR/OpBase.td"

def ShapeInferenceOpInterface : OpInterface<"ShapeInferenceOpInterface"> {
  let description = [{
    实现该接口的算子, 能根据其操作数的静态形状推断结果张量的静态形状,
    并就地刷新结果 Value 的类型.
  }];
  let cppNamespace = "::mlir::edge";

  let methods = [
    InterfaceMethod<
      "根据操作数形状推断并刷新结果类型.",
      /*retTy=*/"void",
      /*methodName=*/"inferShapes">
  ];
}

#endif // EDGE_SHAPE_INFERENCE_OP_INTERFACE
