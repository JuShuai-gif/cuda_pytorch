//===- EdgeOps.td - Edge dialect 算子定义 ------------------*- tablegen -*-===//
//
// 神经网络图层算子: Constant / Conv2D / BatchNorm / Relu / Matmul / Attention,
// 以及融合算子 ConvBnRelu (Conv+BN+ReLU 融合的目标算子).
//
// 设计原则:
//   算子的输入/输出统一使用内建 RankedTensor 类型, 以便后续无摩擦地 lowering
//   到 Linalg/TOSA. 量化相关用自定义 qtensor 类型与 quant_params 属性承载.
//   这与 torch-mlir / TPU-MLIR 的图层方言设计一致.
//
//===----------------------------------------------------------------------===//

#ifndef EDGE_OPS
#define EDGE_OPS

include "Edge/EdgeTypes.td"
include "Edge/EdgeAttrs.td"
include "Edge/ShapeInferenceOpInterface.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// edge.constant : 常量张量 (常量折叠的源头)
//===----------------------------------------------------------------------===//
def Edge_ConstantOp : Edge_Op<"constant", [Pure, ConstantLike]> {
  let summary = "常量张量";
  let description = [{ 产生一个由 dense 属性指定的常量张量, 是常量折叠的基础. }];
  let arguments = (ins ElementsAttr:$value);
  let results = (outs AnyRankedTensor:$output);
  // dense 字面量自带类型 (dense<...> : tensor<...>), 故用自定义 parse/print,
  // 由 value 属性的类型推导结果类型 (与官方 Toy ConstantOp 同).
  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// edge.relu : 逐元素 ReLU
//===----------------------------------------------------------------------===//
def Edge_ReluOp : Edge_Op<"relu", [Pure, SameOperandsAndResultType]> {
  let summary = "逐元素 ReLU: max(x, 0)";
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

//===----------------------------------------------------------------------===//
// edge.conv2d : 二维卷积
//   strides/pads/dilations 用 DenseI64Array 承载, layout 用枚举承载.
//   bias 为可选输入 (无 bias 时由后续 BN 融合补上).
//===----------------------------------------------------------------------===//
def Edge_Conv2DOp : Edge_Op<"conv2d", [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "二维卷积 (NCHW)";
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$weight,
    Optional<AnyRankedTensor>:$bias,
    DenseI64ArrayAttr:$strides,
    DenseI64ArrayAttr:$pads,
    DenseI64ArrayAttr:$dilations,
    DefaultValuedAttr<I64Attr, "1">:$group
  );
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = [{
    $input `,` $weight (`,` $bias^)? attr-dict
    `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// edge.batch_norm : 批归一化 (推理期可被折叠进卷积)
//===----------------------------------------------------------------------===//
def Edge_BatchNormOp : Edge_Op<"batch_norm", [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "批归一化 (推理期: y = scale*(x-mean)/sqrt(var+eps) + bias)";
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$scale,
    AnyRankedTensor:$bias,
    AnyRankedTensor:$mean,
    AnyRankedTensor:$variance,
    DefaultValuedAttr<F64Attr, "1e-5">:$epsilon
  );
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = [{
    $input `,` $scale `,` $bias `,` $mean `,` $variance attr-dict
    `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// edge.matmul : 矩阵乘
//===----------------------------------------------------------------------===//
def Edge_MatmulOp : Edge_Op<"matmul", [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "矩阵乘 lhs x rhs";
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// edge.attention : 缩放点积注意力 (VLA 策略网络核心算子)
//   softmax(Q K^T / sqrt(d)) V
//===----------------------------------------------------------------------===//
def Edge_AttentionOp : Edge_Op<"attention", [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "缩放点积注意力 softmax(QK^T*scale)V";
  let description = [{
    Transformer/VLA 策略网络的核心算子. 后端可 lowering 为 FlashAttention 风格
    的融合 kernel, 这是机器人推理延迟优化的关键路径.
  }];
  let arguments = (ins
    AnyRankedTensor:$query,
    AnyRankedTensor:$key,
    AnyRankedTensor:$value,
    DefaultValuedAttr<F64Attr, "1.0">:$scale
  );
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = [{
    $query `,` $key `,` $value attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// edge.conv_bn_relu : Conv+BN+ReLU 融合后的目标算子
//   这是 Module 05 图优化里 ConvBnReluFusion 的产物.
//===----------------------------------------------------------------------===//
def Edge_ConvBnReluOp : Edge_Op<"conv_bn_relu", [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "融合算子: Conv2D + BatchNorm + ReLU";
  let description = [{
    将 conv2d -> batch_norm -> relu 三个算子融合成单算子, 减少访存与 kernel 启动
    开销. TensorRT 的 ConvBNReLU 融合, TPU-MLIR 的 conv+bn 折叠都是同一思想.
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$weight,
    AnyRankedTensor:$bias,
    DenseI64ArrayAttr:$strides,
    DenseI64ArrayAttr:$pads,
    DenseI64ArrayAttr:$dilations,
    DefaultValuedAttr<I64Attr, "1">:$group
  );
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = [{
    $input `,` $weight `,` $bias attr-dict
    `:` functional-type(operands, results)
  }];
}

#endif // EDGE_OPS
