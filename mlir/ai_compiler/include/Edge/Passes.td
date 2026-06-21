//===- Passes.td - Edge 方言的 Pass 声明 -------------------*- tablegen -*-===//
//
// 用 ODS 声明 pass, 由 -gen-pass-decls 生成 pass 基类与注册函数.
// 这是 MLIR 内建 pass 的标准定义方式 (mlir/Dialect/*/Transforms/Passes.td).
//
//===----------------------------------------------------------------------===//

#ifndef EDGE_PASSES
#define EDGE_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// ShapeInference: 形状推断 (Analysis 驱动的 Transform)
//===----------------------------------------------------------------------===//
def ShapeInference : Pass<"edge-shape-inference", "::mlir::func::FuncOp"> {
  let summary = "推断并细化 Edge 算子的结果形状 (动态维 ? -> 静态维)";
  let description = [{
    基于 ShapeInferenceOpInterface, 沿 use-def 链把已知的静态形状从输入传播到输出,
    把结果类型里的动态维 `?` 细化为静态维. 这是所有推理编译器的前置步骤:
    没有静态 shape, 后续的内存规划/tiling/量化都无从谈起.
  }];
  let constructor = "::mlir::edge::createShapeInferencePass()";
  let dependentDialects = ["::mlir::edge::EdgeDialect"];
}

//===----------------------------------------------------------------------===//
// Statistics: 计算图统计 (Analysis Pass)
//===----------------------------------------------------------------------===//
def Statistics : Pass<"edge-statistics", "::mlir::ModuleOp"> {
  let summary = "统计算子数量与 MAC, 打印计算图 profile 报告";
  let description = [{
    遍历计算图, 统计各算子出现次数, 并对 conv/matmul 估算乘加次数 (MAC).
    输出 Markdown 风格报告. 这是部署前评估模型计算量/找瓶颈的基础工具,
    对应 TensorRT 的 layer 信息打印 / TVM 的 relay.analysis.
  }];
  let constructor = "::mlir::edge::createStatisticsPass()";
}

//===----------------------------------------------------------------------===//
// IRPrinter: 带标签打印 IR (调试用 Transform, 不改 IR)
//===----------------------------------------------------------------------===//
def IRPrinter : Pass<"edge-ir-printer", "::mlir::ModuleOp"> {
  let summary = "带横幅标签打印当前 IR (流水线调试)";
  let description = [{
    在 pass 流水线的任意位置插入, 打印当前 IR 并带一个标签横幅,
    便于观察某一步前后的 IR. 比 --mlir-print-ir-after-all 更可控.
  }];
  let constructor = "::mlir::edge::createIRPrinterPass()";
  let options = [
    Option<"label", "label", "std::string", /*default=*/"\"IR\"",
           "打印横幅上的标签">
  ];
}

//===----------------------------------------------------------------------===//
// FuseConvBnRelu: Conv+BN+ReLU 融合 (把 BN 折叠进卷积权重/偏置)
//===----------------------------------------------------------------------===//
def FuseConvBnRelu : Pass<"edge-fuse-conv-bn-relu", "::mlir::func::FuncOp"> {
  let summary = "把 relu(batch_norm(conv2d)) 折叠为 edge.conv_bn_relu";
  let description = [{
    匹配 relu(batch_norm(conv2d(x, w, b))) 链, 当 weight 与 BN 参数均为常量时, 把 BN 的
    仿射变换数学折叠进卷积的权重与偏置, 生成单个 edge.conv_bn_relu. 这是 TensorRT builder
    与 TPU-MLIR 的标准推理期优化: 消除 BN 算子 + 减少 kernel 启动与全张量访存.

      factor[c]    = bn_scale[c] / sqrt(bn_var[c] + eps)
      new_w[c]     = w[c] * factor[c]
      new_bias[c]  = (b[c] - bn_mean[c]) * factor[c] + bn_bias[c]
  }];
  let constructor = "::mlir::edge::createFuseConvBnReluPass()";
  let dependentDialects = ["::mlir::edge::EdgeDialect"];
}

//===----------------------------------------------------------------------===//
// LowerEdgeToLinalg: 把 Edge 图层算子 lowering 到 Linalg/Tensor/Arith
//===----------------------------------------------------------------------===//
def LowerEdgeToLinalg : Pass<"edge-lower-to-linalg", "::mlir::func::FuncOp"> {
  let summary = "把 edge.constant/relu/matmul 等降级到 linalg/tensor/arith";
  let description = [{
    用 dialect-conversion 框架 (ConversionPattern + ConversionTarget) 把 EdgeDialect
    的图层算子降级到 Linalg (destination-passing style) + Tensor + Arith. 这是通往
    bufferization -> MemRef -> Affine/SCF -> LLVM 后端路径的第一步.
      edge.constant -> arith.constant
      edge.relu     -> tensor.empty + linalg.generic(maxf(x,0))
      edge.matmul   -> tensor.empty + linalg.fill(0) + linalg.matmul
  }];
  let constructor = "::mlir::edge::createLowerEdgeToLinalgPass()";
  let dependentDialects = [
    "::mlir::linalg::LinalgDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::arith::ArithDialect"
  ];
}

#endif // EDGE_PASSES
