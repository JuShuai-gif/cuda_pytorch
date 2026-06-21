//===- EdgeDialect.td - Edge dialect 定义 ------------------*- tablegen -*-===//
//
// edge_ai_compiler_pro
//
// 本文件定义 EdgeDialect 这一面向边缘/机器人推理的高层计算图方言.
// 设计动机:
//   类似 TPU-MLIR 的 `top` 方言, 或 torch-mlir 的 `torch` 方言, EdgeDialect
//   作为前端 (ONNX/PyTorch) 与后端 (Linalg/LLVM) 之间的"图层 IR".
//   在这一层做算子融合 / 常量折叠 / 量化等高层优化, 再逐级 lowering.
//
//===----------------------------------------------------------------------===//

#ifndef EDGE_DIALECT
#define EDGE_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Edge dialect 定义
//===----------------------------------------------------------------------===//

def Edge_Dialect : Dialect {
  let name = "edge";
  let summary = "面向边缘/机器人/VLA 部署的高层计算图方言.";
  let description = [{
    EdgeDialect 是一个图层 (graph-level) 方言, 承载神经网络算子 (Conv / Matmul /
    Relu / Attention 等), 用于在 lowering 到 Linalg 之前做计算图优化:
    算子融合, 常量折叠, 量化, 内存规划. 设计上对标 TPU-MLIR 的 top/tpu 方言.
  }];

  // C++ 命名空间, 生成代码会放在 ::mlir::edge 下
  let cppNamespace = "::mlir::edge";

  // 使用默认的 type/attribute 打印解析器 (基于 mnemonic), 配合 TypeDef/AttrDef
  // 的 assemblyFormat 自动生成 parse/print.
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;

  // 允许方言把折叠出的常量物化为 edge.constant (供常量折叠/规范化使用)
  let hasConstantMaterializer = 1;

  // 在 Dialect::initialize() 中我们手动注册类型与属性, 因此声明这两个钩子.
  let extraClassDeclaration = [{
    void registerTypes();
    void registerAttributes();
  }];
}

//===----------------------------------------------------------------------===//
// Edge 算子基类
//===----------------------------------------------------------------------===//

class Edge_Op<string mnemonic, list<Trait> traits = []>
    : Op<Edge_Dialect, mnemonic, traits>;

#endif // EDGE_DIALECT
