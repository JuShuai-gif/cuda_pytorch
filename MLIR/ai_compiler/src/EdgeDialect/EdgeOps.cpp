//===- EdgeOps.cpp - Edge 方言算子实现 ---------------------------*- C++ -*-===//
//
// 包含 TableGen 生成的算子定义, 并实现需要 C++ 的部分:
//   - ConstantOp 折叠
//   - ShapeInferenceOpInterface::inferShapes 各算子实现
//
//===----------------------------------------------------------------------===//

#include "Edge/EdgeOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::edge;

// 形状推断接口的方法分派定义 (由 -gen-op-interface-defs 生成)
#include "Edge/ShapeInferenceOpInterface.cpp.inc"

// 算子的 build/verify/parse/print 等定义由 TableGen 生成
#define GET_OP_CLASSES
#include "Edge/EdgeOps.cpp.inc"

//===----------------------------------------------------------------------===//
// ConstantOp::fold
//   常量算子折叠时直接返回其携带的 value 属性. 这是常量传播的基础.
//===----------------------------------------------------------------------===//
OpFoldResult ConstantOp::fold(ConstantOp::FoldAdaptor adaptor) {
  return getValue();
}

// dense 字面量自带类型, 故自定义 parse/print, 由 value 属性推导结果类型.
mlir::ParseResult ConstantOp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  mlir::DenseElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(value, "value", result.attributes))
    return mlir::failure();
  result.addTypes(value.getType());
  return mlir::success();
}

void ConstantOp::print(mlir::OpAsmPrinter &printer) {
  printer << " ";
  printer.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
  printer << getValue();
}

//===----------------------------------------------------------------------===//
// 形状推断辅助
//===----------------------------------------------------------------------===//
namespace {
// 卷积单维输出: out = floor((in + pBegin + pEnd - dil*(k-1) - 1)/stride) + 1
static int64_t convOutDim(int64_t in, int64_t k, int64_t stride, int64_t pBegin,
                          int64_t pEnd, int64_t dil) {
  if (ShapedType::isDynamic(in) || ShapedType::isDynamic(k))
    return ShapedType::kDynamic;
  return (in + pBegin + pEnd - dil * (k - 1) - 1) / stride + 1;
}
} // namespace

//===----------------------------------------------------------------------===//
// Conv2DOp / ConvBnReluOp 共享的卷积形状推断 (NCHW)
//===----------------------------------------------------------------------===//
template <typename ConvOpT>
static void inferConvShape(ConvOpT op) {
  auto in = llvm::dyn_cast<RankedTensorType>(op.getInput().getType());
  auto w = llvm::dyn_cast<RankedTensorType>(op.getWeight().getType());
  if (!in || !w || in.getRank() != 4 || w.getRank() != 4)
    return;
  ArrayRef<int64_t> strides = op.getStrides();
  ArrayRef<int64_t> pads = op.getPads();
  ArrayRef<int64_t> dil = op.getDilations();
  if (strides.size() < 2 || pads.size() < 4 || dil.size() < 2)
    return;

  int64_t n = in.getDimSize(0);
  int64_t h = in.getDimSize(2);
  int64_t wid = in.getDimSize(3);
  int64_t cout = w.getDimSize(0);
  int64_t kh = w.getDimSize(2);
  int64_t kw = w.getDimSize(3);

  // pads 约定: [pad_h_begin, pad_w_begin, pad_h_end, pad_w_end]
  int64_t hout = convOutDim(h, kh, strides[0], pads[0], pads[2], dil[0]);
  int64_t wout = convOutDim(wid, kw, strides[1], pads[1], pads[3], dil[1]);

  op.getResult().setType(
      RankedTensorType::get({n, cout, hout, wout}, in.getElementType()));
}

void Conv2DOp::inferShapes() { inferConvShape(*this); }
void ConvBnReluOp::inferShapes() { inferConvShape(*this); }

//===----------------------------------------------------------------------===//
// BatchNormOp: 输出形状 == 输入形状
//===----------------------------------------------------------------------===//
void BatchNormOp::inferShapes() { getResult().setType(getInput().getType()); }

//===----------------------------------------------------------------------===//
// AttentionOp: 输出形状 == query 形状
//===----------------------------------------------------------------------===//
void AttentionOp::inferShapes() { getResult().setType(getQuery().getType()); }

//===----------------------------------------------------------------------===//
// MatmulOp: [.., M, K] x [.., K, N] -> [.., M, N]
//===----------------------------------------------------------------------===//
void MatmulOp::inferShapes() {
  auto lhs = llvm::dyn_cast<RankedTensorType>(getLhs().getType());
  auto rhs = llvm::dyn_cast<RankedTensorType>(getRhs().getType());
  if (!lhs || !rhs || lhs.getRank() < 2 || rhs.getRank() < 2)
    return;
  llvm::SmallVector<int64_t> shape(lhs.getShape().begin(),
                                   lhs.getShape().end());
  // 末维 N 取自 rhs 的最后一维, M (倒数第二维) 保持 lhs 不变
  shape.back() = rhs.getShape().back();
  getResult().setType(RankedTensorType::get(shape, lhs.getElementType()));
}
