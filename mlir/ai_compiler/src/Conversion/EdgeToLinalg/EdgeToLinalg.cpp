//===- EdgeToLinalg.cpp - Edge -> Linalg lowering ----------------*- C++ -*-===//
//
// 用 dialect-conversion 框架把 EdgeDialect 图层算子降级到 Linalg/Tensor/Arith.
// Linalg 采用 destination-passing style (DPS): 结果张量由 outs 提供的 init 张量承载.
//
//===----------------------------------------------------------------------===//

#include "Edge/EdgeOps.h"
#include "Edge/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::edge;

namespace mlir {
namespace edge {

#define GEN_PASS_DEF_LOWEREDGETOLINALG
#include "Edge/Passes.h.inc"

namespace {

// edge.constant -> arith.constant (dense 属性直接复用)
struct ConstantOpLowering : OpConversionPattern<ConstantOp> {
  using OpConversionPattern<ConstantOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto typed = llvm::dyn_cast<TypedAttr>(op.getValue());
    if (!typed)
      return failure();
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, typed);
    return success();
  }
};

// edge.relu -> tensor.empty + linalg.generic(maxf(x, 0))
struct ReluOpLowering : OpConversionPattern<ReluOp> {
  using OpConversionPattern<ReluOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resTy = llvm::dyn_cast<RankedTensorType>(op.getType());
    if (!resTy || !resTy.hasStaticShape())
      return failure();

    Value empty = rewriter.create<tensor::EmptyOp>(loc, resTy.getShape(),
                                                   resTy.getElementType());
    unsigned rank = resTy.getRank();
    AffineMap id = rewriter.getMultiDimIdentityMap(rank);
    llvm::SmallVector<AffineMap> maps = {id, id};
    llvm::SmallVector<utils::IteratorType> iters(rank,
                                                 utils::IteratorType::parallel);

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{resTy}, ValueRange{adaptor.getInput()},
        ValueRange{empty}, maps, iters,
        [&](OpBuilder &b, Location l, ValueRange args) {
          Value zero = b.create<arith::ConstantOp>(
              l, b.getZeroAttr(resTy.getElementType()));
          Value mx = b.create<arith::MaximumFOp>(l, args[0], zero);
          b.create<linalg::YieldOp>(l, mx);
        });
    rewriter.replaceOp(op, generic.getResults());
    return success();
  }
};

// edge.matmul -> tensor.empty + linalg.fill(0) + linalg.matmul  (仅 rank-2)
struct MatmulOpLowering : OpConversionPattern<MatmulOp> {
  using OpConversionPattern<MatmulOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resTy = llvm::dyn_cast<RankedTensorType>(op.getType());
    if (!resTy || !resTy.hasStaticShape() || resTy.getRank() != 2)
      return failure();

    Value empty = rewriter.create<tensor::EmptyOp>(loc, resTy.getShape(),
                                                   resTy.getElementType());
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(resTy.getElementType()));
    Value filled =
        rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{empty})
            .getResult(0);
    auto mm = rewriter.create<linalg::MatmulOp>(
        loc, TypeRange{resTy},
        ValueRange{adaptor.getLhs(), adaptor.getRhs()}, ValueRange{filled});
    rewriter.replaceOp(op, mm.getResults());
    return success();
  }
};

struct LowerEdgeToLinalgPass
    : impl::LowerEdgeToLinalgBase<LowerEdgeToLinalgPass> {
  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<linalg::LinalgDialect, tensor::TensorDialect,
                           arith::ArithDialect, func::FuncDialect>();
    // 仅把已实现 lowering 的算子标记为非法; 其余 Edge 算子保持不变.
    target.addIllegalOp<ConstantOp, ReluOp, MatmulOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<ConstantOpLowering, ReluOpLowering, MatmulOpLowering>(
        &getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createLowerEdgeToLinalgPass() {
  return std::make_unique<LowerEdgeToLinalgPass>();
}

} // namespace edge
} // namespace mlir
