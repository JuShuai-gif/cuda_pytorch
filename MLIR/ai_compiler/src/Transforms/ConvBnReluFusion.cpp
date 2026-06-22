//===- ConvBnReluFusion.cpp - Conv+BN+ReLU 融合 ------------------*- C++ -*-===//
//
// 匹配 relu(batch_norm(conv2d(x, w, b))) 链, 当 weight 与 BN 参数均为常量时, 把 BN
// 的仿射变换数学折叠进卷积权重/偏置, 生成单个 edge.conv_bn_relu.
//
//   factor[c]   = bn_scale[c] / sqrt(bn_var[c] + eps)
//   new_w[c]    = w[c] * factor[c]
//   new_bias[c] = (b[c] - bn_mean[c]) * factor[c] + bn_bias[c]
//
// 这是 TensorRT builder / TPU-MLIR 的标准推理期优化: 消除 BN, 减少 kernel 启动与访存.
//
//===----------------------------------------------------------------------===//

#include "Edge/EdgeOps.h"
#include "Edge/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

#include <cmath>

using namespace mlir;
using namespace mlir::edge;

namespace mlir {
namespace edge {

#define GEN_PASS_DEF_FUSECONVBNRELU
#include "Edge/Passes.h.inc"

namespace {

// 从一个 Value 取其 edge.constant 的 DenseElementsAttr (取不到返回 null).
static DenseElementsAttr getConstDense(Value v) {
  if (auto c = v.getDefiningOp<ConstantOp>())
    return llvm::dyn_cast<DenseElementsAttr>(c.getValue());
  return {};
}

// 把 DenseElementsAttr<f32> 读进 SmallVector<float>.
static void readF32(DenseElementsAttr attr, llvm::SmallVectorImpl<float> &out) {
  for (float f : attr.getValues<float>())
    out.push_back(f);
}

struct ConvBnReluFusionPattern : OpRewritePattern<ReluOp> {
  using OpRewritePattern<ReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReluOp relu,
                                PatternRewriter &rewriter) const override {
    // 1) 匹配 relu <- batch_norm <- conv2d 的单使用链
    auto bn = relu.getInput().getDefiningOp<BatchNormOp>();
    if (!bn || !bn->hasOneUse())
      return failure();
    auto conv = bn.getInput().getDefiningOp<Conv2DOp>();
    if (!conv || !conv->hasOneUse())
      return failure();

    // 2) weight 与 BN 参数必须是常量, 且 weight 为 4D f32
    auto weightTy = llvm::dyn_cast<RankedTensorType>(conv.getWeight().getType());
    if (!weightTy || weightTy.getRank() != 4 ||
        !weightTy.getElementType().isF32())
      return failure();

    DenseElementsAttr wAttr = getConstDense(conv.getWeight());
    DenseElementsAttr scaleAttr = getConstDense(bn.getScale());
    DenseElementsAttr bnBiasAttr = getConstDense(bn.getBias());
    DenseElementsAttr meanAttr = getConstDense(bn.getMean());
    DenseElementsAttr varAttr = getConstDense(bn.getVariance());
    if (!wAttr || !scaleAttr || !bnBiasAttr || !meanAttr || !varAttr)
      return failure();

    int64_t cout = weightTy.getDimSize(0);
    int64_t perFilter = weightTy.getNumElements() / cout; // Cin*kH*kW

    // 3) 读取数据
    llvm::SmallVector<float> w, scale, bnBias, mean, var;
    readF32(wAttr, w);
    readF32(scaleAttr, scale);
    readF32(bnBiasAttr, bnBias);
    readF32(meanAttr, mean);
    readF32(varAttr, var);
    if ((int64_t)scale.size() != cout || (int64_t)bnBias.size() != cout ||
        (int64_t)mean.size() != cout || (int64_t)var.size() != cout)
      return failure();

    // conv 原始 bias (可选), 缺省为 0
    llvm::SmallVector<float> origBias(cout, 0.0f);
    if (conv.getBias()) {
      DenseElementsAttr bAttr = getConstDense(conv.getBias());
      if (!bAttr)
        return failure();
      llvm::SmallVector<float> tmp;
      readF32(bAttr, tmp);
      if ((int64_t)tmp.size() != cout)
        return failure();
      origBias.assign(tmp.begin(), tmp.end());
    }

    double eps = bn.getEpsilon().convertToDouble();

    // 4) 折叠计算
    llvm::SmallVector<float> newW(cout * perFilter);
    llvm::SmallVector<float> newB(cout);
    for (int64_t c = 0; c < cout; ++c) {
      double factor = scale[c] / std::sqrt((double)var[c] + eps);
      newB[c] = (float)((origBias[c] - mean[c]) * factor + bnBias[c]);
      for (int64_t j = 0; j < perFilter; ++j)
        newW[c * perFilter + j] = (float)(w[c * perFilter + j] * factor);
    }

    // 5) 物化新常量
    Location loc = relu.getLoc();
    auto newWAttr = DenseElementsAttr::get(weightTy, llvm::ArrayRef<float>(newW));
    auto biasTy = RankedTensorType::get({cout}, rewriter.getF32Type());
    auto newBAttr = DenseElementsAttr::get(biasTy, llvm::ArrayRef<float>(newB));
    Value newWeight = rewriter.create<ConstantOp>(loc, weightTy, newWAttr);
    Value newBias = rewriter.create<ConstantOp>(loc, biasTy, newBAttr);

    // 6) 用 edge.conv_bn_relu 替换 relu (旧 conv/bn/常量因 Pure 会被 DCE)
    rewriter.replaceOpWithNewOp<ConvBnReluOp>(
        relu, relu.getType(), conv.getInput(), newWeight, newBias,
        conv.getStrides(), conv.getPads(), conv.getDilations(), conv.getGroup());
    return success();
  }
};

struct FuseConvBnReluPass : impl::FuseConvBnReluBase<FuseConvBnReluPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvBnReluFusionPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createFuseConvBnReluPass() {
  return std::make_unique<FuseConvBnReluPass>();
}

} // namespace edge
} // namespace mlir
