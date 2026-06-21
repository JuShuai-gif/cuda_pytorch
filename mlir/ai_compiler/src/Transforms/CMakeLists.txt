# ===----------------------------------------------------------------------===#
# MLIREdgeTransforms - Edge 方言的优化/分析 Pass 库
# ===----------------------------------------------------------------------===#

add_mlir_dialect_library(MLIREdgeTransforms
  ShapeInference.cpp
  Statistics.cpp
  IRPrinter.cpp
  ConvBnReluFusion.cpp

  ADDITIONAL_HEADER_DIRS
  ${EDGE_SOURCE_DIR}/include/Edge

  DEPENDS
  MLIREdgeIncGen
  MLIREdgeShapeInferenceInterfaceIncGen
  MLIREdgePassIncGen

  LINK_LIBS PUBLIC
  MLIREdge
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRTransformUtils
)
