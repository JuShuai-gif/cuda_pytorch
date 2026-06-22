# ===----------------------------------------------------------------------===#
# MLIREdgeConversion - Edge -> Linalg lowering + 端到端流水线注册
# ===----------------------------------------------------------------------===#

add_mlir_dialect_library(MLIREdgeConversion
  EdgeToLinalg/EdgeToLinalg.cpp
  Pipelines.cpp

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
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRArithDialect
  MLIRTransformUtils
)
