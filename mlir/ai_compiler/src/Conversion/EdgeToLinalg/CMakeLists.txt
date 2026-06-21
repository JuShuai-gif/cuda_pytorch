# ===----------------------------------------------------------------------===#
# MLIREdgeConversion - Edge -> Linalg 等跨方言 lowering 库
# ===----------------------------------------------------------------------===#

add_mlir_dialect_library(MLIREdgeConversion
  EdgeToLinalg.cpp

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
