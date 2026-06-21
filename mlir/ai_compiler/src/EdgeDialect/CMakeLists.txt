# ===----------------------------------------------------------------------===#
# MLIREdge - Edge 方言库
#
# 使用官方 add_mlir_dialect_library 宏构建. DEPENDS 确保 TableGen 产物
# (MLIREdgeIncGen) 在编译本库前生成.
# ===----------------------------------------------------------------------===#

add_mlir_dialect_library(MLIREdge
  EdgeDialect.cpp
  EdgeTypes.cpp
  EdgeAttrs.cpp
  EdgeOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${EDGE_SOURCE_DIR}/include/Edge

  DEPENDS
  MLIREdgeIncGen
  MLIREdgeShapeInferenceInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  MLIRFuncDialect
)
