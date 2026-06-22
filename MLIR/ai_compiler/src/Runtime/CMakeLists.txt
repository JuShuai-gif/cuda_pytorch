# ===----------------------------------------------------------------------===#
# MLIREdgeRuntime - 图执行运行时 + Profiler
# ===----------------------------------------------------------------------===#

add_mlir_library(MLIREdgeRuntime
  Runtime.cpp

  ADDITIONAL_HEADER_DIRS
  ${EDGE_SOURCE_DIR}/include/Edge

  DEPENDS
  MLIREdgeIncGen
  MLIREdgeShapeInferenceInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIREdge
  MLIRIR
  MLIRFuncDialect
  MLIRSupport
)
