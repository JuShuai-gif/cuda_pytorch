# ===----------------------------------------------------------------------===#
# edge-run - 图执行 + Profiling 工具 (Module 11/12)
# ===----------------------------------------------------------------------===#

add_llvm_executable(edge-run edge-run.cpp)
llvm_update_compile_flags(edge-run)
target_link_libraries(edge-run
  PRIVATE
  MLIRIR
  MLIRParser
  MLIRFuncDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRSupport
  MLIREdge
  MLIREdgeRuntime
)
mlir_check_all_link_libraries(edge-run)
