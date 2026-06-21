# ===----------------------------------------------------------------------===#
# edge-memplan - 张量生命周期分析 + 内存复用规划工具 (Module 09)
# ===----------------------------------------------------------------------===#

add_llvm_executable(edge-memplan edge-memplan.cpp)
llvm_update_compile_flags(edge-memplan)
target_link_libraries(edge-memplan
  PRIVATE
  MLIRIR
  MLIRParser
  MLIRFuncDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRSupport
  MLIREdge
)
mlir_check_all_link_libraries(edge-memplan)
