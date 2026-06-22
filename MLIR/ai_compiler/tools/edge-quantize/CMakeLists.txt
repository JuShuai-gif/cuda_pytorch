# ===----------------------------------------------------------------------===#
# edge-quantize - PTQ 量化工具 (Module 07)
# ===----------------------------------------------------------------------===#

add_llvm_executable(edge-quantize edge-quantize.cpp)
llvm_update_compile_flags(edge-quantize)
target_link_libraries(edge-quantize
  PRIVATE
  MLIRIR
  MLIRParser
  MLIRFuncDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRSupport
  MLIREdge
)
mlir_check_all_link_libraries(edge-quantize)
