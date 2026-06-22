# ===----------------------------------------------------------------------===#
# edge-introspect - IR 遍历/打印/分析工具 (Module 01 配套)
# ===----------------------------------------------------------------------===#

add_llvm_executable(edge-introspect edge-introspect.cpp)
llvm_update_compile_flags(edge-introspect)
target_link_libraries(edge-introspect
  PRIVATE
  MLIRIR
  MLIRParser
  MLIRFuncDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRSupport
  MLIREdge
)
mlir_check_all_link_libraries(edge-introspect)
