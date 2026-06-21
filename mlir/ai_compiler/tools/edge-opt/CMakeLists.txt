# ===----------------------------------------------------------------------===#
# edge-opt - opt 驱动工具
# ===----------------------------------------------------------------------===#

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIROptLib
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIREdge
  MLIREdgeTransforms
  MLIREdgeConversion
)

add_llvm_executable(edge-opt edge-opt.cpp)
llvm_update_compile_flags(edge-opt)
target_link_libraries(edge-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(edge-opt)
