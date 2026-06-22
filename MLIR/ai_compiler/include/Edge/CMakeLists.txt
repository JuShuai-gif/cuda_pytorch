# ===----------------------------------------------------------------------===#
# Edge 方言的 TableGen 生成规则
#
# 这里不使用 add_mlir_dialect (它不生成 attribute/enum), 而是手动罗列全部
# mlir_tablegen 调用, 以便同时生成 op / dialect / type / attribute / enum /
# interface / pass 的代码. 这是 torch-mlir / IREE 等大型项目的常见做法.
# ===----------------------------------------------------------------------===#

# ---- 方言主体: op / dialect / type / attribute / enum ----
set(LLVM_TARGET_DEFINITIONS EdgeOps.td)
mlir_tablegen(EdgeOps.h.inc -gen-op-decls)
mlir_tablegen(EdgeOps.cpp.inc -gen-op-defs)
mlir_tablegen(EdgeDialect.h.inc -gen-dialect-decls -dialect=edge)
mlir_tablegen(EdgeDialect.cpp.inc -gen-dialect-defs -dialect=edge)
mlir_tablegen(EdgeTypes.h.inc -gen-typedef-decls -typedefs-dialect=edge)
mlir_tablegen(EdgeTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=edge)
mlir_tablegen(EdgeAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=edge)
mlir_tablegen(EdgeAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=edge)
mlir_tablegen(EdgeEnums.h.inc -gen-enum-decls)
mlir_tablegen(EdgeEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIREdgeIncGen)

# ---- 自定义 OpInterface: 形状推断 ----
set(LLVM_TARGET_DEFINITIONS ShapeInferenceOpInterface.td)
mlir_tablegen(ShapeInferenceOpInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(ShapeInferenceOpInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIREdgeShapeInferenceInterfaceIncGen)

# ---- Pass 声明/注册 ----
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Edge)
add_public_tablegen_target(MLIREdgePassIncGen)

# ---- 文档 ----
add_mlir_doc(EdgeDialect EdgeDialect Edge/ -gen-dialect-doc)
add_mlir_doc(EdgeOps EdgeOps Edge/ -gen-op-doc)
