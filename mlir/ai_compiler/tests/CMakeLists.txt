# ===----------------------------------------------------------------------===#
# tests/ - lit + FileCheck 回归测试
#
# 复用 LLVM 构建目录中的 llvm-lit / FileCheck (本地 install 未安装它们).
# ===----------------------------------------------------------------------===#

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

# 测试依赖: 只依赖本项目自己的工具 (FileCheck 等来自外部 LLVM 构建目录)
set(EDGE_TEST_DEPENDS edge-opt edge-introspect edge-memplan edge-run edge-quantize)

add_lit_testsuite(check-edge "Running the edge regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${EDGE_TEST_DEPENDS}
)
set_target_properties(check-edge PROPERTIES FOLDER "Tests")

add_lit_testsuites(EDGE ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${EDGE_TEST_DEPENDS})
