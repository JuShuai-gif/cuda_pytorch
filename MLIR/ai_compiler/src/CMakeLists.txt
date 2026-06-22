# ===----------------------------------------------------------------------===#
# src/ - C++ 实现总入口
#
# 随着模块推进, 这里会逐步加入 Transforms / Conversion / Quantization /
# MemoryPlanner / Runtime / Profiler 等子目录.
# ===----------------------------------------------------------------------===#

add_subdirectory(EdgeDialect)
add_subdirectory(Transforms)
add_subdirectory(Conversion)
add_subdirectory(Runtime)
