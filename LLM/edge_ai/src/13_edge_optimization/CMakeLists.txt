cmake_minimum_required(VERSION 3.14)
project(EdgeOptimization LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(edge_optimization
    main.cpp
    memory_bench.cpp
    neon_convert.cpp
    fail_closed.cpp
)

target_compile_options(edge_optimization PRIVATE -Wall -Wextra -O2 -pthread)

# ARM NEON: 交叉编译或原生 aarch64 时启用
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|armv8|arm64")
    target_compile_options(edge_optimization PRIVATE -mfpu=neon)
    target_compile_definitions(edge_optimization PRIVATE __aarch64__=1)
    message(STATUS "ARM aarch64 架构检测到 - 启用 NEON 指令集")
else()
    message(STATUS "非 ARM 架构（${CMAKE_SYSTEM_PROCESSOR}）- NEON 将以标量 fallback 运行")
endif()

# x86_64: 可选的 AVX2（用于更准确的 uncached 模拟）
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(edge_optimization PRIVATE -mavx2)
        message(STATUS "x86_64 架构 - 已启用 AVX2")
    endif()
endif()

target_link_libraries(edge_optimization PRIVATE pthread)

# 发布构建建议
# 使用: cmake .. -DCMAKE_BUILD_TYPE=Release 以获得最佳性能
