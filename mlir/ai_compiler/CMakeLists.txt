# ===----------------------------------------------------------------------===#
# edge_ai_compiler_pro - 顶层构建脚本
#
# 设计目标:
#   复用本地 MLIR 安装 (/home/ghr/code/llvm-project/install), 完全使用官方
#   MLIR/LLVM 的 CMake 宏 (add_mlir_dialect / add_mlir_dialect_library /
#   MlirOptMain), 不自造任何编译器框架.
#
# 关键构建事实 (已通过 llvm-config 验证):
#   - LLVM/MLIR 版本: 23.0.0git, Release+Asserts
#   - RTTI 关闭 (-fno-rtti), 异常关闭 (-fno-exceptions)  -> 由 HandleLLVMOptions 传播
#   - 本地 MLIR 库为静态库, 且未启用 ASan -> ASan 设为默认 OFF 的可选项
#   - 链接器使用 lld
# ===----------------------------------------------------------------------===#
cmake_minimum_required(VERSION 3.20.0)
project(edge_ai_compiler_pro VERSION 0.1 LANGUAGES C CXX)

# 工程要求 C++20 (现代 CMake + 现代 C++)
set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 导出 compile_commands.json, 供 clangd / IDE 使用
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 默认指向本地 MLIR 安装的 CMake 配置目录, 允许命令行 -DMLIR_DIR=... 覆盖
if(NOT DEFINED MLIR_DIR)
  set(MLIR_DIR "/home/ghr/code/llvm-project/install/lib/cmake/mlir"
      CACHE PATH "Path to MLIRConfig.cmake")
endif()

# 在指定目录寻找 MLIR (会同时带出 LLVM 配置)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# 将 MLIR/LLVM 的 CMake 模块目录加入搜索路径, 以便使用官方构建宏
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

# 导入官方构建辅助模块:
#   TableGen          - 处理 .td 文件
#   AddLLVM/AddMLIR   - add_llvm_executable / add_mlir_dialect 等宏
#   HandleLLVMOptions - 统一编译选项 (会根据 LLVM_ENABLE_RTTI 传播 -fno-rtti 等)
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# ASan 可选开关. 本地 MLIR 库未启用 ASan, 因此默认 OFF;
# 若将来重新编译带 ASan 的 MLIR, 可以 -DEDGE_ENABLE_ASAN=ON 打开.
option(EDGE_ENABLE_ASAN "Build with AddressSanitizer (requires ASan-enabled MLIR)" OFF)
if(EDGE_ENABLE_ASAN)
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address)
endif()

# 头文件搜索路径:
#   - LLVM/MLIR 安装头文件
#   - 本工程公共头文件 (include/)
#   - 构建目录下的生成头文件 (TableGen 产物 *.inc)
set(EDGE_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(EDGE_BINARY_DIR ${PROJECT_BINARY_DIR})

# 统一可执行/库输出目录到 build/bin 与 build/lib (便于 lit 定位工具)
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${EDGE_SOURCE_DIR}/include)
include_directories(${EDGE_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# lit 默认参数
set(LLVM_LIT_ARGS "-sv" CACHE STRING "Default options for lit")

# 子目录:
#   include/ - TableGen 定义与公共头文件 (生成 *IncGen 目标)
#   src/     - C++ 实现 (方言 / 优化 / 转换 / 量化 / 运行时 / Profiler)
#   tools/   - 命令行工具 (edge-opt 等)
#   test/    - lit + FileCheck 测试
add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(tools)

option(EDGE_INCLUDE_TESTS "Generate build targets for tests" ON)
if(EDGE_INCLUDE_TESTS)
  # 本地 install 未安装 lit/FileCheck, 复用 LLVM 构建目录下的 llvm-lit.
  if(NOT DEFINED LLVM_EXTERNAL_LIT)
    set(LLVM_EXTERNAL_LIT "/home/ghr/code/llvm-project/build/bin/llvm-lit"
        CACHE STRING "Path to llvm-lit")
  endif()
  add_subdirectory(tests)
endif()
