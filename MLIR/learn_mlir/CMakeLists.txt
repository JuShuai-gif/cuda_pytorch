cmake_minimum_required(VERSION 3.12)
project(MLIR-Toy VERSION 0.0 LANGUAGES C CXX)

# 强制使用C++17标准，且要求编译器必须支持
# MLIR/LLVM 和 Toy教程代码都基于C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 生成 compile_commands.json 文件
# 便于 clangd（或者 VSCode、CLion 等 IDE）获取编译参数，实现语法提示、跳转等功能
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # for clangd

# 指定MLIR的CMake配置文件目录
# 安装MLIR后，它的cmake配置文件位于，上面设置的-DCMAKE_INSTALL_PREFIX=/path/to/install 内，可以让 find_package(MLIR) 能够找到 MLIR
set(MLIR_DIR /home/ghr/code/llvm-project/install/lib/cmake/mlir) # set /path/to/install

# 在指定目录下寻找MLIR
find_package(MLIR REQUIRED CONFIG)

# 将 MLIR 和 LLVM 提供的 CMake 模块路径加入 CMAKE_MODULE_PATH，便于后续使用它们的自定义 CMake 宏和函数（如 AddMLIR、HandleLLVMOptions）
list(APPEND CMAKE_MODULE_PATH ${MLIR_CMAKE_DIR})
list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})

# 导入 LLVM/MLIR 提供的一些构建辅助模块
# TableGen：处理 .td 文件生成代码的工具支持（LLVM/MLIR 的核心技术）
# AddLLVM：提供添加 LLVM 组件的 CMake 宏
# AddMLIR：提供添加 MLIR 组件的 CMake 宏
# HandleLLVMOptions：统一处理 LLVM/MLIR 推荐的编译选项
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# 将 MLIR 和 LLVM 的头文件目录添加到编译器搜索路径中
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})

add_executable(mlir_toy main.cpp)

# 为目标 mlir-toy 添加额外的链接选项
target_link_options(mlir_toy PRIVATE -fuse-ld=lld) # if you use lld


# 为目标 mlir-toy 链接以下库：
# MLIRIR：MLIR 的核心中间表示库
# MLIRParser：MLIR 文本格式解析库
# MLIRArithDialect：MLIR 内置的算术方言（arith dialect）
# MLIRFuncDialect：MLIR 内置的函数方言（func dialect）
target_link_libraries(mlir_toy
  PRIVATE
  MLIRIR
  MLIRParser
  MLIRArithDialect
  MLIRFuncDialect
  -fsanitize=address # if you enabled AddressSanitizer when building MLIR
)