cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(CudaTest LANGUAGES CXX CUDA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

message("[TOP] CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
message("[TOP] CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")

# 查找CUDA工具包
find_package(CUDAToolkit REQUIRED)

# 启用可重定位设备代码（便于调试）
set(CUDA_SEPARABLE_COMPILATION ON)

include_directories(./include)

# 添加可执行文件
# add_executable(cuda_test src/vector_add.cu)

# add_subdirectory(base)

add_subdirectory(pytorch_bind)

add_subdirectory(ops)

# 设置CUDA架构（根据您的GPU调整）
# set_target_properties(cuda_test PROPERTIES
#     CUDA_ARCHITECTURES "75"  # Tesla T4/RTX 20xx: 75; A100: 80; RTX 30xx: 86
# )

# 添加测试
# enable_testing()
# add_test(NAME RunCudaTest COMMAND cuda_test)