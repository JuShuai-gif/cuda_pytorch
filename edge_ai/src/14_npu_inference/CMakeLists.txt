cmake_minimum_required(VERSION 3.14)
project(NPUInferenceOptimization LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(npu_inference_demo
    main.cpp
    io_persistence.cpp
    double_letterbox.cpp
    pipeline_sim.cpp
)

target_compile_options(npu_inference_demo PRIVATE -Wall -Wextra -O3 -march=native -pthread)
target_link_libraries(npu_inference_demo PRIVATE pthread)

# 提示: 对于 ARM 平台 (RK3588)，可用 -mfpu=neon 启用 NEON 指令集
# target_compile_options(npu_inference_demo PRIVATE -mfpu=neon -march=armv8-a)
