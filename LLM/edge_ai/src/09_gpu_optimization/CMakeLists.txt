cmake_minimum_required(VERSION 3.18)
project(GPUOptimizationDemos LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture to match your GPU
# Common values: 70=Volta(V100), 75=Turing(RTX20xx), 80=Ampere(A100),
#                86=Ampere(RTX30xx), 89=Ada(RTX40xx)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Find CUDA toolkit
find_package(CUDA REQUIRED)

add_executable(gpu_optimization_demos
    matmul_naive.cu
    matmul_tiled.cu
    coalesced_demo.cu
    stream_pipeline.cu
    kernel_fusion.cu
    main.cu
)

set_target_properties(gpu_optimization_demos PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_options(gpu_optimization_demos PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O2>
    $<$<COMPILE_LANGUAGE:CUDA>:-O2 -arch=sm_${CMAKE_CUDA_ARCHITECTURES}>
)

# NVTX support for Nsight annotations
# NVTX is bundled with CUDA toolkit (in nvtx3/ subdir since CUDA 12+)
target_compile_definitions(gpu_optimization_demos PRIVATE USE_NVTX)
target_include_directories(gpu_optimization_demos PRIVATE
    ${CUDA_INCLUDE_DIRS}/nvtx3)
