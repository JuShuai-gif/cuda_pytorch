cmake_minimum_required(VERSION 3.18)
project(CPUGPUHybridBench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(cpu_gpu_bench
    main.cu
    cpu_preprocess.cpp
    pipeline_runner.cpp
    gpu_inference.cu
)

set_target_properties(cpu_gpu_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES native
)

target_link_libraries(cpu_gpu_bench PRIVATE CUDA::cudart)

target_compile_options(cpu_gpu_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O2>
    $<$<COMPILE_LANGUAGE:CUDA>:-O2>
)
