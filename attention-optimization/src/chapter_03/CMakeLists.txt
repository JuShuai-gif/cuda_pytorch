cmake_minimum_required(VERSION 3.18)
project(chapter_03_profiling LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2 -lineinfo")

# NVTX benchmark helper
add_library(nvtx_benchmark STATIC benchmark.cu)
target_include_directories(nvtx_benchmark PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(nvtx_benchmark PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_80,code=sm_80>)
