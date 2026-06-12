cmake_minimum_required(VERSION 3.18)
project(chapter_03_profiling LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2 -lineinfo")

add_executable(profiling_benchmark benchmark.cu)
target_compile_options(profiling_benchmark PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_80,code=sm_80>)
add_test(NAME chapter_03_profiling_benchmark COMMAND profiling_benchmark)
