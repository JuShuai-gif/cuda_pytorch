cmake_minimum_required(VERSION 3.18)
project(chapter_02_gpu_attention LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2")

add_executable(naive_attention_gpu cuda_naive_attention.cu)
target_compile_options(naive_attention_gpu PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_80,code=sm_80>)

add_executable(chapter_02_unit_test unit_test.cu)
target_compile_options(chapter_02_unit_test PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_80,code=sm_80>)
add_test(NAME chapter_02_unit_test COMMAND chapter_02_unit_test)
