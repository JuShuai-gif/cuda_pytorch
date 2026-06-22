# Final Project CMake Configuration

cmake_minimum_required(VERSION 3.18)
project(mini_trt_attention_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2 -gencode arch=compute_80,code=sm_80")

include_directories(include)

add_executable(attention_engine
    src/attention_engine.cu
    src/flash_attn.cu
    src/kv_cache.cu
    src/paged_attn.cu
    src/batcher.cu
)

add_executable(test_attention
    tests/test_attention.cpp
    src/attention_engine.cu
    src/flash_attn.cu
    src/kv_cache.cu
    src/paged_attn.cu
    src/batcher.cu
)
target_include_directories(test_attention PRIVATE include)
