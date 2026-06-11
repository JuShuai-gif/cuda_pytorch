cmake_minimum_required(VERSION 3.18)
project(attention_optimization LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA toolkit
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Global compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -march=native")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2 -gencode arch=compute_80,code=sm_80")

# Build output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/chapters)

# Top-level include
include_directories(${CMAKE_SOURCE_DIR})

# Chapter 01: Attention Basics (C++)
add_subdirectory(src/chapter_01 chapter_01_build)

# Chapter 02: GPU Attention (CUDA)
add_subdirectory(src/chapter_02 chapter_02_build)

# Chapter 03: Profiling (C++/CUDA)
add_subdirectory(src/chapter_03 chapter_03_build)

# Chapter 04: FlashAttention V1 (CUDA)
add_subdirectory(src/chapter_04 chapter_04_build)

# Chapter 05: FlashAttention V2 (CUDA)
add_subdirectory(src/chapter_05 chapter_05_build)

# Chapter 06: FlashAttention V3 Simulation (CUDA)
add_subdirectory(src/chapter_06 chapter_06_build)

# Chapter 07: KV Cache (C++)
add_subdirectory(src/chapter_07 chapter_07_build)

# Chapter 08: PagedAttention (C++)
add_subdirectory(src/chapter_08 chapter_08_build)

# Chapter 09: MQA/GQA (C++)
add_subdirectory(src/chapter_09 chapter_09_build)

# Chapter 10: Sliding Window (C++/CUDA)
add_subdirectory(src/chapter_10 chapter_10_build)

# Chapter 11: Sparse Attention (C++/CUDA)
add_subdirectory(src/chapter_11 chapter_11_build)

# Chapter 12: Linear Attention (C++)
add_subdirectory(src/chapter_12 chapter_12_build)

# Chapter 13: Quantized Attention (C++)
add_subdirectory(src/chapter_13 chapter_13_build)

# Chapter 14: TensorRT-LLM Attention (CUDA)
add_subdirectory(src/chapter_14 chapter_14_build)

# Chapter 15: xFormers (no build)
# Chapter 16: vLLM (no build)
