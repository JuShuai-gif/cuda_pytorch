cmake_minimum_required(VERSION 3.18)
project(chapter_06_flash_attention_v3 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2 -gencode arch=compute_80,code=sm_80")

add_executable(flash_attention_v3_sim flash_attention_v3_sim.cu)
