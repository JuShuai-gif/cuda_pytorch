cmake_minimum_required(VERSION 3.18)
project(chapter_11_sparse_attention LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(block_sparse_attention block_sparse_attention.cpp)
