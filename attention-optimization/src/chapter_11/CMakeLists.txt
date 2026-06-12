cmake_minimum_required(VERSION 3.18)
project(chapter_11_sparse_attention LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -march=native")

add_executable(block_sparse_attention block_sparse_attention.cpp)
add_test(NAME chapter_11_block_sparse_attention COMMAND block_sparse_attention)
