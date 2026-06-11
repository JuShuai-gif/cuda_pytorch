cmake_minimum_required(VERSION 3.18)
project(chapter_08_paged_attention LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -march=native")

add_executable(mini_paged_attention mini_paged_attention.cpp)
