cmake_minimum_required(VERSION 3.18)
project(chapter_12_linear_attention LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(linear_attention linear_attention.cpp)
