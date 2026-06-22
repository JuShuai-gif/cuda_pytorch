cmake_minimum_required(VERSION 3.18)
project(chapter_13_quantized_attention LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(quantized_attention quantized_attention.cpp)
add_test(NAME chapter_13_quantized_attention COMMAND quantized_attention)
