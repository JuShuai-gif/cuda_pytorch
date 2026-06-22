cmake_minimum_required(VERSION 3.18)
project(chapter_10_sliding_window LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -march=native")

add_executable(sliding_window_attention sliding_window_attention.cpp)
add_test(NAME chapter_10_sliding_window_attention COMMAND sliding_window_attention)
