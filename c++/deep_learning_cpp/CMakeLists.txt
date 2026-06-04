cmake_minimum_required(VERSION 3.22)
project(Deep_Learning_with_CPlusPlus LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90" CACHE STRING "CUDA architectures to target")

# External library paths — override via cmake -DLIBTORCH_ROOT=/path/to/libtorch
set(LIBTORCH_ROOT "$ENV{HOME}/Downloads/libtorch" CACHE PATH "Path to LibTorch installation")
set(ONNXRUNTIME_ROOT "$ENV{HOME}/Downloads/onnxruntime-linux-x64-gpu-1.25.1" CACHE PATH "Path to ONNX Runtime installation")

option(BUILD_CHAPTERS "Build all chapter examples" ON)

if(BUILD_CHAPTERS)
    add_subdirectory(src/chapter-01)
    add_subdirectory(src/chapter-02)
    add_subdirectory(src/chapter-03)
    add_subdirectory(src/chapter-04)
    add_subdirectory(src/chapter-05)
    add_subdirectory(src/chapter-06)
    add_subdirectory(src/chapter-07)
    add_subdirectory(src/chapter-08)
    add_subdirectory(src/chapter-09)
    add_subdirectory(src/chapter-10)
    add_subdirectory(src/chapter-11)
    add_subdirectory(src/chapter-12)
    add_subdirectory(src/chapter-13)
    add_subdirectory(src/chapter-14)
endif()
