cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(trt_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(CUDAToolkit REQUIRED)

# TensorRT is installed system-wide (JetPack apt); no find_package module, so
# point at the headers and link the shared libraries directly.
set(TRT_INCLUDE_DIR "/usr/include/aarch64-linux-gnu" CACHE PATH "TensorRT include dir")
set(TRT_LIB_DIR "/usr/lib/aarch64-linux-gnu" CACHE PATH "TensorRT lib dir")

add_executable(build_engine src/build_engine.cpp)
add_executable(run_engine src/run_engine.cpp)
add_executable(rmsnorm_demo plugin/rmsnorm_plugin.cu plugin/rmsnorm_demo.cu)

foreach(tgt build_engine run_engine rmsnorm_demo)
    target_include_directories(${tgt} PRIVATE ${CMAKE_SOURCE_DIR}/common ${TRT_INCLUDE_DIR})
    target_link_directories(${tgt} PRIVATE ${TRT_LIB_DIR})
    target_link_libraries(${tgt} PRIVATE
        CUDA::cudart
        nvinfer
        nvonnxparser
        nvinfer_plugin)
endforeach()
