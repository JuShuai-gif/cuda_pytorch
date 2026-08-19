cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cuda_core LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Target this machine's GPU (NVIDIA Thor, sm_110).  Override with
# -DCUDA_CORE_ARCH=90 for an A100, etc.
set(CUDA_CORE_ARCH "110" CACHE STRING "CUDA compute capability (no dot)")
set(CMAKE_CUDA_ARCHITECTURES "${CUDA_CORE_ARCH}")

# Show per-kernel register / shared-memory usage at compile time.  This is the
# first-hand evidence for occupancy limits.
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xptxas> $<$<COMPILE_LANGUAGE:CUDA>:-v>)

find_package(CUDAToolkit REQUIRED)

add_executable(coalescing src/coalescing.cu)
add_executable(bank_conflict src/bank_conflict.cu)
add_executable(occupancy src/occupancy.cu)
add_executable(async_copy src/async_copy.cu)
add_executable(stream_overlap src/stream_overlap.cu)
add_executable(graph_launch src/graph_launch.cu)

foreach(tgt coalescing bank_conflict occupancy async_copy stream_overlap graph_launch)
    target_include_directories(${tgt} PRIVATE ${CMAKE_SOURCE_DIR}/common)
    target_link_libraries(${tgt} PRIVATE CUDA::cudart)
endforeach()
