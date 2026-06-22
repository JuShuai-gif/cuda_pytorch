cmake_minimum_required(VERSION 3.22)
project(Chapter03_CUDA_GPU_Acceleration LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- CUDA ---
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# Detect GPU compute capability or use a safe default
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75) # Turing (T4), safe default
endif()

# ================================================================
# CUDA Environment Verification
# ================================================================

add_executable(cuda_hello 00_cuda_hello.cu)

# ================================================================
# CPU Baseline: Vector Addition
# ================================================================

add_executable(vector_add_cpu 01_vector_add_cpu.cpp)

# ================================================================
# CUDA Vector Addition: Step-by-Step Evolution
# ================================================================

add_executable(vector_add_cuda 02_vector_add_cuda.cu)

add_executable(vector_add_parallel 03_vector_add_parallel.cu)

add_executable(vector_add_grid 04_vector_add_grid.cu)

# ================================================================
# CUDA Error Handling
# ================================================================

add_executable(cuda_error_handling 05_cuda_error_handling.cu)

# ================================================================
# CUDA dim3: 2D and 3D Grid/Block
# ================================================================

add_executable(cuda_dim3 06_cuda_dim3.cu)

# ================================================================
# CUDA Profiling and Optimization
# ================================================================

add_executable(cuda_profiling 07_cuda_profiling.cu)

add_executable(cuda_optimization 08_cuda_optimization.cu)
