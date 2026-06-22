cmake_minimum_required(VERSION 3.22)
project(Chapter04_BasicNeuralNetwork LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- LibTorch ---
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_ROOT})
find_package(Torch QUIET)

# --- Eigen 3.4+ (header-only, for from-scratch MLP) ---
find_package(Eigen3 QUIET)

# ================================================================
# Regression Basics (STL only, no external deps)
# ================================================================

add_executable(linear_regression 00_linear_regression.cpp)

add_executable(logistic_regression 01_logistic_regression.cpp)

# ================================================================
# LibTorch Neuron Demo (requires LibTorch)
# ================================================================

if(Torch_FOUND)
    add_executable(neuron_demo 02_neuron_demo.cpp)
    target_link_libraries(neuron_demo ${TORCH_LIBRARIES})
    target_include_directories(neuron_demo PRIVATE ${TORCH_INCLUDE_DIRS})
else()
    message(WARNING "LibTorch not found — skipping neuron_demo target")
endif()

# ================================================================
# MLP from Scratch with Eigen
# ================================================================

if(Eigen3_FOUND)
    add_executable(mlp_eigen 03_mlp_eigen.cpp)
    target_include_directories(mlp_eigen PRIVATE ${EIGEN3_INCLUDE_DIR})

    add_executable(backprop_training 04_backprop_training.cpp)
    target_include_directories(backprop_training PRIVATE ${EIGEN3_INCLUDE_DIR})
else()
    message(WARNING "Eigen3 not found — skipping mlp_eigen and backprop_training targets")
endif()
