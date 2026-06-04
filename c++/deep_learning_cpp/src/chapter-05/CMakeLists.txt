cmake_minimum_required(VERSION 3.22)
project(Chapter05_MultilayerPerceptron LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- LibTorch ---
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_ROOT})
find_package(Torch QUIET)

# --- Eigen 3.4+ (header-only) ---
find_package(Eigen3 QUIET)

# ================================================================
# Eigen MLP: Layer + MultilayerPerceptron + XOR training
# ================================================================

if(Eigen3_FOUND)
    add_executable(eigen_mlp_xor 00_eigen_mlp_xor.cpp)
    target_include_directories(eigen_mlp_xor PRIVATE ${EIGEN3_INCLUDE_DIR})
else()
    message(WARNING "Eigen3 not found -- skipping eigen_mlp_xor target")
endif()

# ================================================================
# Activation Functions: 17 activations from scratch (STL only)
# ================================================================

add_executable(activation_functions 02_activation_functions.cpp)

# ================================================================
# LibTorch MLP: torch::nn::Linear + SGD training loop
# ================================================================

if(Torch_FOUND)
    add_executable(libtorch_mlp 01_libtorch_mlp.cpp)
    target_link_libraries(libtorch_mlp ${TORCH_LIBRARIES})
    target_include_directories(libtorch_mlp PRIVATE ${TORCH_INCLUDE_DIRS})
else()
    message(WARNING "LibTorch not found -- skipping libtorch_mlp target")
endif()

# ================================================================
# Optimizers: GD / RMSprop / Momentum / Adam / AdaGrad / AdaDelta
# ================================================================

if(Torch_FOUND)
    add_executable(optimizers 03_optimizers_libtorch.cpp)
    target_link_libraries(optimizers ${TORCH_LIBRARIES})
    target_include_directories(optimizers PRIVATE ${TORCH_INCLUDE_DIRS})
else()
    message(WARNING "LibTorch not found -- skipping optimizers target")
endif()
