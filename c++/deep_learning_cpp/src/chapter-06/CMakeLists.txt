cmake_minimum_required(VERSION 3.22)
project(Chapter06_ConvolutionalNeuralNetworks LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- LibTorch ---
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_ROOT})
find_package(Torch QUIET)

# --- Eigen 3.4+ ---
find_package(Eigen3 QUIET)

# ================================================================
# CPU Convolution: Nested-loop implementation (STL only)
# ================================================================

add_executable(convolution_cpu 00_convolution_cpu.cpp)

# ================================================================
# Eigen Convolution: im2col + matrix multiplication
# ================================================================

if(Eigen3_FOUND)
    add_executable(convolution_eigen 01_convolution_eigen.cpp)
    target_include_directories(convolution_eigen PRIVATE ${EIGEN3_INCLUDE_DIR})
else()
    message(WARNING "Eigen3 not found -- skipping convolution_eigen target")
endif()

# ================================================================
# LibTorch Convolution: Conv2d + ReLU demo
# ================================================================

if(Torch_FOUND)
    add_executable(convolution_libtorch 02_convolution_libtorch.cpp)
    target_link_libraries(convolution_libtorch ${TORCH_LIBRARIES})
    target_include_directories(convolution_libtorch PRIVATE ${TORCH_INCLUDE_DIRS})

    add_executable(vgg16_mnist 03_vgg16_mnist.cpp)
    target_link_libraries(vgg16_mnist ${TORCH_LIBRARIES})
    target_include_directories(vgg16_mnist PRIVATE ${TORCH_INCLUDE_DIRS})

    add_executable(image_augmentation 04_image_augmentation.cpp)
    target_link_libraries(image_augmentation ${TORCH_LIBRARIES})
    target_include_directories(image_augmentation PRIVATE ${TORCH_INCLUDE_DIRS})

    add_executable(cnn_terminology 05_cnn_terminology.cpp)
    target_link_libraries(cnn_terminology ${TORCH_LIBRARIES})
    target_include_directories(cnn_terminology PRIVATE ${TORCH_INCLUDE_DIRS})
else()
    message(WARNING "LibTorch not found -- skipping convolution_libtorch, vgg16_mnist, image_augmentation, cnn_terminology targets")
endif()
