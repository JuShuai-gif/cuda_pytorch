cmake_minimum_required(VERSION 3.22)
project(Chapter08_GenerativeNetworks LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- LibTorch ---
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_ROOT})
find_package(Torch QUIET)

# ================================================================
# CNN Autoencoder + VAE + GAN (require LibTorch)
# ================================================================

if(Torch_FOUND)
    add_executable(autoencoder 00_autoencoder.cpp)
    target_link_libraries(autoencoder ${TORCH_LIBRARIES})
    target_include_directories(autoencoder PRIVATE ${TORCH_INCLUDE_DIRS})

    add_executable(vae 01_vae.cpp)
    target_link_libraries(vae ${TORCH_LIBRARIES})
    target_include_directories(vae PRIVATE ${TORCH_INCLUDE_DIRS})

    add_executable(gan 02_gan.cpp)
    target_link_libraries(gan ${TORCH_LIBRARIES})
    target_include_directories(gan PRIVATE ${TORCH_INCLUDE_DIRS})
else()
    message(WARNING "LibTorch not found -- skipping autoencoder, vae, gan targets")
endif()

# ================================================================
# Sampling Strategies + Evaluation Metrics (pure STL, always builds)
# ================================================================

add_executable(sampling_strategies 03_sampling_strategies.cpp)

add_executable(evaluation_metrics 04_evaluation_metrics.cpp)
