cmake_minimum_required(VERSION 3.22)
project(Chapter09_Transformers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- LibTorch ---
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_ROOT})
find_package(Torch QUIET)

if(NOT Torch_FOUND)
    message(WARNING "LibTorch not found -- skipping Chapter09 Transformer targets")
    return()
endif()

# ================================================================
# Attention Mechanism
# ================================================================

add_executable(self_attention 01_self_attention.cpp)
target_include_directories(self_attention PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(self_attention ${TORCH_LIBRARIES})

add_executable(multi_head_attention 02_multi_head_attention.cpp)
target_include_directories(multi_head_attention PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(multi_head_attention ${TORCH_LIBRARIES})

# ================================================================
# Positional Encoding
# ================================================================

add_executable(positional_encoding 03_positional_encoding.cpp)
target_include_directories(positional_encoding PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(positional_encoding ${TORCH_LIBRARIES})

# ================================================================
# Transformer Architecture
# ================================================================

add_executable(transformer_encoder 04_transformer_encoder.cpp)
target_include_directories(transformer_encoder PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(transformer_encoder ${TORCH_LIBRARIES})

add_executable(transformer_decoder 05_transformer_decoder.cpp)
target_include_directories(transformer_decoder PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(transformer_decoder ${TORCH_LIBRARIES})

# ================================================================
# Model Compression
# ================================================================

add_executable(knowledge_distillation 06_knowledge_distillation.cpp)
target_include_directories(knowledge_distillation PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(knowledge_distillation ${TORCH_LIBRARIES})

add_executable(pruning 07_pruning.cpp)
target_include_directories(pruning PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(pruning ${TORCH_LIBRARIES})

add_executable(quantization 08_quantization.cpp)
target_include_directories(quantization PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(quantization ${TORCH_LIBRARIES})
