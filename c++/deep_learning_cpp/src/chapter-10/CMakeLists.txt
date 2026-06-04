cmake_minimum_required(VERSION 3.22)
project(Chapter10_Deployment)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- LibTorch ---
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_ROOT})
find_package(Torch REQUIRED)

# --- ONNX Runtime (optional, for onnx_inference target) ---
if(EXISTS ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)
    add_library(onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
        INTERFACE_INCLUDE_DIRECTORIES ${ONNXRUNTIME_ROOT}/include
    )
    set(HAS_ONNX TRUE)
else()
    set(HAS_ONNX FALSE)
endif()

# ================================================================
# Model Export
# ================================================================

add_executable(torchscript_export 01_torchscript_export.cpp)
target_include_directories(torchscript_export PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(torchscript_export ${TORCH_LIBRARIES})

add_executable(parity_check 02_parity_check.cpp)
target_include_directories(parity_check PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(parity_check ${TORCH_LIBRARIES})

# ================================================================
# Micro-Batch Scheduling
# ================================================================

add_executable(micro_batcher 03_micro_batcher.cpp)
target_include_directories(micro_batcher PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(micro_batcher ${TORCH_LIBRARIES})

# ================================================================
# Inference Optimization
# ================================================================

add_executable(optimization_benchmark 04_optimization_benchmark.cpp)
target_include_directories(optimization_benchmark PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(optimization_benchmark ${TORCH_LIBRARIES})

# ONNX Runtime inference (optional)
add_executable(onnx_inference 05_onnx_inference.cpp)
target_include_directories(onnx_inference PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(onnx_inference ${TORCH_LIBRARIES})
if(HAS_ONNX)
    target_include_directories(onnx_inference PRIVATE ${ONNXRUNTIME_ROOT}/include)
    target_link_libraries(onnx_inference onnxruntime)
    target_compile_definitions(onnx_inference PRIVATE HAS_ONNX)
endif()

# ================================================================
# Model Serving
# ================================================================

add_executable(model_serving 06_model_serving.cpp)
target_include_directories(model_serving PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(model_serving ${TORCH_LIBRARIES})

# ================================================================
# Operations & Evolution (Pure STL, no LibTorch needed)
# ================================================================

add_executable(drift_detection 07_drift_detection.cpp)

add_executable(safe_release 08_safe_release.cpp)
