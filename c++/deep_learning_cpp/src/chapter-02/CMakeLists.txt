cmake_minimum_required(VERSION 3.22)
project(Chapter02_DataPreprocessing)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- LibTorch ---
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_ROOT})
find_package(Torch REQUIRED)

# --- Eigen 3.4+ (header-only) ---
find_package(Eigen3 REQUIRED)

# --- Armadillo 12+ (for library-powered preprocessing demo) ---
find_package(Armadillo QUIET)

# --- OpenCV 4.x (optional, for image_resol and video_frame) ---
find_package(OpenCV QUIET)

# --- ONNX Runtime (optional, for contextual_embedding) ---
if(EXISTS ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)
    add_library(onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
        INTERFACE_INCLUDE_DIRECTORIES ${ONNXRUNTIME_ROOT}/include
    )
    set(ONNXRUNTIME_AVAILABLE TRUE)
else()
    set(ONNXRUNTIME_AVAILABLE FALSE)
endif()

# ================================================================
# Structured Data: Missing Values, Encoding, Scaling
# ================================================================

add_executable(handling_missing_values 01_handling_missing_values.cpp)

add_executable(Encoding_categorical_features 02_Encoding_categorical_features.cpp)

add_executable(feature_scaling 03_feature_scaling_and_standardization.cpp)

# ================================================================
# Structured Data: Dimensionality Reduction
# ================================================================

add_executable(Dimensionality_reduction 04_Dimensionality_reduction.cpp)
target_include_directories(Dimensionality_reduction PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(Dimensionality_reduction ${TORCH_LIBRARIES})

# ================================================================
# Structured Data: Time-Series & Feature Engineering
# ================================================================

add_executable(time_series_engineering 05_time_series_engineering.cpp)

add_executable(feature_interaction 06_feature_interation_engineering.cpp)

# ================================================================
# Text Preprocessing
# ================================================================

add_executable(embedding_vectorization 10_embedding_based_vectorization.cpp)

add_executable(tokenization 09_tokenization_stop_word_removal_stemming.cpp)

add_executable(sequence_padding 12_sequence_padding_and_truncation.cpp)

# ================================================================
# Image Preprocessing (requires OpenCV)
# ================================================================

if(OpenCV_FOUND)
    add_executable(image_resol 13_image_resol.cpp)
    target_link_libraries(image_resol ${OpenCV_LIBS})
    target_include_directories(image_resol PRIVATE ${OpenCV_INCLUDE_DIRS})
else()
    message(WARNING "OpenCV not found — skipping image_resol target")
endif()

# ================================================================
# Audio Preprocessing (standalone DFT, no external deps)
# ================================================================

add_executable(audio_video 15_audio_video.cpp)

# ================================================================
# LibTorch Examples: Autoencoder, Custom Dataset
# ================================================================

add_executable(libtorch_ae 08_libtorch.cpp)
target_link_libraries(libtorch_ae ${TORCH_LIBRARIES})
target_include_directories(libtorch_ae PRIVATE ${TORCH_INCLUDE_DIRS})

add_executable(custom_dataset 17_custom_dataset_demo.cpp)
target_link_libraries(custom_dataset ${TORCH_LIBRARIES})
target_include_directories(custom_dataset PRIVATE ${TORCH_INCLUDE_DIRS})

# ================================================================
# Contextual Embedding (ONNX Runtime - optional)
# ================================================================

if(ONNXRUNTIME_AVAILABLE)
    add_executable(contextual_embedding 11_contextual_embedding.cpp)
    target_link_libraries(contextual_embedding onnxruntime)
else()
    # Compile without linking; the code has #if USE_REAL_MODEL guard
    add_executable(contextual_embedding 11_contextual_embedding.cpp)
endif()

# ================================================================
# Advanced Techniques
# ================================================================

add_executable(advanced_tech 16_advanced_tech.cpp)

# ================================================================
# Memory Mapping (Linux-specific: mmap)
# ================================================================

add_executable(memory_mapping 18_memory_mapping.cpp)

# ================================================================
# Library-Powered Preprocessing (Armadillo, PDF p51-56)
# ================================================================

if(Armadillo_FOUND)
    add_executable(preprocessing_with_libraries
        07_preprocessing_with_libraries.cpp)
    target_link_libraries(preprocessing_with_libraries
        ${ARMADILLO_LIBRARIES})
else()
    message(WARNING
        "Armadillo not found — skipping preprocessing_with_libraries")
endif()

# ================================================================
# Video Frame Extraction (OpenCV, PDF p67)
# ================================================================

if(OpenCV_FOUND)
    add_executable(video_frame_extraction
        14_video_frame_extraction.cpp)
    target_link_libraries(video_frame_extraction ${OpenCV_LIBS})
    target_include_directories(video_frame_extraction PRIVATE
        ${OpenCV_INCLUDE_DIRS})
else()
    message(WARNING
        "OpenCV not found — skipping video_frame_extraction")
endif()

# ================================================================
# Large-Scale Data Management (PDF p73-74)
# ================================================================

add_executable(large_scale_data 19_large_scale_data.cpp)
