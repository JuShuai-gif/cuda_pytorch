cmake_minimum_required(VERSION 3.20)

project(multhread
    VERSION 1.0.0
    DESCRIPTION "From C++ Templates to CUTLASS to AI Inference Engines"
    LANGUAGES CXX
)

# ============================================================
# Compiler settings
# ============================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================
# Options
# ============================================================
option(ENABLE_CUDA "Enable CUDA support" OFF)
option(BUILD_TESTS "Build tests" OFF)

# ============================================================
# Top-level include directories
# ============================================================
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/cutlass_style)
include_directories(${PROJECT_SOURCE_DIR}/cutlass_style/include)
include_directories(${PROJECT_SOURCE_DIR}/project/mini_inference_engine)
include_directories(${PROJECT_SOURCE_DIR}/project/mini_inference_engine/include)

# ============================================================
# Common compile flags
# ============================================================
set(COMMON_CXX_FLAGS
    -Wall
    -Wextra
    -Wpedantic
    -O2
)

# Suppress some noisy-but-harmless warnings when using heavy templates
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    list(APPEND COMMON_CXX_FLAGS
        -Wno-unused-parameter
        -Wno-unused-variable
        -Wno-unused-but-set-variable
    )
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    list(APPEND COMMON_CXX_FLAGS
        -Wno-unused-parameter
        -Wno-unused-variable
    )
endif()

# ============================================================
# Output directory
# ============================================================
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

# ============================================================
# Helper function: add a target from a single source file
# ============================================================
function(add_chapter_example TARGET_NAME SOURCE_FILE)
    add_executable(${TARGET_NAME} ${SOURCE_FILE})
    target_compile_options(${TARGET_NAME} PRIVATE ${COMMON_CXX_FLAGS})
    target_include_directories(${TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR})
    target_include_directories(${TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/cutlass_style/include)
endfunction()

# ============================================================
# Chapter examples (src/chapter-XX/*.cpp)
# ============================================================
file(GLOB CHAPTER_DIRS LIST_DIRECTORIES true
    ${PROJECT_SOURCE_DIR}/src/chapter-*
)

foreach(CHAPTER_DIR ${CHAPTER_DIRS})
    if(IS_DIRECTORY ${CHAPTER_DIR})
        file(GLOB CHAPTER_SOURCES ${CHAPTER_DIR}/*.cpp)
        get_filename_component(CHAPTER_NAME ${CHAPTER_DIR} NAME)
        foreach(SOURCE_FILE ${CHAPTER_SOURCES})
            get_filename_component(EXE_NAME ${SOURCE_FILE} NAME_WE)
            add_chapter_example("${CHAPTER_NAME}_${EXE_NAME}" ${SOURCE_FILE})
        endforeach()
    endif()
endforeach()

# ============================================================
# CUTLASS-style examples
# ============================================================
file(GLOB CUTLASS_EXAMPLES
    ${PROJECT_SOURCE_DIR}/cutlass_style/examples/*.cpp
)

foreach(EXAMPLE_SRC ${CUTLASS_EXAMPLES})
    get_filename_component(EXAMPLE_NAME ${EXAMPLE_SRC} NAME_WE)
    add_chapter_example(${EXAMPLE_NAME} ${EXAMPLE_SRC})
endforeach()

# ============================================================
# Mini Inference Engine
# ============================================================
set(INFERENCE_DIR ${PROJECT_SOURCE_DIR}/project/mini_inference_engine)

file(GLOB_RECURSE INFERENCE_SOURCES
    ${INFERENCE_DIR}/src/*.cpp
)

file(GLOB_RECURSE INFERENCE_EXAMPLES
    ${INFERENCE_DIR}/examples/*.cpp
)

# Build inference engine library
add_library(mini_inference_engine STATIC ${INFERENCE_SOURCES})
target_include_directories(mini_inference_engine
    PUBLIC
        ${INFERENCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/cutlass_style/include
        ${PROJECT_SOURCE_DIR}
)
target_compile_options(mini_inference_engine PRIVATE ${COMMON_CXX_FLAGS})

# Build inference engine examples
foreach(EXAMPLE_SRC ${INFERENCE_EXAMPLES})
    get_filename_component(EXAMPLE_NAME ${EXAMPLE_SRC} NAME_WE)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_SRC})
    target_link_libraries(${EXAMPLE_NAME} PRIVATE mini_inference_engine)
    target_include_directories(${EXAMPLE_NAME}
        PRIVATE
            ${INFERENCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/cutlass_style/include
            ${PROJECT_SOURCE_DIR}
    )
    target_compile_options(${EXAMPLE_NAME} PRIVATE ${COMMON_CXX_FLAGS})
endforeach()

# ============================================================
# CUDA support (optional)
# ============================================================
if(ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # CUDA compile flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2 -arch=sm_80")

    message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA include: ${CUDAToolkit_INCLUDE_DIRS}")

    # --- CUTLASS-style CUDA examples ---
    file(GLOB CUTLASS_CUDA_EXAMPLES
        ${PROJECT_SOURCE_DIR}/cutlass_style/examples/*.cu
    )

    foreach(CUDA_SRC ${CUTLASS_CUDA_EXAMPLES})
        get_filename_component(CUDA_EXE_NAME ${CUDA_SRC} NAME_WE)
        add_executable(${CUDA_EXE_NAME}_cuda ${CUDA_SRC})
        target_include_directories(${CUDA_EXE_NAME}_cuda
            PRIVATE
                ${PROJECT_SOURCE_DIR}/cutlass_style/include
                ${PROJECT_SOURCE_DIR}
                ${CUDAToolkit_INCLUDE_DIRS}
        )
        target_link_libraries(${CUDA_EXE_NAME}_cuda
            PRIVATE
                ${CUDAToolkit_LIBRARY_DIR}
                cudart
        )
        target_compile_options(${CUDA_EXE_NAME}_cuda PRIVATE ${COMMON_CXX_FLAGS})
    endforeach()

    # --- CUDA inference engine ---
    file(GLOB_RECURSE INFERENCE_CUDA_SOURCES
        ${INFERENCE_DIR}/kernels/*.cu
    )

    if(INFERENCE_CUDA_SOURCES)
        add_library(mini_inference_engine_cuda STATIC ${INFERENCE_CUDA_SOURCES})
        target_include_directories(mini_inference_engine_cuda
            PUBLIC
                ${INFERENCE_DIR}/include
                ${PROJECT_SOURCE_DIR}/cutlass_style/include
                ${PROJECT_SOURCE_DIR}
                ${CUDAToolkit_INCLUDE_DIRS}
        )
        target_link_libraries(mini_inference_engine_cuda
            PUBLIC
                ${CUDAToolkit_LIBRARY_DIR}
                cudart
        )
    endif()
endif()

# ============================================================
# Status summary
# ============================================================
message(STATUS "============================================")
message(STATUS "multhread build configuration")
message(STATUS "  C++ Standard: 20")
message(STATUS "  CUDA Enabled: ${ENABLE_CUDA}")
message(STATUS "  Build Type:   ${CMAKE_BUILD_TYPE}")
message(STATUS "  Output Dir:   ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "============================================")
