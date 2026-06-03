cmake_minimum_required(VERSION 3.14)
project(IndustrialConcurrency VERSION 1.0.0 LANGUAGES CXX)

# Chapter 10.2: Build configuration with multiple toolchain support.
# Supports regular Debug/Release as well as TSan (ThreadSanitizer) builds.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Chapter 10.2.3: Compiler warnings for concurrency bug detection.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Clang-specific: thread safety annotations
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wthread-safety)
endif()

# Required: Threads library (Ch2.1: std::thread depends on pthread on Linux).
find_package(Threads REQUIRED)

# Required: {fmt} library for portable string formatting (used when <format> unavailable).
find_package(fmt REQUIRED)

# ============================================================================
# Core Library (header-only interface library)
# ============================================================================
add_library(task_scheduler_lib INTERFACE)
target_include_directories(task_scheduler_lib INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(task_scheduler_lib INTERFACE Threads::Threads fmt::fmt)

# ============================================================================
# Source files for the library (non-template implementations)
# ============================================================================
set(LIBRARY_SOURCES
    src/thread_pool.cpp
    src/task_scheduler.cpp
    src/logger.cpp
)

# Optional: compile library sources as a static library for faster builds.
add_library(task_scheduler_static STATIC ${LIBRARY_SOURCES})
target_include_directories(task_scheduler_static PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(task_scheduler_static PUBLIC Threads::Threads fmt::fmt)

# ============================================================================
# Main Entry Point
# ============================================================================
add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE task_scheduler_static)

# ============================================================================
# Tests (each independently compilable)
# ============================================================================
add_executable(test_thread_pool tests/test_thread_pool.cpp)
target_link_libraries(test_thread_pool PRIVATE task_scheduler_static)

add_executable(test_task_queue tests/test_task_queue.cpp)
target_link_libraries(test_task_queue PRIVATE task_scheduler_static)

add_executable(test_task_scheduler tests/test_task_scheduler.cpp)
target_link_libraries(test_task_scheduler PRIVATE task_scheduler_static)

add_executable(test_stress tests/test_stress.cpp)
target_link_libraries(test_stress PRIVATE task_scheduler_static)

# ============================================================================
# Examples (each independently compilable)
# ============================================================================
add_executable(example_basic examples/example_basic.cpp)
target_link_libraries(example_basic PRIVATE task_scheduler_static)

add_executable(example_pipeline examples/example_pipeline.cpp)
target_link_libraries(example_pipeline PRIVATE task_scheduler_static)

add_executable(example_inference examples/example_inference.cpp)
target_link_libraries(example_inference PRIVATE task_scheduler_static)

add_executable(example_producer_consumer examples/example_producer_consumer.cpp)
target_link_libraries(example_producer_consumer PRIVATE task_scheduler_static)

# ============================================================================
# Custom Build Types
# ============================================================================

# Chapter 10.2.4: ThreadSanitizer build type for data race detection.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_C_FLAGS_TSAN
        "-fsanitize=thread -fno-omit-frame-pointer -g -O1"
        CACHE STRING "Flags for ThreadSanitizer build" FORCE)
    set(CMAKE_CXX_FLAGS_TSAN
        "-fsanitize=thread -fno-omit-frame-pointer -g -O1"
        CACHE STRING "Flags for ThreadSanitizer build" FORCE)
    set(CMAKE_EXE_LINKER_FLAGS_TSAN
        "-fsanitize=thread"
        CACHE STRING "Linker flags for ThreadSanitizer build" FORCE)
endif()

# ============================================================================
# Custom Targets
# ============================================================================

# Convenience target to build all tests
add_custom_target(tests DEPENDS
    test_thread_pool test_task_queue test_task_scheduler test_stress)

# Convenience target to build all examples
add_custom_target(examples DEPENDS
    example_basic example_pipeline example_inference example_producer_consumer)

# Run all tests via CTest
enable_testing()
add_test(NAME ThreadPool     COMMAND test_thread_pool)
add_test(NAME TaskQueue      COMMAND test_task_queue)
add_test(NAME TaskScheduler  COMMAND test_task_scheduler)
add_test(NAME Stress         COMMAND test_stress)

# ============================================================================
# Installation
# ============================================================================
install(TARGETS task_scheduler_static
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib)
install(TARGETS task_scheduler_lib EXPORT task_schedulerTargets)
install(DIRECTORY include/ DESTINATION include)
