cmake_minimum_required(VERSION 3.16)

project(chp_projects CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wconversion)
    add_compile_options(-fno-omit-frame-pointer)
endif()

option(ENABLE_PROJECT_TESTS "Build project tests" ON)
option(ENABLE_PROJECT_BENCHMARKS "Build project benchmarks" ON)

find_package(Threads REQUIRED)

find_library(TBB_LIBRARY NAMES tbb HINTS /usr/lib/x86_64-linux-gnu)
if(TBB_LIBRARY)
    message(STATUS "Found TBB: ${TBB_LIBRARY}")
endif()

add_subdirectory(object_pool)
add_subdirectory(task_system)
add_subdirectory(parallel_pipeline)
add_subdirectory(high_performance_container)
