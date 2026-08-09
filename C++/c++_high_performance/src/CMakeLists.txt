cmake_minimum_required(VERSION 3.16)

project(cpp_high_performance CXX)

# ---------------------------------------------------------------------------
# Project options
# ---------------------------------------------------------------------------
option(ENABLE_TESTS "Build correctness tests" ON)
option(ENABLE_BENCHMARKS "Build benchmarks" ON)
option(ENABLE_SANITIZERS "Build with AddressSanitizer/UBSan/LeakSanitizer" OFF)
option(ENABLE_THREAD_SANITIZER "Build with ThreadSanitizer" OFF)
option(ENABLE_CPP20_EXAMPLES "Build C++20 ranges examples" OFF)
option(ENABLE_PARALLEL_STL "Build Parallel STL examples" OFF)
option(ENABLE_BOOST "Enable Boost usage" OFF)
option(ENABLE_BOOST_COMPUTE "Enable Boost.Compute GPU examples" OFF)
option(ENABLE_OPENCL "Enable OpenCL examples" OFF)
option(ENABLE_NATIVE_OPTIMIZATION "Enable -march=native (host specific)" OFF)

# ---------------------------------------------------------------------------
# Language standard and common compiler flags
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wconversion)
    # Keep the frame pointer in all configurations so perf/gprof stacks are
    # reliable (perf works best with -fno-omit-frame-pointer).
    add_compile_options(-fno-omit-frame-pointer)
endif()

if(ENABLE_NATIVE_OPTIMIZATION AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-march=native)
endif()

# ---------------------------------------------------------------------------
# Sanitizers
# ---------------------------------------------------------------------------
if(ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-fsanitize=address,undefined)
    add_link_options(-fsanitize=address,undefined)
endif()

if(ENABLE_THREAD_SANITIZER AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # TSAN cannot be combined with ASAN.
    add_compile_options(-fsanitize=thread)
    add_link_options(-fsanitize=thread)
endif()

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# Parallel STL (std::execution) backend: libstdc++ delegates to oneTBB.
# ---------------------------------------------------------------------------
find_library(TBB_LIBRARY NAMES tbb HINTS /usr/lib/x86_64-linux-gnu)
if(TBB_LIBRARY)
    message(STATUS "Found TBB: ${TBB_LIBRARY}")
endif()

# ---------------------------------------------------------------------------
# Subdirectories. Chapters are added as they are implemented.
# ---------------------------------------------------------------------------
add_subdirectory(common)
add_subdirectory(chapter01_zero_cost)
add_subdirectory(chapter02_modern_cpp)
add_subdirectory(chapter03_measurement)
add_subdirectory(chapter04_data_structures)
add_subdirectory(chapter05_iterators)
add_subdirectory(chapter06_algorithms)
add_subdirectory(chapter07_memory)
add_subdirectory(chapter08_metaprogramming)
add_subdirectory(chapter09_lazy_evaluation)
add_subdirectory(chapter10_concurrency)
add_subdirectory(chapter11_parallel_stl)
