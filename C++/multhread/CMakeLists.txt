cmake_minimum_required(VERSION 3.14)
project(cpp_concurrency_learning VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Thread support
find_package(Threads REQUIRED)

enable_testing()

# Global include directories
include_directories(${CMAKE_SOURCE_DIR})

# Chapter 01: Hello, C++ Concurrency World
add_subdirectory(src/chapter-01)

# Chapter 02: Thread Management
add_subdirectory(src/chapter-02)

# Chapter 03: Sharing Data Between Threads
add_subdirectory(src/chapter-03)

# Chapter 04: Synchronizing Concurrent Operations
add_subdirectory(src/chapter-04)

# Chapter 05: C++ Memory Model and Atomic Operations
add_subdirectory(src/chapter-05)

# Chapter 06: Lock-based Concurrent Data Structures
add_subdirectory(src/chapter-06)

# Chapter 07: Lock-free Concurrent Data Structures
add_subdirectory(src/chapter-07)

# Chapter 08: Concurrent Code Design
add_subdirectory(src/chapter-08)

# Chapter 09: Advanced Thread Management
add_subdirectory(src/chapter-09)

# Chapter 10: Parallel Algorithms (C++17)
add_subdirectory(src/chapter-10)

# Chapter 11: Testing and Debugging
add_subdirectory(src/chapter-11)

# Chapter 12: Modern C++20 Concurrency Features
add_subdirectory(src/chapter-12)

# Chapter 13: Memory Model and Cache Optimization (Advanced)
add_subdirectory(src/chapter-13)

# Chapter 14: Advanced Parallel Algorithms
add_subdirectory(src/chapter-14)

# Chapter 15: Performance Analysis and Optimization
add_subdirectory(src/chapter-15)

# Chapter 16: High Performance Computing (HPC)
add_subdirectory(src/chapter-16)

# Chapter 17: Engineering Practices
add_subdirectory(src/chapter-17)

# Industrial Concurrency Project
add_subdirectory(project/industrial-concurrency-project)
