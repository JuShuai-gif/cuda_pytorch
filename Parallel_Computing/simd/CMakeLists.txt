cmake_minimum_required(VERSION 3.14)
project(simd_tutorial VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Common compile options
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -O2 -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2 -g")

# Common include directory -- all targets see the common headers
include_directories(${CMAKE_SOURCE_DIR}/common)

# Build options: enable/disable platform-specific targets
option(BUILD_ARM "Build ARM SIMD targets (NEON)" OFF)
option(BUILD_ARM_SVE "Build ARM SVE targets" OFF)
option(BUILD_X86 "Build x86 SIMD targets (SSE/AVX2)" ON)
option(BUILD_X86_AVX512 "Build x86 AVX-512 targets" OFF)

# Auto-detect host platform
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(BUILD_ARM ON CACHE BOOL "Building on ARM" FORCE)
    set(BUILD_X86 OFF CACHE BOOL "Not on x86" FORCE)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|i686")
    set(BUILD_X86 ON CACHE BOOL "Building on x86" FORCE)
endif()

# Add platform subdirectories
if(BUILD_ARM)
    if(EXISTS ${CMAKE_SOURCE_DIR}/arm/CMakeLists.txt)
        add_subdirectory(arm)
    else()
        message(STATUS "ARM subdirectory not found, skipping")
    endif()
endif()

if(BUILD_X86)
    if(EXISTS ${CMAKE_SOURCE_DIR}/x86/CMakeLists.txt)
        add_subdirectory(x86)
    else()
        message(STATUS "x86 subdirectory not found, skipping")
    endif()
endif()
