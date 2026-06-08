cmake_minimum_required(VERSION 3.16)
project(zhihu_hpc VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Compiler options
# ---------------------------------------------------------------------------
option(ENABLE_SSE2   "Build with SSE2 support"   ON)
option(ENABLE_SSE4_1 "Build with SSE4.1 support" ON)
option(ENABLE_AVX2    "Build with AVX2 support"   OFF)
option(ENABLE_AVX512  "Build with AVX512 support" OFF)
option(BUILD_TESTS    "Build test programs"       OFF)
option(BUILD_DEMOS    "Build demonstration programs" ON)

# Common compile options
add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)

# Platform-specific flags
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    set(HAS_X86 ON)
else()
    set(HAS_X86 OFF)
endif()

# ---------------------------------------------------------------------------
# Chapter 7 demo: C++ language construct efficiency
# ---------------------------------------------------------------------------
if(BUILD_DEMOS)
    add_executable(ch07_demo demos/ch07_demo.cpp)
    target_include_directories(ch07_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# ---------------------------------------------------------------------------
# Chapter 8 demo: Compiler optimizations
# ---------------------------------------------------------------------------
if(BUILD_DEMOS)
    add_executable(ch08_demo demos/ch08_demo.cpp)
    target_include_directories(ch08_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# ---------------------------------------------------------------------------
# Chapter 9 demo: Memory access optimization
# ---------------------------------------------------------------------------
if(BUILD_DEMOS)
    add_executable(ch09_demo demos/ch09_demo.cpp)
    target_include_directories(ch09_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# ---------------------------------------------------------------------------
# Chapter 11 demo: Out-of-order execution
# ---------------------------------------------------------------------------
if(BUILD_DEMOS)
    add_executable(ch11_demo demos/ch11_demo.cpp)
    target_include_directories(ch11_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# ---------------------------------------------------------------------------
# Chapter 12 demo: Vector operations (requires SSE2+)
# ---------------------------------------------------------------------------
if(BUILD_DEMOS AND HAS_X86)
    add_executable(ch12_demo demos/ch12_demo.cpp)
    target_include_directories(ch12_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(ch12_demo PRIVATE -msse2)
    if(ENABLE_SSE4_1)
        target_compile_options(ch12_demo PRIVATE -msse4.1)
    endif()
    if(ENABLE_AVX2)
        target_compile_options(ch12_demo PRIVATE -mavx2)
    endif()
endif()

# ---------------------------------------------------------------------------
# Chapter 13 demo: CPU dispatching
# ---------------------------------------------------------------------------
if(BUILD_DEMOS AND HAS_X86)
    add_executable(ch13_demo demos/ch13_demo.cpp)
    target_include_directories(ch13_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(ch13_demo PRIVATE -msse2)
endif()

# ---------------------------------------------------------------------------
# Chapter 14 demo: Specific optimization topics
# ---------------------------------------------------------------------------
if(BUILD_DEMOS)
    add_executable(ch14_demo demos/ch14_demo.cpp)
    target_include_directories(ch14_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# ---------------------------------------------------------------------------
# Chapter 15 demo: Metaprogramming
# ---------------------------------------------------------------------------
if(BUILD_DEMOS)
    add_executable(ch15_demo demos/ch15_demo.cpp)
    target_include_directories(ch15_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# ---------------------------------------------------------------------------
# Chapter 16 demo: Testing speed
# ---------------------------------------------------------------------------
if(BUILD_DEMOS AND HAS_X86)
    add_executable(ch16_demo demos/ch16_demo.cpp)
    target_include_directories(ch16_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# ---------------------------------------------------------------------------
# Print configuration summary
# ---------------------------------------------------------------------------
message(STATUS "zhihu_hpc configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  x86 Platform: ${HAS_X86}")
message(STATUS "  Build Demos:  ${BUILD_DEMOS}")
message(STATUS "  Build Tests:  ${BUILD_TESTS}")
if(HAS_X86)
    message(STATUS "  SSE2:         ${ENABLE_SSE2}")
    message(STATUS "  SSE4.1:       ${ENABLE_SSE4_1}")
    message(STATUS "  AVX2:         ${ENABLE_AVX2}")
    message(STATUS "  AVX512:       ${ENABLE_AVX512}")
endif()
