cmake_minimum_required(VERSION 3.14)
project(ArchBenchmarks LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(arch_benchmarks
    main.cpp
    cache_bench.cpp
    numa_bench.cpp
    simd_bench.cpp
    sys_info.cpp
)

target_compile_options(arch_benchmarks PRIVATE -Wall -Wextra -O2 -pthread)

# Enable AVX2 intrinsics
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(arch_benchmarks PRIVATE -mavx2 -mfma)
endif()

target_link_libraries(arch_benchmarks PRIVATE pthread)

# Optional: NUMA support
find_library(NUMA_LIB numa)
if(NUMA_LIB)
    target_link_libraries(arch_benchmarks PRIVATE ${NUMA_LIB})
    target_compile_definitions(arch_benchmarks PRIVATE HAS_LIBNUMA)
    message(STATUS "libnuma found - NUMA benchmarks enabled")
else()
    message(STATUS "libnuma not found - NUMA benchmarks disabled")
endif()

# For best benchmarking: -march=native -O3 -DNDEBUG
