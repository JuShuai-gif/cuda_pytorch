cmake_minimum_required(VERSION 3.16)
project(optimizing_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Common warnings across all targets (PDF p174 recommended flags).
add_compile_options(-Wall -Wextra -Wpedantic)

# Per-build-type optimization settings.
# Release keeps -g and -fno-omit-frame-pointer so perf can symbolicate
# hot spots (PDF ch.16 p169; perf needs frame pointers).
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -fno-omit-frame-pointer")

# Optional: -march=native. OFF by default to keep binaries portable.
# Enable with -DUSE_MARCH_NATIVE=ON (never enables AVX-512 implicitly
# beyond what the build CPU supports).
option(USE_MARCH_NATIVE "Compile with -march=native" OFF)
if(USE_MARCH_NATIVE)
  add_compile_options(-march=native)
endif()

# Optional auto-vectorization diagnostics (GCC). Add with
# -DENABLE_VECTORIZATION_REPORT=ON to see which loops were vectorized.
option(ENABLE_VECTORIZATION_REPORT "GCC -fopt-info-vec reports" OFF)
if(ENABLE_VECTORIZATION_REPORT)
  add_compile_options(-fopt-info-vec -fopt-info-vec-missed)
endif()

add_subdirectory(common)
add_subdirectory(01_profiling)
add_subdirectory(02_integer_float)
add_subdirectory(03_branch)
add_subdirectory(04_loop)
add_subdirectory(05_function)
add_subdirectory(06_class_virtual)
add_subdirectory(07_container)
add_subdirectory(08_memory_cache)
add_subdirectory(09_alignment)
add_subdirectory(10_multithreading)
add_subdirectory(11_false_sharing)
add_subdirectory(12_auto_vectorization)
add_subdirectory(13_intrinsics)
add_subdirectory(14_cpu_dispatch)
add_subdirectory(15_lookup_table)
add_subdirectory(16_division_optimization)
add_subdirectory(17_template_metaprogramming)
add_subdirectory(18_benchmark)
