cmake_minimum_required(VERSION 3.12)
project(cpu_memory_experiments CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------
# Options
# ---------------------------------------------------------------
option(ENABLE_NATIVE_OPTIMIZATION "Use -march=native" OFF)
option(ENABLE_NUMA_EXAMPLES "Build NUMA examples (requires libnuma)" OFF)
option(ENABLE_AVX2_EXAMPLES "Build AVX2 examples" OFF)
option(ENABLE_AVX512_EXAMPLES "Build AVX-512 examples" OFF)

# ---------------------------------------------------------------
# Common flags
# ---------------------------------------------------------------
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -fno-omit-frame-pointer")

if(ENABLE_NATIVE_OPTIMIZATION)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-march=native" HAVE_MARCH_NATIVE)
  if(HAVE_MARCH_NATIVE)
    add_compile_options(-march=native)
  endif()
endif()

find_package(Threads REQUIRED)

# libnuma detection (optional)
find_library(NUMA_LIBRARY numa)
if(NUMA_LIBRARY)
  set(NUMA_FOUND TRUE)
else()
  set(NUMA_FOUND FALSE)
endif()

# ---------------------------------------------------------------
# Common static library
# ---------------------------------------------------------------
add_library(common
  common/benchmark.cpp
  common/cpu_info.cpp
  common/statistics.cpp
)
target_include_directories(common PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/common
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(common PUBLIC Threads::Threads)

# ---------------------------------------------------------------
# Helper: declare an experiment
#   add_experiment(name src...)
#   -> builds ${name} executable, Release benchmarks are the default target.
# ---------------------------------------------------------------
function(add_experiment name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE common)
endfunction()

# ---------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------
add_experiment(memory_latency
  01_memory_latency/memory_latency.cpp)

add_experiment(sequential_random_access
  02_sequential_random_access/sequential_random_access.cpp)

add_experiment(cache_line_size
  03_cache_line_size/cache_line_size.cpp)

add_experiment(stride_access
  04_stride_access/stride_access.cpp)

add_experiment(cache_capacity
  05_cache_capacity/cache_capacity.cpp)

add_experiment(cache_associativity
  06_cache_associativity/cache_associativity.cpp)

add_experiment(cache_conflict
  07_cache_conflict/cache_conflict.cpp)

add_experiment(write_back_behavior
  08_write_back_behavior/write_back_behavior.cpp)

add_experiment(matrix_traversal
  09_matrix_traversal/matrix_traversal.cpp)

add_experiment(cache_blocking
  10_cache_blocking/cache_blocking.cpp)

add_experiment(aos_soa
  11_aos_soa/aos_soa.cpp)

add_experiment(pointer_chasing
  12_pointer_chasing/pointer_chasing.cpp)

add_experiment(false_sharing
  13_false_sharing/false_sharing.cpp)

add_experiment(atomic_contention
  14_atomic_contention/atomic_contention.cpp)

add_experiment(thread_affinity
  15_thread_affinity/thread_affinity.cpp)

add_experiment(prefetch
  16_prefetch/prefetch.cpp)

add_experiment(non_temporal_store
  17_non_temporal_store/non_temporal_store.cpp)

add_experiment(tlb_capacity
  18_tlb_capacity/tlb_capacity.cpp)

add_experiment(page_size
  19_page_size/page_size.cpp)

add_experiment(huge_pages
  20_huge_pages/huge_pages.cpp)

add_experiment(page_fault
  21_page_fault/page_fault.cpp)

add_experiment(memory_mapping
  22_memory_mapping/memory_mapping.cpp)

add_experiment(memory_bandwidth
  23_memory_bandwidth/memory_bandwidth.cpp)

# NUMA experiments: build only when libnuma is available and requested.
if(ENABLE_NUMA_EXAMPLES AND NUMA_FOUND)
  add_executable(numa_local_remote 24_numa_local_remote/numa_local_remote.cpp)
  target_link_libraries(numa_local_remote PRIVATE common ${NUMA_LIBRARY})

  add_executable(numa_first_touch 25_numa_first_touch/numa_first_touch.cpp)
  target_link_libraries(numa_first_touch PRIVATE common ${NUMA_LIBRARY})

  add_executable(numa_replication 26_numa_replication/numa_replication.cpp)
  target_link_libraries(numa_replication PRIVATE common ${NUMA_LIBRARY})
elseif(ENABLE_NUMA_EXAMPLES AND NOT NUMA_FOUND)
  message(WARNING "ENABLE_NUMA_EXAMPLES requested but libnuma not found; "
                  "NUMA experiments will be skipped.")
endif()

add_experiment(instruction_cache
  27_instruction_cache/instruction_cache.cpp)

add_experiment(integrated_project
  28_integrated_project/integrated_project.cpp)

# ---------------------------------------------------------------
# Engineering pitfalls (experiment 29): real-world gotchas.
# Each pitfall is a separate small executable, all in one directory.
# ---------------------------------------------------------------
add_experiment(p1_debug_vs_release
  29_engineering_pitfalls/p1_debug_vs_release.cpp)

add_experiment(p2_dead_code_elimination
  29_engineering_pitfalls/p2_dead_code_elimination.cpp)

add_experiment(p3_vector_bool
  29_engineering_pitfalls/p3_vector_bool.cpp)

add_experiment(p4_shared_ptr_contention
  29_engineering_pitfalls/p4_shared_ptr_contention.cpp)

add_experiment(p5_alignment
  29_engineering_pitfalls/p5_alignment.cpp)

add_experiment(p6_benchmark_noise
  29_engineering_pitfalls/p6_benchmark_noise.cpp)

add_experiment(p7_atomic_memory_order
  29_engineering_pitfalls/p7_atomic_memory_order.cpp)

add_experiment(p8_volatile_not_atomic
  29_engineering_pitfalls/p8_volatile_not_atomic.cpp)

add_experiment(p9_hugepage_verify
  29_engineering_pitfalls/p9_hugepage_verify.cpp)
