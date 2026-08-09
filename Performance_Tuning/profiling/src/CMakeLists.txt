cmake_minimum_required(VERSION 3.18)
project(profiling_lab LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(Threads REQUIRED)
find_package(OpenMP QUIET)

set(CPP_DEMOS
  01_cpu_hotspot 02_cache_miss 03_branch_miss 04_memory_bandwidth
  05_false_sharing 06_lock_contention 07_allocation_hotspot
  08_memory_leak 09_use_after_free 10_syscall_overhead 11_io_bottleneck
  12_context_switch 13_simd 14_openmp_scaling
  20_cpu_hotspot_bad_good 21_cache_locality_bad_good 22_tlb_pagefault
  23_stream_bad_good 24_aos_soa 25_allocation_bad_good
  26_lock_contention_bad_good 27_shared_mutex 28_thread_imbalance
  29_syscall_bad_good 30_realtime_jitter 31_memory_fragmentation
  32_heap_buffer_overflow 34_v4l2_capture_benchmark)
foreach(name IN LISTS CPP_DEMOS)
  add_executable(${name} cpp/${name}.cpp)
  target_link_libraries(${name} PRIVATE Threads::Threads)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/common)
endforeach()

find_path(NUMA_INCLUDE_DIR numa.h)
find_library(NUMA_LIBRARY numa)
if(NUMA_INCLUDE_DIR AND NUMA_LIBRARY)
  add_executable(33_numa_local_remote cpp/33_numa_local_remote.cpp)
  target_include_directories(33_numa_local_remote PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/common ${NUMA_INCLUDE_DIR})
  target_link_libraries(33_numa_local_remote PRIVATE Threads::Threads ${NUMA_LIBRARY})
else()
  message(STATUS "libnuma开发文件未找到：跳过33_numa_local_remote")
endif()

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(FFMPEG QUIET IMPORTED_TARGET libavformat libavcodec libavutil libswscale)
endif()
if(TARGET PkgConfig::FFMPEG)
  add_executable(35_ffmpeg_decode_benchmark cpp/35_ffmpeg_decode_benchmark.cpp)
  target_include_directories(35_ffmpeg_decode_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/common)
  target_link_libraries(35_ffmpeg_decode_benchmark PRIVATE PkgConfig::FFMPEG)
else()
  message(STATUS "FFmpeg开发库未找到：跳过35_ffmpeg_decode_benchmark；CLI脚本仍可用")
endif()

if(OpenMP_CXX_FOUND)
  target_link_libraries(14_openmp_scaling PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(14_openmp_scaling PRIVATE HAVE_OPENMP=1)
  target_link_libraries(23_stream_bad_good PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(23_stream_bad_good PRIVATE HAVE_OPENMP=1)
else()
  message(STATUS "OpenMP 未找到：14_openmp_scaling 使用串行回退")
endif()

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  find_package(CUDAToolkit QUIET)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CUDA_DEMOS 01_vector_add 02_memory_bound 03_compute_bound
    04_uncoalesced_access 05_coalesced_access 06_branch_divergence
    07_shared_memory_bank_conflict 08_low_occupancy 09_kernel_launch_overhead
    10_h2d_d2h_transfer 11_cpu_gpu_sync 12_nvtx_pipeline 13_kernel_hotspot_bad_good
    14_pipeline_overlap_bad_good)
  foreach(name IN LISTS CUDA_DEMOS)
    add_executable(cuda_${name} cuda/${name}.cu)
    set_target_properties(cuda_${name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  endforeach()
  if(TARGET CUDA::nvToolsExt)
    target_link_libraries(cuda_12_nvtx_pipeline PRIVATE CUDA::nvToolsExt)
    target_compile_definitions(cuda_12_nvtx_pipeline PRIVATE HAVE_NVTX=1)
    target_link_libraries(cuda_13_kernel_hotspot_bad_good PRIVATE CUDA::nvToolsExt)
    target_compile_definitions(cuda_13_kernel_hotspot_bad_good PRIVATE HAVE_NVTX=1)
  endif()
else()
  message(STATUS "CUDA 编译器未找到：跳过 CUDA 示例，CPU 示例不受影响")
endif()

