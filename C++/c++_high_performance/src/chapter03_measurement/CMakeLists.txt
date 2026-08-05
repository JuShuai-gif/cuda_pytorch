# Chapter 3: Measuring Performance

# --- scoped_timer: instrumentation profiler style timer ---
add_executable(ch03_scoped_timer_example scoped_timer/example.cpp)

# --- complexity_growth: linear vs binary search across sizes ---
if(ENABLE_BENCHMARKS)
    add_executable(ch03_complexity_benchmark
        complexity_growth/benchmark.cpp
        complexity_growth/baseline.cpp
    )
    target_link_libraries(ch03_complexity_benchmark PRIVATE chp_common)
endif()
if(ENABLE_TESTS)
    add_executable(ch03_complexity_tests
        complexity_growth/tests.cpp
        complexity_growth/baseline.cpp
    )
    target_link_libraries(ch03_complexity_tests PRIVATE chp_common)
endif()

# --- vector_growth: amortized push_back / reserve ---
add_executable(ch03_vector_growth_example vector_growth/example.cpp)
if(ENABLE_BENCHMARKS)
    add_executable(ch03_vector_growth_benchmark vector_growth/benchmark.cpp)
    target_link_libraries(ch03_vector_growth_benchmark PRIVATE chp_common)
endif()
if(ENABLE_TESTS)
    add_executable(ch03_vector_growth_tests vector_growth/tests.cpp)
    target_link_libraries(ch03_vector_growth_tests PRIVATE chp_common)
endif()

# --- latency_throughput: serial chain vs independent batch ---
if(ENABLE_BENCHMARKS)
    add_executable(ch03_latency_throughput_benchmark
                   latency_throughput/benchmark.cpp)
    target_link_libraries(ch03_latency_throughput_benchmark PRIVATE chp_common)
endif()

# --- cpu_vs_memory_bound ---
if(ENABLE_BENCHMARKS)
    add_executable(ch03_cpu_mem_benchmark cpu_vs_memory_bound/benchmark.cpp)
    target_link_libraries(ch03_cpu_mem_benchmark PRIVATE chp_common)
endif()

# --- release_vs_debug: same source, different build type ---
add_executable(ch03_release_debug_benchmark release_vs_debug/benchmark.cpp)

# --- warmup_effect ---
if(ENABLE_BENCHMARKS)
    add_executable(ch03_warmup_benchmark warmup_effect/benchmark.cpp)
    target_link_libraries(ch03_warmup_benchmark PRIVATE chp_common)
endif()

# --- hotspot_profiling: for perf/gprof analysis ---
add_executable(ch03_hotspot_profiling hotspot_profiling/example.cpp)
