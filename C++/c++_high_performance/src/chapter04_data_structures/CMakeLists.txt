# Chapter 4: Data Structures

# --- cache_thrashing: row-major vs column-major matrix fill ---
if(ENABLE_BENCHMARKS)
    add_executable(ch04_cache_thrashing_benchmark
                   cache_thrashing/benchmark.cpp)
    target_link_libraries(ch04_cache_thrashing_benchmark PRIVATE chp_common)
endif()
if(ENABLE_TESTS)
    add_executable(ch04_cache_thrashing_tests cache_thrashing/tests.cpp)
    target_link_libraries(ch04_cache_thrashing_tests PRIVATE chp_common)
endif()

# --- sequence_containers: traversal and insertion ---
if(ENABLE_BENCHMARKS)
    add_executable(ch04_sequence_benchmark sequence_containers/benchmark.cpp)
    target_link_libraries(ch04_sequence_benchmark PRIVATE chp_common)
    add_executable(ch04_sequence_insert_benchmark
                   sequence_containers/insert_benchmark.cpp)
    target_link_libraries(ch04_sequence_insert_benchmark PRIVATE chp_common)
endif()
if(ENABLE_TESTS)
    add_executable(ch04_sequence_tests sequence_containers/tests.cpp)
    target_link_libraries(ch04_sequence_tests PRIVATE chp_common)
endif()

# --- associative_containers: map vs unordered_map lookup ---
if(ENABLE_BENCHMARKS)
    add_executable(ch04_associative_benchmark
                   associative_containers/benchmark.cpp)
    target_link_libraries(ch04_associative_benchmark PRIVATE chp_common)
endif()
if(ENABLE_TESTS)
    add_executable(ch04_associative_tests associative_containers/tests.cpp)
    target_link_libraries(ch04_associative_tests PRIVATE chp_common)
endif()

# --- hash_policy: buckets, load_factor, good vs bad hash ---
if(ENABLE_BENCHMARKS)
    add_executable(ch04_hash_policy_benchmark hash_policy/benchmark.cpp)
    target_link_libraries(ch04_hash_policy_benchmark PRIVATE chp_common)
endif()
if(ENABLE_TESTS)
    add_executable(ch04_hash_policy_tests hash_policy/tests.cpp)
    target_link_libraries(ch04_hash_policy_tests PRIVATE chp_common)
endif()

# --- priority_queue: partial sort ---
add_executable(ch04_priority_queue_example priority_queue/example.cpp)

# --- vector_bool: specialized bit array ---
add_executable(ch04_vector_bool_example vector_bool/example.cpp)
target_link_libraries(ch04_vector_bool_example PRIVATE chp_common)

# --- aos_vs_soa: AoS vs parallel arrays ---
if(ENABLE_BENCHMARKS)
    add_executable(ch04_aos_vs_soa_benchmark
        aos_vs_soa/benchmark.cpp
        aos_vs_soa/baseline.cpp
        aos_vs_soa/optimized.cpp
    )
    target_link_libraries(ch04_aos_vs_soa_benchmark PRIVATE chp_common)
endif()
if(ENABLE_TESTS)
    add_executable(ch04_aos_vs_soa_tests
        aos_vs_soa/tests.cpp
        aos_vs_soa/baseline.cpp
        aos_vs_soa/optimized.cpp
    )
    target_link_libraries(ch04_aos_vs_soa_tests PRIVATE chp_common)
endif()
