# Chapter 9: Proxy Objects and Lazy Evaluation

# --- eager_vs_lazy: eager value vs lazy factory fallback ---
add_executable(ch09_eager_vs_lazy_example eager_vs_lazy/example.cpp)
if(ENABLE_TESTS)
    add_executable(ch09_eager_vs_lazy_tests eager_vs_lazy/tests.cpp)
    target_link_libraries(ch09_eager_vs_lazy_tests PRIVATE chp_common)
endif()

# --- concat_proxy: compare concatenated strings without a temporary ---
add_executable(ch09_concat_proxy_example concat_proxy/example.cpp)
if(ENABLE_BENCHMARKS)
    add_executable(ch09_concat_proxy_benchmark concat_proxy/benchmark.cpp)
    target_link_libraries(ch09_concat_proxy_benchmark PRIVATE chp_common)
endif()
if(ENABLE_TESTS)
    add_executable(ch09_concat_proxy_tests concat_proxy/tests.cpp)
    target_link_libraries(ch09_concat_proxy_tests PRIVATE chp_common)
endif()

# --- distance_proxy: postpone std::sqrt when comparing distances ---
add_executable(ch09_distance_proxy_example distance_proxy/example.cpp)
if(ENABLE_BENCHMARKS)
    add_executable(ch09_distance_proxy_benchmark distance_proxy/benchmark.cpp)
    target_link_libraries(ch09_distance_proxy_benchmark PRIVATE chp_common)
endif()
if(ENABLE_TESTS)
    add_executable(ch09_distance_proxy_tests distance_proxy/tests.cpp)
    target_link_libraries(ch09_distance_proxy_tests PRIVATE chp_common)
endif()

# --- pipe_operator: range | contains(value) ---
add_executable(ch09_pipe_example pipe_operator/example.cpp)
if(ENABLE_TESTS)
    add_executable(ch09_pipe_tests pipe_operator/tests.cpp)
    target_link_libraries(ch09_pipe_tests PRIVATE chp_common)
endif()

# --- infix_operator: value <in> range ---
add_executable(ch09_infix_example infix_operator/example.cpp)
if(ENABLE_TESTS)
    add_executable(ch09_infix_tests infix_operator/tests.cpp)
    target_link_libraries(ch09_infix_tests PRIVATE chp_common)
endif()
