# Chapter 1: A Brief Introduction to C++ (zero-cost abstractions)
#
# Each experiment follows the layout:
#   <name>/baseline.cpp   intuitive, possibly inefficient implementation
#   <name>/optimized.cpp  abstraction/optimized implementation
#   <name>/benchmark.cpp  performance comparison
#   <name>/tests.cpp      correctness tests
#   <name>/README.md      explanation, run instructions, result notes
#
# Note: const_correctness/compile_error_example.cpp is intentionally
# ill-formed and is NOT added to any target. See its header comment.

# --- linked_list_search: C-style linked list vs STL search ---
if(ENABLE_BENCHMARKS)
    add_executable(ch01_lls_benchmark
        linked_list_search/benchmark.cpp
        linked_list_search/baseline.cpp
        linked_list_search/optimized.cpp
    )
    target_link_libraries(ch01_lls_benchmark PRIVATE chp_common)
endif()
if(ENABLE_TESTS)
    add_executable(ch01_lls_tests
        linked_list_search/tests.cpp
        linked_list_search/baseline.cpp
        linked_list_search/optimized.cpp
    )
    target_link_libraries(ch01_lls_tests PRIVATE chp_common)
endif()

# --- loop_vs_count: hand-written loop vs std::count ---
if(ENABLE_BENCHMARKS)
    add_executable(ch01_lvc_benchmark
        loop_vs_count/benchmark.cpp
        loop_vs_count/baseline.cpp
        loop_vs_count/optimized.cpp
    )
    target_link_libraries(ch01_lvc_benchmark PRIVATE chp_common)
endif()
if(ENABLE_TESTS)
    add_executable(ch01_lvc_tests
        loop_vs_count/tests.cpp
        loop_vs_count/baseline.cpp
        loop_vs_count/optimized.cpp
    )
    target_link_libraries(ch01_lvc_tests PRIVATE chp_common)
endif()

# --- value_semantics: value vs reference/shared semantics ---
add_executable(ch01_vs_example value_semantics/example.cpp)
add_executable(ch01_vs_tests value_semantics/tests.cpp)
    target_link_libraries(ch01_vs_tests PRIVATE chp_common)

# --- contiguous_vs_pointer: contiguous vs pointer storage ---
if(ENABLE_BENCHMARKS)
    add_executable(ch01_cvp_benchmark
        contiguous_vs_pointer/benchmark.cpp
        contiguous_vs_pointer/baseline.cpp
        contiguous_vs_pointer/optimized.cpp
    )
    target_link_libraries(ch01_cvp_benchmark PRIVATE chp_common)
endif()
if(ENABLE_TESTS)
    add_executable(ch01_cvp_tests
        contiguous_vs_pointer/tests.cpp
        contiguous_vs_pointer/baseline.cpp
        contiguous_vs_pointer/optimized.cpp
    )
    target_link_libraries(ch01_cvp_tests PRIVATE chp_common)
endif()

# --- const_correctness: const member functions ---
add_executable(ch01_cc_example const_correctness/example.cpp)
add_executable(ch01_cc_tests const_correctness/tests.cpp)
    target_link_libraries(ch01_cc_tests PRIVATE chp_common)

# --- raii_resource: manual release vs RAII ---
if(ENABLE_BENCHMARKS)
    add_executable(ch01_raii_benchmark
        raii_resource/benchmark.cpp
        raii_resource/baseline.cpp
        raii_resource/optimized.cpp
    )
    target_link_libraries(ch01_raii_benchmark PRIVATE chp_common)
endif()
if(ENABLE_TESTS)
    add_executable(ch01_raii_tests
        raii_resource/tests.cpp
        raii_resource/baseline.cpp
        raii_resource/optimized.cpp
    )
    target_link_libraries(ch01_raii_tests PRIVATE chp_common)
endif()

# --- exceptions_vs_error_codes: unthrown exception cost ---
if(ENABLE_BENCHMARKS)
    add_executable(ch01_evc_benchmark
        exceptions_vs_error_codes/benchmark.cpp
        exceptions_vs_error_codes/baseline.cpp
        exceptions_vs_error_codes/optimized.cpp
    )
    target_link_libraries(ch01_evc_benchmark PRIVATE chp_common)
endif()
if(ENABLE_TESTS)
    add_executable(ch01_evc_tests
        exceptions_vs_error_codes/tests.cpp
        exceptions_vs_error_codes/baseline.cpp
        exceptions_vs_error_codes/optimized.cpp
    )
    target_link_libraries(ch01_evc_tests PRIVATE chp_common)
endif()

# --- zero_cost_asm: compare generated assembly ---
add_executable(ch01_zca_example zero_cost_asm/example.cpp)
add_executable(ch01_zca_tests zero_cost_asm/tests.cpp)
    target_link_libraries(ch01_zca_tests PRIVATE chp_common)
