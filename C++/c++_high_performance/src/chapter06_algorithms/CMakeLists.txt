# Chapter 6: STL Algorithms and Beyond
#
# The C++20 ranges example is built only when ENABLE_CPP20_EXAMPLES=ON and
# the compiler supports C++20 (GCC 10+ / Clang 10+). The rest of the chapter
# stays C++17.

# --- algorithm_basics: iterators, remove/unique idioms ---
add_executable(ch06_algorithm_basics_example algorithm_basics/example.cpp)
if(ENABLE_TESTS)
    add_executable(ch06_algorithm_basics_tests algorithm_basics/tests.cpp)
    target_link_libraries(ch06_algorithm_basics_tests PRIVATE chp_common)
endif()

# --- output_iterators: preallocated vs insert iterators ---
add_executable(ch06_output_iterators_example output_iterators/example.cpp)
if(ENABLE_TESTS)
    add_executable(ch06_output_iterators_tests output_iterators/tests.cpp)
    target_link_libraries(ch06_output_iterators_tests PRIVATE chp_common)
endif()

# --- comparators: custom comparators and predicates ---
add_executable(ch06_comparators_example comparators/example.cpp)
if(ENABLE_TESTS)
    add_executable(ch06_comparators_tests comparators/tests.cpp)
    target_link_libraries(ch06_comparators_tests PRIVATE chp_common)
endif()

# --- rotate_example: for-loop vs std::rotate ---
add_executable(ch06_rotate_example rotate_example/example.cpp)
if(ENABLE_TESTS)
    add_executable(ch06_rotate_tests rotate_example/tests.cpp)
    target_link_libraries(ch06_rotate_tests PRIVATE chp_common)
endif()

# --- find_unroll: naive vs unrolled std::find ---
if(ENABLE_BENCHMARKS)
    add_executable(ch06_find_unroll_benchmark find_unroll/benchmark.cpp
                   find_unroll/find.hpp)
    target_link_libraries(ch06_find_unroll_benchmark PRIVATE chp_common)
endif()
if(ENABLE_TESTS)
    add_executable(ch06_find_unroll_tests find_unroll/tests.cpp
                   find_unroll/find.hpp)
    target_link_libraries(ch06_find_unroll_tests PRIVATE chp_common)
endif()

# --- partial_sorting: sort vs partial_sort vs nth_element ---
if(ENABLE_BENCHMARKS)
    add_executable(ch06_partial_sorting_benchmark
                   partial_sorting/benchmark.cpp)
    target_link_libraries(ch06_partial_sorting_benchmark PRIVATE chp_common)
endif()
if(ENABLE_TESTS)
    add_executable(ch06_partial_sorting_tests partial_sorting/tests.cpp)
    target_link_libraries(ch06_partial_sorting_tests PRIVATE chp_common)
endif()

# --- loop_vs_algorithms: hand loops vs STL algorithms ---
if(ENABLE_BENCHMARKS)
    add_executable(ch06_loop_vs_algos_benchmark
        loop_vs_algorithms/benchmark.cpp
        loop_vs_algorithms/baseline.cpp
        loop_vs_algorithms/optimized.cpp
    )
    target_link_libraries(ch06_loop_vs_algos_benchmark PRIVATE chp_common)
endif()
if(ENABLE_TESTS)
    add_executable(ch06_loop_vs_algos_tests
        loop_vs_algorithms/tests.cpp
        loop_vs_algorithms/baseline.cpp
        loop_vs_algorithms/optimized.cpp
    )
    target_link_libraries(ch06_loop_vs_algos_tests PRIVATE chp_common)
endif()

# --- cpp20_ranges: modern std::ranges (optional, needs C++20) ---
if(ENABLE_CPP20_EXAMPLES)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 10)
        set(CH06_CPP20_OK TRUE)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 10)
        set(CH06_CPP20_OK TRUE)
    else()
        set(CH06_CPP20_OK FALSE)
    endif()
    if(CH06_CPP20_OK)
        add_executable(ch06_cpp20_ranges_example cpp20_ranges/example.cpp)
        target_compile_features(ch06_cpp20_ranges_example PRIVATE cxx_std_20)
        if(ENABLE_TESTS)
            add_executable(ch06_cpp20_ranges_tests cpp20_ranges/tests.cpp)
            target_compile_features(ch06_cpp20_ranges_tests PRIVATE cxx_std_20)
            target_link_libraries(ch06_cpp20_ranges_tests PRIVATE chp_common)
        endif()
    else()
        message(STATUS "Chapter 6: C++20 ranges skipped (compiler too old)")
    endif()
endif()
