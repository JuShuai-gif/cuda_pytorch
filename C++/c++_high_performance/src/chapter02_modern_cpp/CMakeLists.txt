# Chapter 2: Modern C++ Concepts
#
# Experiment layout (same as Chapter 1):
#   <name>/example.cpp       demonstration (self-validating where noted)
#   <name>/baseline.cpp      intuitive implementation
#   <name>/optimized.cpp     optimization
#   <name>/benchmark.cpp     performance comparison
#   <name>/tests.cpp         correctness tests
#   <name>/README.md         explanation, run instructions, result notes

# --- auto_deduction: auto / const auto& / auto& / auto&& ---
add_executable(ch02_auto_example auto_deduction/example.cpp)
target_link_libraries(ch02_auto_example PRIVATE chp_common)

# --- lambda_basics: captures, mutable, init-capture, capture-all ---
add_executable(ch02_lambda_example lambda_basics/example.cpp)
target_link_libraries(ch02_lambda_example PRIVATE chp_common)

# --- callable_overhead: lambda vs functor vs std::function ---
if(ENABLE_BENCHMARKS)
    add_executable(ch02_callback_benchmark callable_overhead/benchmark.cpp)
    target_link_libraries(ch02_callback_benchmark PRIVATE chp_common)
endif()
if(ENABLE_TESTS)
    add_executable(ch02_callback_tests callable_overhead/tests.cpp)
    target_link_libraries(ch02_callback_tests PRIVATE chp_common)
endif()

# --- std_function_cost: SBO and heap allocation detection ---
add_executable(ch02_function_cost_example std_function_cost/example.cpp)
if(ENABLE_BENCHMARKS)
    add_executable(ch02_function_cost_benchmark std_function_cost/benchmark.cpp)
    target_link_libraries(ch02_function_cost_benchmark PRIVATE chp_common)
endif()

# --- capture_size: lambda size == sum of captures ---
add_executable(ch02_capture_size_example capture_size/example.cpp)

# --- copy_vs_move: when the compiler moves instead of copies ---
add_executable(ch02_copy_move_example copy_vs_move/example.cpp
               copy_vs_move/baseline.cpp)
if(ENABLE_BENCHMARKS)
    add_executable(ch02_copy_move_benchmark copy_vs_move/benchmark.cpp
                   copy_vs_move/baseline.cpp)
    target_link_libraries(ch02_copy_move_benchmark PRIVATE chp_common)
endif()
if(ENABLE_TESTS)
    add_executable(ch02_copy_move_tests copy_vs_move/tests.cpp
                   copy_vs_move/baseline.cpp)
    target_link_libraries(ch02_copy_move_tests PRIVATE chp_common)
endif()

# --- rule_of_three_five_zero ---
add_executable(ch02_rule_of_example rule_of_three_five_zero/example.cpp)
if(ENABLE_TESTS)
    add_executable(ch02_rule_of_tests rule_of_three_five_zero/tests.cpp)
    target_link_libraries(ch02_rule_of_tests PRIVATE chp_common)
endif()

# --- noexcept_move: noexcept affects container reallocation ---
add_executable(ch02_noexcept_example noexcept_move/example.cpp
               noexcept_move/baseline.cpp)
if(ENABLE_BENCHMARKS)
    add_executable(ch02_noexcept_benchmark noexcept_move/benchmark.cpp
                   noexcept_move/baseline.cpp)
    target_link_libraries(ch02_noexcept_benchmark PRIVATE chp_common)
endif()
if(ENABLE_TESTS)
    add_executable(ch02_noexcept_tests noexcept_move/tests.cpp
                   noexcept_move/baseline.cpp)
    target_link_libraries(ch02_noexcept_tests PRIVATE chp_common)
endif()

# --- optional_demo ---
add_executable(ch02_optional_example optional_demo/example.cpp)
if(ENABLE_BENCHMARKS)
    add_executable(ch02_optional_benchmark optional_demo/benchmark.cpp)
    target_link_libraries(ch02_optional_benchmark PRIVATE chp_common)
endif()
if(ENABLE_TESTS)
    add_executable(ch02_optional_tests optional_demo/tests.cpp)
    target_link_libraries(ch02_optional_tests PRIVATE chp_common)
endif()

# --- any_demo ---
add_executable(ch02_any_example any_demo/example.cpp)
if(ENABLE_BENCHMARKS)
    add_executable(ch02_any_benchmark any_demo/benchmark.cpp)
    target_link_libraries(ch02_any_benchmark PRIVATE chp_common)
endif()
if(ENABLE_TESTS)
    add_executable(ch02_any_tests any_demo/tests.cpp)
    target_link_libraries(ch02_any_tests PRIVATE chp_common)
endif()

# --- optional_variant_pointer: four maybe-absent representations ---
add_executable(ch02_optional_variant_example
               optional_variant_pointer/example.cpp)
if(ENABLE_BENCHMARKS)
    add_executable(ch02_optional_variant_benchmark
                   optional_variant_pointer/benchmark.cpp)
    target_link_libraries(ch02_optional_variant_benchmark PRIVATE chp_common)
endif()
if(ENABLE_TESTS)
    add_executable(ch02_optional_variant_tests
                   optional_variant_pointer/tests.cpp)
    target_link_libraries(ch02_optional_variant_tests PRIVATE chp_common)
endif()
