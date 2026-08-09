# high_performance_container: open-addressing hash set

add_executable(hash_set_example example.cpp)
if(ENABLE_PROJECT_TESTS)
    add_executable(hash_set_tests tests.cpp)
endif()
if(ENABLE_PROJECT_BENCHMARKS)
    add_executable(hash_set_benchmark benchmark.cpp)
endif()
