# object_pool: fixed-size object pool

add_executable(object_pool_example example.cpp)
if(ENABLE_PROJECT_TESTS)
    add_executable(object_pool_tests tests.cpp)
endif()
if(ENABLE_PROJECT_BENCHMARKS)
    add_executable(object_pool_benchmark benchmark.cpp)
endif()
