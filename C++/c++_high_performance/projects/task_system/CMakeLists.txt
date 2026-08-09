# task_system: thread pool

add_executable(task_system_example example.cpp)
if(ENABLE_PROJECT_TESTS)
    add_executable(task_system_tests tests.cpp)
    target_link_libraries(task_system_tests PRIVATE Threads::Threads)
endif()
if(ENABLE_PROJECT_BENCHMARKS)
    add_executable(task_system_benchmark benchmark.cpp)
    target_link_libraries(task_system_benchmark PRIVATE Threads::Threads)
endif()
