# Chapter 10: Concurrency

# --- threads_basics: std::thread lifecycle ---
add_executable(ch10_threads_basics_example threads_basics/example.cpp)
if(ENABLE_TESTS)
    add_executable(ch10_threads_basics_tests threads_basics/tests.cpp)
    target_link_libraries(ch10_threads_basics_tests PRIVATE chp_common)
endif()

# --- data_race: plain vs mutex vs atomic ---
add_executable(ch10_data_race_example data_race/example.cpp)
if(ENABLE_TESTS)
    add_executable(ch10_data_race_tests data_race/tests.cpp)
    target_link_libraries(ch10_data_race_tests PRIVATE chp_common)
endif()

# --- deadlock: std::lock for multi-lock transactions ---
add_executable(ch10_deadlock_example deadlock/example.cpp)
if(ENABLE_TESTS)
    add_executable(ch10_deadlock_tests deadlock/tests.cpp)
    target_link_libraries(ch10_deadlock_tests PRIVATE chp_common)
endif()

# --- producer_consumer: condition variables ---
add_executable(ch10_producer_consumer_example producer_consumer/example.cpp)
if(ENABLE_TESTS)
    add_executable(ch10_producer_consumer_tests producer_consumer/tests.cpp)
    target_link_libraries(ch10_producer_consumer_tests PRIVATE chp_common)
endif()

# --- promise_future: return data/errors across threads ---
add_executable(ch10_promise_future_example promise_future/example.cpp)
if(ENABLE_TESTS)
    add_executable(ch10_promise_future_tests promise_future/tests.cpp)
    target_link_libraries(ch10_promise_future_tests PRIVATE chp_common)
endif()

# --- async_tasks: packaged_task / std::async ---
add_executable(ch10_async_tasks_example async_tasks/example.cpp)
if(ENABLE_TESTS)
    add_executable(ch10_async_tasks_tests async_tasks/tests.cpp)
    target_link_libraries(ch10_async_tasks_tests PRIVATE chp_common)
endif()

# --- lock_free_queue: single-reader/single-writer queue ---
add_executable(ch10_lock_free_queue_example lock_free_queue/example.cpp)
if(ENABLE_TESTS)
    add_executable(ch10_lock_free_queue_tests lock_free_queue/tests.cpp)
    target_link_libraries(ch10_lock_free_queue_tests PRIVATE chp_common)
endif()

# --- false_sharing: cache-line padding ---
add_executable(ch10_false_sharing_example false_sharing/example.cpp)
if(ENABLE_BENCHMARKS)
    add_executable(ch10_false_sharing_benchmark false_sharing/benchmark.cpp)
    target_link_libraries(ch10_false_sharing_benchmark PRIVATE chp_common)
endif()
if(ENABLE_TESTS)
    add_executable(ch10_false_sharing_tests false_sharing/tests.cpp)
    target_link_libraries(ch10_false_sharing_tests PRIVATE chp_common)
endif()
