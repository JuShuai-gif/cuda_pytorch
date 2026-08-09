# Chapter 11: Parallel STL

# --- par_transform: naive vs divide-and-conquer ---
add_executable(ch11_par_transform_example par_transform/example.cpp)
if(ENABLE_BENCHMARKS)
    add_executable(ch11_par_transform_benchmark par_transform/benchmark.cpp)
    target_link_libraries(ch11_par_transform_benchmark PRIVATE chp_common Threads::Threads ${TBB_LIBRARY})
endif()
if(ENABLE_TESTS)
    add_executable(ch11_par_transform_tests par_transform/tests.cpp)
    target_link_libraries(ch11_par_transform_tests PRIVATE chp_common Threads::Threads ${TBB_LIBRARY})
endif()

# --- par_copy_if: atomic index vs split ---
add_executable(ch11_par_copy_if_example par_copy_if/example.cpp)
if(ENABLE_BENCHMARKS)
    add_executable(ch11_par_copy_if_benchmark par_copy_if/benchmark.cpp)
    target_link_libraries(ch11_par_copy_if_benchmark PRIVATE chp_common Threads::Threads ${TBB_LIBRARY})
endif()
if(ENABLE_TESTS)
    add_executable(ch11_par_copy_if_tests par_copy_if/tests.cpp)
    target_link_libraries(ch11_par_copy_if_tests PRIVATE chp_common Threads::Threads ${TBB_LIBRARY})
endif()

# --- execution_policies: seq / par / par_unseq, reduce ---
add_executable(ch11_execution_policies_example execution_policies/example.cpp)
target_link_libraries(ch11_execution_policies_example PRIVATE ${TBB_LIBRARY})
if(ENABLE_TESTS)
    add_executable(ch11_execution_policies_tests execution_policies/tests.cpp)
    target_link_libraries(ch11_execution_policies_tests PRIVATE chp_common Threads::Threads ${TBB_LIBRARY})
endif()

# --- parallel_for: index loops via LinearRange ---
add_executable(ch11_parallel_for_example parallel_for/example.cpp)
target_link_libraries(ch11_parallel_for_example PRIVATE ${TBB_LIBRARY})
if(ENABLE_TESTS)
    add_executable(ch11_parallel_for_tests parallel_for/tests.cpp)
    target_link_libraries(ch11_parallel_for_tests PRIVATE chp_common Threads::Threads ${TBB_LIBRARY})
endif()

# --- boost_compute: GPU algorithms (optional, needs OpenCL + Boost) ---
if(ENABLE_BOOST_COMPUTE AND ENABLE_OPENCL)
    find_path(OPENCL_INCLUDE_DIR NAMES CL/cl.h PATHS
        /usr/local/cuda-12.8/targets/x86_64-linux/include
        /usr/local/cuda/targets/x86_64-linux/include
        /usr/include)
    find_library(OPENCL_LIBRARY NAMES OpenCL PATHS
        /usr/local/cuda-12.8/targets/x86_64-linux/lib
        /usr/local/cuda/targets/x86_64-linux/lib
        /usr/lib/x86_64-linux-gnu)
    if(OPENCL_INCLUDE_DIR AND OPENCL_LIBRARY)
        add_executable(ch11_boost_compute_example boost_compute/example.cpp)
        target_include_directories(ch11_boost_compute_example PRIVATE
            ${OPENCL_INCLUDE_DIR})
        target_link_libraries(ch11_boost_compute_example PRIVATE
            ${OPENCL_LIBRARY} Threads::Threads)
    endif()
endif()
