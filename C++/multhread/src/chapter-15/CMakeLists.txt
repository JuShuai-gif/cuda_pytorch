cmake_minimum_required(VERSION 3.16)
project(Chapter15_Performance LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

set(SOURCES
    01_benchmark_basic.cpp
    02_latency_throughput.cpp
    03_contention_analysis.cpp
    04_lock_vs_lockfree_bench.cpp
    05_perf_guide.cpp
)

foreach(src ${SOURCES})
    get_filename_component(name ${src} NAME_WE)
    add_executable(ch15_${name} ${src})
    target_link_libraries(ch15_${name} PRIVATE Threads::Threads)
endforeach()
