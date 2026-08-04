cmake_minimum_required(VERSION 3.16)
project(Chapter13_MemoryModel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

set(SOURCES
    01_cache_line_alignment.cpp
    02_false_sharing_bench.cpp
    03_memory_fence.cpp
    04_numa_affinity.cpp
    05_data_layout_optimization.cpp
)

foreach(src ${SOURCES})
    get_filename_component(name ${src} NAME_WE)
    add_executable(ch13_${name} ${src})
    target_link_libraries(ch13_${name} PRIVATE Threads::Threads)
endforeach()
