cmake_minimum_required(VERSION 3.16)
project(Chapter05_Atomics LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

set(SOURCES
    01_atomic_basic.cpp
    02_compare_exchange.cpp
    03_memory_order_relaxed.cpp
    04_memory_order_acq_rel.cpp
    05_memory_order_seq_cst.cpp
    06_spinlock.cpp
    07_lockfree_counter.cpp
)

foreach(src ${SOURCES})
    get_filename_component(name ${src} NAME_WE)
    add_executable(ch05_${name} ${src})
    target_link_libraries(ch05_${name} PRIVATE Threads::Threads)
endforeach()
