cmake_minimum_required(VERSION 3.16)
project(chapter09_thread_pools LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -pthread")

set(SOURCES
    01_simple_thread_pool.cpp
    02_thread_pool_with_future.cpp
    03_work_stealing_pool.cpp
    04_interruptible_thread.cpp
)

foreach(src ${SOURCES})
    get_filename_component(name ${src} NAME_WE)
    add_executable(ch09_${name} ${src})
    target_link_libraries(ch09_${name} PRIVATE pthread)
endforeach()
