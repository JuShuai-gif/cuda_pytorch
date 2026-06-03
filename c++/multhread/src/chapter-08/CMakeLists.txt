cmake_minimum_required(VERSION 3.16)
project(chapter08_parallel_algorithms LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -pthread")

set(SOURCES
    01_parallel_for_each.cpp
    02_parallel_find.cpp
    03_parallel_partial_sum.cpp
    04_false_sharing.cpp
    05_exception_safe_parallel.cpp
)

foreach(src ${SOURCES})
    get_filename_component(name ${src} NAME_WE)
    add_executable(ch08_${name} ${src})
    target_link_libraries(ch08_${name} PRIVATE pthread)
endforeach()
