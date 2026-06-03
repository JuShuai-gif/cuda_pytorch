cmake_minimum_required(VERSION 3.16)
project(Chapter12_ModernCXX20 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

set(SOURCES
    01_jthread_basic.cpp
    02_stop_token.cpp
    03_semaphore.cpp
    04_latch.cpp
    05_barrier.cpp
    06_coroutine_demo.cpp
)

foreach(src ${SOURCES})
    get_filename_component(name ${src} NAME_WE)
    add_executable(ch12_${name} ${src})
    target_link_libraries(ch12_${name} PRIVATE Threads::Threads)
endforeach()
