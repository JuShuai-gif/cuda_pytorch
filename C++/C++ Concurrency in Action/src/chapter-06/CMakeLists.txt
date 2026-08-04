cmake_minimum_required(VERSION 3.16)
project(Chapter06_DataStructures LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

set(SOURCES
    01_threadsafe_stack.cpp
    02_threadsafe_queue.cpp
    03_threadsafe_queue_fine.cpp
    04_threadsafe_lookup_table.cpp
    05_threadsafe_list.cpp
)

foreach(src ${SOURCES})
    get_filename_component(name ${src} NAME_WE)
    add_executable(ch06_${name} ${src})
    target_link_libraries(ch06_${name} PRIVATE Threads::Threads)
endforeach()
