cmake_minimum_required(VERSION 3.16)
project(Chapter17_Engineering LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

set(SOURCES
    01_sanitizer_demo.cpp
    02_stress_test.cpp
    03_concurrent_unit_test.cpp
    04_cmake_best_practices.cpp
)

foreach(src ${SOURCES})
    get_filename_component(name ${src} NAME_WE)
    add_executable(ch17_${name} ${src})
    target_link_libraries(ch17_${name} PRIVATE Threads::Threads)
endforeach()
