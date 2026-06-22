cmake_minimum_required(VERSION 3.16)
project(Chapter16_HPC LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

# 可选: OpenMP
find_package(OpenMP QUIET)

set(SOURCES
    01_openmp_hello.cpp
    02_openmp_parallel_reduce.cpp
    03_openmp_schedule.cpp
    04_gpu_concepts.cpp
)

foreach(src ${SOURCES})
    get_filename_component(name ${src} NAME_WE)
    add_executable(ch16_${name} ${src})
    target_link_libraries(ch16_${name} PRIVATE Threads::Threads)
    if(OpenMP_FOUND AND (name MATCHES "openmp"))
        target_link_libraries(ch16_${name} PRIVATE OpenMP::OpenMP_CXX)
    endif()
endforeach()
