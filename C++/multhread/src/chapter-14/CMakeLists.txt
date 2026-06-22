cmake_minimum_required(VERSION 3.16)
project(Chapter14_ParallelAlgorithms LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

set(SOURCES
    01_pipeline.cpp
    02_parallel_reduce.cpp
    03_parallel_scan.cpp
    04_batch_processing.cpp
    05_work_stealing_deep.cpp
    06_data_task_parallelism.cpp
)

foreach(src ${SOURCES})
    get_filename_component(name ${src} NAME_WE)
    add_executable(ch14_${name} ${src})
    target_link_libraries(ch14_${name} PRIVATE Threads::Threads)
endforeach()
