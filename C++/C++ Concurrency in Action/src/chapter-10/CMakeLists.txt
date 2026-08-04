cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(TBB QUIET)

set(HAS_TBB FALSE)
if(TBB_FOUND OR TBB_tbb_FOUND)
    set(HAS_TBB TRUE)
endif()

set(SOURCES
    01_parallel_sort.cpp
    02_parallel_for_each.cpp
    03_parallel_transform_reduce.cpp
    04_execution_policies.cpp
)

foreach(src ${SOURCES})
    get_filename_component(name ${src} NAME_WE)
    add_executable(ch10_${name} ${src})
    target_link_libraries(ch10_${name} PRIVATE Threads::Threads)
    if(HAS_TBB)
        target_link_libraries(ch10_${name} PRIVATE TBB::tbb)
        target_compile_definitions(ch10_${name} PRIVATE HAS_TBB=1)
    endif()
endforeach()
