cmake_minimum_required(VERSION 3.16)
project(chapter07_lockfree LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 开启编译优化
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -pthread")

# 所有源文件
set(SOURCES
    01_lockfree_stack.cpp
    02_lockfree_queue_spmc.cpp
    03_lockfree_queue_mpmc.cpp
    04_aba_demo.cpp
    05_hazard_pointer.cpp
    06_backoff.cpp
    07_lockfree_ringbuffer.cpp
    08_epoch_reclamation.cpp
)

# 为每个源文件创建独立的可执行目标
foreach(src ${SOURCES})
    get_filename_component(name ${src} NAME_WE)
    add_executable(ch07_${name} ${src})
    target_link_libraries(ch07_${name} PRIVATE pthread)
endforeach()
