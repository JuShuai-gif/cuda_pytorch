cmake_minimum_required(VERSION 3.16)
project(Chapter04_Synchronization LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 查找线程库 (Linux 通常需要 -pthread)
find_package(Threads REQUIRED)

# 每个 .cpp 编译为独立可执行文件
set(SOURCES
    01_condition_variable.cpp
    02_producer_consumer_queue.cpp
    03_future_promise.cpp
    04_async_task.cpp
    05_packaged_task.cpp
    06_shared_future.cpp
    07_timed_wait.cpp
)

foreach(src ${SOURCES})
    get_filename_component(name ${src} NAME_WE)
    add_executable(ch04_${name} ${src})
    target_link_libraries(ch04_${name} PRIVATE Threads::Threads)
endforeach()
