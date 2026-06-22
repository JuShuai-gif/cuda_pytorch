# Chapter 01: Hello, C++ Concurrency World
# 每个示例独立可执行

set(CHAPTER_01_SOURCES
    01_hello_concurrent.cpp
    02_concurrent_vs_parallel.cpp
    03_num_threads.cpp
    04_industrial_timer.cpp
)

foreach(src_file ${CHAPTER_01_SOURCES})
    get_filename_component(exe_name ${src_file} NAME_WE)
    add_executable(ch01_${exe_name} ${src_file})
    target_link_libraries(ch01_${exe_name} PRIVATE Threads::Threads)
endforeach()
