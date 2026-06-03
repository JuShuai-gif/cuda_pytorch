# Chapter 02: Thread Management
# 每个示例独立可执行

set(CHAPTER_02_SOURCES
    01_basic_thread.cpp
    02_thread_guard.cpp
    03_joining_thread.cpp
    04_param_pass.cpp
    05_thread_ownership.cpp
    06_parallel_accumulate.cpp
)

foreach(src_file ${CHAPTER_02_SOURCES})
    get_filename_component(exe_name ${src_file} NAME_WE)
    add_executable(ch02_${exe_name} ${src_file})
    target_link_libraries(ch02_${exe_name} PRIVATE Threads::Threads)
endforeach()
