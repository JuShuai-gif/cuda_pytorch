# Chapter 03: Sharing Data Between Threads
# 每个示例独立可执行

set(CHAPTER_03_SOURCES
    01_race_condition.cpp
    02_mutex_basic.cpp
    03_scoped_lock.cpp
    04_unique_lock.cpp
    05_hierarchical_mutex.cpp
    06_call_once.cpp
    07_shared_mutex.cpp
    08_threadsafe_stack.cpp
    09_recursive_mutex.cpp
    10_timed_mutex.cpp
)

foreach(src_file ${CHAPTER_03_SOURCES})
    get_filename_component(exe_name ${src_file} NAME_WE)
    add_executable(ch03_${exe_name} ${src_file})
    target_link_libraries(ch03_${exe_name} PRIVATE Threads::Threads)
endforeach()
