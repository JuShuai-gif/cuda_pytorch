cmake_minimum_required(VERSION 3.14)
project(IndustrialConcurrency VERSION 1.0.0 LANGUAGES CXX)

# Ch10.2：支持多种工具链的构建配置。
# 支持常规 Debug/Release 以及 TSan（ThreadSanitizer）构建。

# 设置 C++20 标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Ch10.2.3：用于并发 bug 检测的编译器警告。
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Clang 专用：线程安全注解（thread safety annotations）
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wthread-safety)
endif()

# 必需：Threads 库（Ch2.1：std::thread 在 Linux 上依赖 pthread）。
find_package(Threads REQUIRED)

# 必需：{fmt} 库，用于可移植的字符串格式化（当 <format> 不可用时使用）。
find_package(fmt REQUIRED)

# ============================================================================
# 核心库（仅头文件的接口库）
# ============================================================================
add_library(task_scheduler_lib INTERFACE)
target_include_directories(task_scheduler_lib INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(task_scheduler_lib INTERFACE Threads::Threads fmt::fmt)

# ============================================================================
# 库的源文件（非模板实现部分）
# ============================================================================
set(LIBRARY_SOURCES
    src/thread_pool.cpp
    src/task_scheduler.cpp
    src/logger.cpp
)

# 可选：将库源文件编译为静态库以加快构建速度。
add_library(task_scheduler_static STATIC ${LIBRARY_SOURCES})
target_include_directories(task_scheduler_static PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(task_scheduler_static PUBLIC Threads::Threads fmt::fmt)

# ============================================================================
# 主程序入口
# ============================================================================
add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE task_scheduler_static)

# ============================================================================
# 测试程序（每个均可独立编译）
# ============================================================================
add_executable(test_thread_pool tests/test_thread_pool.cpp)
target_link_libraries(test_thread_pool PRIVATE task_scheduler_static)

add_executable(test_task_queue tests/test_task_queue.cpp)
target_link_libraries(test_task_queue PRIVATE task_scheduler_static)

add_executable(test_task_scheduler tests/test_task_scheduler.cpp)
target_link_libraries(test_task_scheduler PRIVATE task_scheduler_static)

add_executable(test_stress tests/test_stress.cpp)
target_link_libraries(test_stress PRIVATE task_scheduler_static)

# ============================================================================
# 示例程序（每个均可独立编译）
# ============================================================================
add_executable(example_basic examples/example_basic.cpp)
target_link_libraries(example_basic PRIVATE task_scheduler_static)

add_executable(example_pipeline examples/example_pipeline.cpp)
target_link_libraries(example_pipeline PRIVATE task_scheduler_static)

add_executable(example_inference examples/example_inference.cpp)
target_link_libraries(example_inference PRIVATE task_scheduler_static)

add_executable(example_producer_consumer examples/example_producer_consumer.cpp)
target_link_libraries(example_producer_consumer PRIVATE task_scheduler_static)

# ============================================================================
# 自定义构建类型
# ============================================================================

# Ch10.2.4：用于数据竞争检测的 ThreadSanitizer 构建类型。
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_C_FLAGS_TSAN
        "-fsanitize=thread -fno-omit-frame-pointer -g -O1"
        CACHE STRING "ThreadSanitizer 构建的编译选项" FORCE)
    set(CMAKE_CXX_FLAGS_TSAN
        "-fsanitize=thread -fno-omit-frame-pointer -g -O1"
        CACHE STRING "ThreadSanitizer 构建的编译选项" FORCE)
    set(CMAKE_EXE_LINKER_FLAGS_TSAN
        "-fsanitize=thread"
        CACHE STRING "ThreadSanitizer 构建的链接选项" FORCE)
endif()

# ============================================================================
# 自定义构建目标
# ============================================================================

# 便捷目标：构建所有测试
add_custom_target(tests DEPENDS
    test_thread_pool test_task_queue test_task_scheduler test_stress)

# 便捷目标：构建所有示例
add_custom_target(examples DEPENDS
    example_basic example_pipeline example_inference example_producer_consumer)

# 通过 CTest 运行所有测试
enable_testing()
add_test(NAME ThreadPool     COMMAND test_thread_pool)
add_test(NAME TaskQueue      COMMAND test_task_queue)
add_test(NAME TaskScheduler  COMMAND test_task_scheduler)
add_test(NAME Stress         COMMAND test_stress)

# ============================================================================
# 安装规则
# ============================================================================
install(TARGETS task_scheduler_static
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib)
install(TARGETS task_scheduler_lib EXPORT task_schedulerTargets)
install(DIRECTORY include/ DESTINATION include)
