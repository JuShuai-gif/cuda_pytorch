cmake_minimum_required(VERSION 3.14)
project(ROS2RealTimePatterns LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(ros2_rt_demo
    main.cpp
    rt_executor.cpp
    lifecycle.cpp
    qos_demo.cpp
)

target_compile_options(ros2_rt_demo PRIVATE -Wall -Wextra -O2 -pthread)
target_link_libraries(ros2_rt_demo PRIVATE pthread)

# 如需更好的基准测试结果，可用:
# target_compile_options(ros2_rt_demo PRIVATE -march=native -O3 -DNDEBUG)
