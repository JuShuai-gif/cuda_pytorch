cmake_minimum_required(VERSION 3.14)
project(RobotTaskGraphRuntime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(task_graph_demo
    main.cpp
    task_executor.cpp
    pipeline.cpp
)

target_compile_options(task_graph_demo PRIVATE -Wall -Wextra -O2 -pthread)
target_link_libraries(task_graph_demo PRIVATE pthread)
