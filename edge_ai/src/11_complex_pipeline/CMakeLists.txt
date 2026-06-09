cmake_minimum_required(VERSION 3.14)
project(ComplexPipelineSim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(pipeline_sim
    latency_stats.cpp
    stage_ops.cpp
    pipeline_executor.cpp
    main.cpp
)

target_compile_options(pipeline_sim PRIVATE -Wall -Wextra -O2 -pthread)
target_link_libraries(pipeline_sim PRIVATE pthread)
