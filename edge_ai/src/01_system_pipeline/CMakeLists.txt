cmake_minimum_required(VERSION 3.14)
project(RobotPipelineSimulator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(pipeline_sim
    main.cpp
    latency_stats.cpp
    pipeline_stage.cpp
    pipeline_executor.cpp
)

target_compile_options(pipeline_sim PRIVATE -Wall -Wextra -O2 -pthread)
target_link_libraries(pipeline_sim PRIVATE pthread)

# Optional: enable more aggressive optimization for benchmarking
# target_compile_options(pipeline_sim PRIVATE -march=native -mtune=native)
