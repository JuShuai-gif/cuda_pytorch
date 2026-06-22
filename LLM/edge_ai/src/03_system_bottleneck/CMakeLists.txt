cmake_minimum_required(VERSION 3.14)
project(SystemBottleneckDemos LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(bottleneck_demos
    main.cpp
    cache_bench.cpp
    lock_bench.cpp
    memory_bench.cpp
)

target_compile_options(bottleneck_demos PRIVATE -Wall -Wextra -O2 -pthread)
target_link_libraries(bottleneck_demos PRIVATE pthread)

# For best benchmarking results, consider:
# target_compile_options(bottleneck_demos PRIVATE -march=native -O3)
