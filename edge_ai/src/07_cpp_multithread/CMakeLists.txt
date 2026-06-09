cmake_minimum_required(VERSION 3.14)
project(MultithreadDemos LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(multithread_demos
    main.cpp
    benchmarks.cpp
)

target_compile_options(multithread_demos PRIVATE -Wall -Wextra -O2 -pthread)
target_link_libraries(multithread_demos PRIVATE pthread)

# For best benchmarking results, consider:
# target_compile_options(multithread_demos PRIVATE -march=native -O3 -DNDEBUG)
