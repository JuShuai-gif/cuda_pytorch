cmake_minimum_required(VERSION 3.14)
project(RealTimeSchedulingSim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(rt_sim
    rms_scheduler.cpp
    edf_scheduler.cpp
    priority_inversion.cpp
    main.cpp
)

target_compile_options(rt_sim PRIVATE -Wall -Wextra -O2 -pthread)
target_link_libraries(rt_sim PRIVATE pthread)
