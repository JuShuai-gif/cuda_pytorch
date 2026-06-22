cmake_minimum_required(VERSION 3.14)
project(PerformanceContract LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(contract_demo
    main.cpp
    modules.cpp
    validator.cpp
)

target_compile_options(contract_demo PRIVATE -Wall -Wextra -O2 -pthread)
target_link_libraries(contract_demo PRIVATE pthread)
