cmake_minimum_required(VERSION 3.14)
project(HumanoidBalanceWBC LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Eigen3 头文件路径（header-only 库）
find_path(EIGEN3_INCLUDE_DIR NAMES Eigen/Core
    PATHS /usr/include/eigen3 /usr/local/include/eigen3
    ${EIGEN3_INCLUDE_DIR}
)
if(NOT EIGEN3_INCLUDE_DIR)
    message(FATAL_ERROR "Eigen3 未找到。请安装：sudo apt install libeigen3-dev")
endif()
message(STATUS "Eigen3 路径：${EIGEN3_INCLUDE_DIR}")

add_executable(humanoid_balance
    main.cpp
    wbc_core.cpp
    balance_control.cpp
    task_stack.cpp
)

target_include_directories(humanoid_balance PRIVATE ${EIGEN3_INCLUDE_DIR})

target_compile_options(humanoid_balance PRIVATE
    -Wall -Wextra -O2 -pthread -march=native
)

target_link_libraries(humanoid_balance PRIVATE pthread)
