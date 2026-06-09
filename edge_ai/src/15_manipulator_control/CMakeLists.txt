cmake_minimum_required(VERSION 3.14)
project(ManipulatorControl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(manipulator_control
    main.cpp
    kinematics.cpp
    trajectory.cpp
    joint_controller.cpp
)

target_compile_options(manipulator_control PRIVATE -Wall -Wextra -O2 -pthread)

# 不依赖 Eigen3 —— 所有矩阵运算内联实现（固定维度 4×4/6×7/7×7）

target_link_libraries(manipulator_control PRIVATE pthread)
