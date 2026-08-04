cmake_minimum_required(VERSION 3.16)
project(chapter11_testing_debugging LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -pthread")

# 开启 ThreadSanitizer 的 Debug 构建
# cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fsanitize=thread -g -O1" ..

set(SOURCES
    01_deadlock_demo.cpp
    02_race_detector_demo.cpp
    03_stress_test.cpp
    04_logger_for_debug.cpp
)

foreach(src ${SOURCES})
    get_filename_component(name ${src} NAME_WE)
    add_executable(ch11_${name} ${src})
    target_link_libraries(ch11_${name} PRIVATE pthread)
endforeach()
