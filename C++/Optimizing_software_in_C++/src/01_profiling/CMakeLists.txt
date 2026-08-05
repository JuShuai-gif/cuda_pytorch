add_executable(01_simple_timer simple_timer.cpp)
target_link_libraries(01_simple_timer PRIVATE oopt_common)

add_executable(01_hotspot hotspot.cpp)
target_link_libraries(01_hotspot PRIVATE oopt_common)

add_executable(01_benchmark benchmark.cpp)
target_link_libraries(01_benchmark PRIVATE oopt_common)
