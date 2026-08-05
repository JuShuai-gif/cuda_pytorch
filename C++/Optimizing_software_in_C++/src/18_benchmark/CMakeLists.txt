# Debug vs Release from the same source (PDF p85).
add_executable(18_debug debug_release.cpp)
target_compile_options(18_debug PRIVATE -O0)
add_executable(18_release debug_release.cpp)
target_link_libraries(18_release PRIVATE oopt_common)

add_executable(18_warmup warmup.cpp)
target_link_libraries(18_warmup PRIVATE oopt_common)
add_executable(18_size size.cpp)
target_link_libraries(18_size PRIVATE oopt_common)
add_executable(18_eliminate eliminate.cpp)
target_link_libraries(18_eliminate PRIVATE oopt_common)
add_executable(18_fluctuation fluctuation.cpp)
target_link_libraries(18_fluctuation PRIVATE oopt_common)
