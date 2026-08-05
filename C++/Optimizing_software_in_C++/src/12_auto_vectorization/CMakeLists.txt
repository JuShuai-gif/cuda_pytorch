add_executable(12_vectorizable vectorizable.cpp)
target_link_libraries(12_vectorizable PRIVATE oopt_common)
add_executable(12_non_vectorizable non_vectorizable.cpp)
target_link_libraries(12_non_vectorizable PRIVATE oopt_common)
add_executable(12_benchmark benchmark.cpp)
target_link_libraries(12_benchmark PRIVATE oopt_common)
# This experiment is about vectorization; allow AVX2 on this machine.
target_compile_options(12_benchmark PRIVATE -mavx2)
