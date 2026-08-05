add_executable(13_baseline baseline.cpp)
target_link_libraries(13_baseline PRIVATE oopt_common)
add_executable(13_optimized optimized.cpp)
target_link_libraries(13_optimized PRIVATE oopt_common)
add_executable(13_benchmark benchmark.cpp)
target_link_libraries(13_benchmark PRIVATE oopt_common)
# AVX-512 example: compiled for AVX-512 but only RUN when the CPU supports
# it. Never enable AVX-512 globally.
add_executable(13_avx512_example avx512_example.cpp)
target_link_libraries(13_avx512_example PRIVATE oopt_common)
target_compile_options(13_avx512_example PRIVATE -mavx512f -mavx512vl -mavx512bw)
