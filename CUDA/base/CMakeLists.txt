add_executable(01_check_dimmension 01_check_dimmension.cu)

add_executable(02_grid_block 02_grid_block.cu)

add_executable(03_sum_arrays 03_sum_arrays.cu)

add_executable(04_sum_arrays_timer 04_sum_arrays_timer.cu)

add_executable(05_thread_index 05_thread_index.cu)

add_executable(06_sum_matrix 06_sum_matrix.cu)

add_executable(07_device_infomation 07_device_infomation.cu)

add_executable(08_divergence 08_divergence.cu)

add_executable(09_sum_matrix2D 09_sum_matrix2D.cu)

add_executable(10_reduceInteger 10_reduceInteger.cu)

add_executable(11_simple_sum_matrix 11_simple_sum_matrix.cu)

add_executable(12_reduceUnrolling 12_reduceUnrolling.cu)

# 开启分离模式
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
add_executable(13_nested_hello_world 13_nested_hello_world.cu)


add_executable(14_global_variable 14_global_variable.cu)

add_executable(15_pine_memory 15_pine_memory.cu)

add_executable(16_zero_copy_memory 16_zero_copy_memory.cu)

add_executable(17_UVA 17_UVA.cu)

add_executable(18_sum_array_offset 18_sum_array_offset.cu)

add_executable(19_AoS 19_AoS.cu)

add_executable(20_SoA 20_SoA.cu)

add_executable(21_sum_array_offset_unrolling 21_sum_array_offset_unrolling.cu)

add_executable(22_transform_matrix2D 22_transform_matrix2D.cu)

add_executable(23_sum_arrays_uniform_memory 23_sum_arrays_uniform_memory.cu)

add_executable(24_shared_memory_read_data 24_shared_memory_read_data.cu)

add_executable(25_reduce_integer_shared_memory 25_reduce_integer_shared_memory.cu)

add_executable(26_transform_shared_memory 26_transform_shared_memory.cu)

add_executable(27_stencil_1d_constant_read_only 27_stencil_1d_constant_read_only.cu)

add_executable(28_shfl_test 28_shfl_test.cu)

add_executable(29_reduce_shfl 29_reduce_shfl.cu)

add_executable(30_stream 30_stream.cu)

find_package(OpenMP REQUIRED)


add_executable(31_stream_omp 31_stream_omp.cu)
# CUDA target
target_link_libraries(31_stream_omp
    PRIVATE
    OpenMP::OpenMP_CXX
)


add_executable(32_stream_resource 32_stream_resource.cu)

add_executable(33_stream_block 33_stream_block.cu)

add_executable(34_stream_dependence 34_stream_dependence.cu)

add_executable(35_multi_add_dpth 35_multi_add_dpth.cu)

add_executable(36_multi_add_breadth 36_multi_add_breadth.cu)

add_executable(37_asyncAPI 37_asyncAPI.cu)

add_executable(38_stream_call_back 38_stream_call_back.cu)

