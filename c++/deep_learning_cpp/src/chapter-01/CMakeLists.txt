add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
    INTERFACE_INCLUDE_DIRECTORIES ${ONNXRUNTIME_ROOT}/include
)

list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_ROOT})

find_package(Torch REQUIRED)

add_executable(vector_add vector_add.cu)
set_target_properties(vector_add PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(lib_torch lib_torch.cpp)
target_link_libraries(lib_torch ${TORCH_LIBRARIES})
target_include_directories(lib_torch PRIVATE ${TORCH_INCLUDE_DIRS})
set_target_properties(lib_torch PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

add_executable(mini_net mini_net.cpp)
target_link_libraries(mini_net ${TORCH_LIBRARIES})
set_target_properties(mini_net PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
