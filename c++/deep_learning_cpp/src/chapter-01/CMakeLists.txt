cmake_minimum_required(VERSION 3.22)
project(Chapter01_Environment LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(vector_add vector_add.cu)
set_target_properties(vector_add PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
if(BUILD_TESTS)
    add_test(NAME deep_learning_ch01_vector_add COMMAND vector_add)
endif()

list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_ROOT})
find_package(Torch QUIET)

if(Torch_FOUND)
    add_executable(lib_torch lib_torch.cpp)
    target_link_libraries(lib_torch PRIVATE ${TORCH_LIBRARIES})
    target_include_directories(lib_torch PRIVATE ${TORCH_INCLUDE_DIRS})

    add_executable(mini_net mini_net.cpp)
    target_link_libraries(mini_net PRIVATE ${TORCH_LIBRARIES})
    target_include_directories(mini_net PRIVATE ${TORCH_INCLUDE_DIRS})
else()
    message(WARNING "LibTorch not found -- skipping chapter-01 lib_torch and mini_net targets")
endif()
