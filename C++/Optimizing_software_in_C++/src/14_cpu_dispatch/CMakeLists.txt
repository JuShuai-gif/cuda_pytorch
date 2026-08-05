add_executable(14_cpu_dispatch dispatch.cpp impl_avx2.cpp)
target_link_libraries(14_cpu_dispatch PRIVATE oopt_common)
# Only impl_avx2.cpp is compiled with -mavx2; dispatch.cpp stays portable.
set_source_files_properties(impl_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2")
