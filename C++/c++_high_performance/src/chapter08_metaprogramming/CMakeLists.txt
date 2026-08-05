# Chapter 8: Metaprogramming and Compile-Time Evaluation

# --- templates_basics: template functions/classes, non-type params ---
add_executable(ch08_templates_example templates_basics/example.cpp)
if(ENABLE_TESTS)
    add_executable(ch08_templates_tests templates_basics/tests.cpp)
    target_link_libraries(ch08_templates_tests PRIVATE chp_common)
endif()

# --- type_traits: traits, decltype, enable_if, is_detected ---
add_executable(ch08_type_traits_example type_traits/example.cpp)
if(ENABLE_TESTS)
    add_executable(ch08_type_traits_tests type_traits/tests.cpp)
    target_link_libraries(ch08_type_traits_tests PRIVATE chp_common)
endif()

# --- constexpr_compute: constexpr / if constexpr / compile-time verify ---
add_executable(ch08_constexpr_example constexpr_compute/example.cpp)
if(ENABLE_TESTS)
    add_executable(ch08_constexpr_tests constexpr_compute/tests.cpp)
    target_link_libraries(ch08_constexpr_tests PRIVATE chp_common)
endif()

# --- tuple_variadic: tuple, structured bindings, variadic packs ---
add_executable(ch08_tuple_example tuple_variadic/example.cpp)
if(ENABLE_TESTS)
    add_executable(ch08_tuple_tests tuple_variadic/tests.cpp)
    target_link_libraries(ch08_tuple_tests PRIVATE chp_common)
endif()

# --- heterogeneous_containers: vector<any> vs vector<variant> ---
add_executable(ch08_heterogeneous_example
               heterogeneous_containers/example.cpp)
if(ENABLE_TESTS)
    add_executable(ch08_heterogeneous_tests
                   heterogeneous_containers/tests.cpp)
    target_link_libraries(ch08_heterogeneous_tests PRIVATE chp_common)
endif()

# --- reflection: reflect() + generic operators ---
add_executable(ch08_reflection_example reflection/example.cpp)
if(ENABLE_TESTS)
    add_executable(ch08_reflection_tests reflection/tests.cpp)
    target_link_libraries(ch08_reflection_tests PRIVATE chp_common)
endif()

# --- safe_cast: checked casts with if constexpr ---
add_executable(ch08_safe_cast_example safe_cast/example.cpp)

# --- compile_time_hash: PrehashedString ---
add_executable(ch08_compile_time_hash_example compile_time_hash/example.cpp)
