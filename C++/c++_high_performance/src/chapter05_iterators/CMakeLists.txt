# Chapter 5: A Deeper Look at Iterators

# --- int_iterator: custom generators ---
add_executable(ch05_int_iterator_example int_iterator/example.cpp
               int_iterator/iterators.hpp)
if(ENABLE_TESTS)
    add_executable(ch05_int_iterator_tests int_iterator/tests.cpp
                   int_iterator/iterators.hpp)
    target_link_libraries(ch05_int_iterator_tests PRIVATE chp_common)
endif()

# --- linear_range: floating point linear ranges ---
add_executable(ch05_linear_range_example linear_range/example.cpp
               linear_range/linear_range.hpp)
if(ENABLE_TESTS)
    add_executable(ch05_linear_range_tests linear_range/tests.cpp
                   linear_range/linear_range.hpp)
    target_link_libraries(ch05_linear_range_tests PRIVATE chp_common)
endif()

# --- iterator_distance: compile-time dispatch on category ---
add_executable(ch05_iterator_distance_example
               iterator_distance/example.cpp)

# --- iterator_categories: category tags and traits ---
add_executable(ch05_iterator_categories_example
               iterator_categories/example.cpp)
if(ENABLE_TESTS)
    add_executable(ch05_iterator_categories_tests
                   iterator_categories/tests.cpp)
    target_link_libraries(ch05_iterator_categories_tests PRIVATE chp_common)
endif()
