# Chapter 7: Memory Management

# --- stack_vs_heap: growth direction and limits ---
add_executable(ch07_stack_vs_heap_example stack_vs_heap/example.cpp)

# --- new_delete_placement: new/delete, placement new, <memory> ---
add_executable(ch07_new_delete_example new_delete_placement/example.cpp)
if(ENABLE_TESTS)
    add_executable(ch07_new_delete_tests new_delete_placement/tests.cpp)
    target_link_libraries(ch07_new_delete_tests PRIVATE chp_common)
endif()

# --- alignment_padding: alignof, padding, member reorder ---
add_executable(ch07_alignment_example alignment_padding/example.cpp)
if(ENABLE_TESTS)
    add_executable(ch07_alignment_tests alignment_padding/tests.cpp)
    target_link_libraries(ch07_alignment_tests PRIVATE chp_common)
endif()

# --- smart_pointers: unique/shared/weak, make_shared ---
add_executable(ch07_smart_pointers_example smart_pointers/example.cpp)
if(ENABLE_TESTS)
    add_executable(ch07_smart_pointers_tests smart_pointers/tests.cpp)
    target_link_libraries(ch07_smart_pointers_tests PRIVATE chp_common)
endif()

# --- small_string: SSO boundary ---
add_executable(ch07_small_string_example small_string/example.cpp)

# --- arena_allocator: Arena + ShortAlloc ---
add_executable(ch07_arena_example arena_allocator/example.cpp
               arena_allocator/arena.hpp)
if(ENABLE_BENCHMARKS)
    add_executable(ch07_arena_benchmark arena_allocator/benchmark.cpp
                   arena_allocator/arena.hpp)
    target_link_libraries(ch07_arena_benchmark PRIVATE chp_common)
endif()
if(ENABLE_TESTS)
    add_executable(ch07_arena_tests arena_allocator/tests.cpp
                   arena_allocator/arena.hpp)
    target_link_libraries(ch07_arena_tests PRIVATE chp_common)
endif()
