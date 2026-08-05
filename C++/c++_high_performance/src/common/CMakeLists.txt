# Shared infrastructure used by every chapter's examples.
#
#  - benchmark.hpp/cpp      unified micro-benchmark runner + result printing
#  - statistics.hpp/cpp     mean/median/min/max/stddev over time samples
#  - compiler_barrier.hpp   compile-time barrier to defeat optimization
#  - system_info.hpp/cpp    OS / CPU / compiler description
#  - test_utils.hpp         lightweight CHP_CHECK assertion helpers

add_library(chp_common STATIC
    benchmark.cpp
    statistics.cpp
    system_info.cpp
)

target_include_directories(chp_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(chp_common PUBLIC Threads::Threads)
