add_library(oopt_common STATIC benchmark.cpp cpu_info.cpp)
# Expose the src/ directory so targets can `#include "common/benchmark.h"`.
target_include_directories(oopt_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
