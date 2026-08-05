#pragma once

// Compiler barriers and anti-optimization helpers.
// Note: the canonical implementations live in benchmark.h (bm::compiler_barrier,
// bm::do_not_optimize). This header keeps a thin alias for experiment code that
// prefers a standalone name.

#include "benchmark.h"

#define COMPILER_BARRIER() bm::compiler_barrier()
#define DO_NOT_OPTIMIZE(v) bm::do_not_optimize(v)
