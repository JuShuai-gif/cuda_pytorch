#include "benchmark.h"

#include <algorithm>

namespace bm {

// Implementation lives inline in the header for simplicity; this TU
// exists so the common library links cleanly with -Wl,--no-undefined.
void benchmark_keep_symbol() {
    static volatile int s = 0;
    s += static_cast<int>(sizeof(BenchmarkResult));
    do_not_optimize(s);
}

}  // namespace bm
