// 18_benchmark: data size too small -- everything fits in cache, so cache
// optimizations look pointless. Rerun with sizes spanning the cache levels.
//
// PDF 16.2 (p170): a unit test with tiny data hides cache-miss penalties.
#include <cstdio>
#include <string>
#include <vector>

#include "common/benchmark.h"

int main() {
    // All three sizes fit comfortably in this machine's caches
    // (L1d ~48KB/core, L2 32MB, L3 36MB). Sweep past L2/L3 to see the jump.
    for (size_t bytes : {size_t(64u << 10), size_t(1u << 20),
                         size_t(8u << 20), size_t(128u << 20)}) {
        size_t n = bytes / sizeof(int);
        std::vector<int> v(n, 1);
        bench(("sum  " + std::to_string(bytes / (1024 * 1024)) + " MiB").c_str(),
              [&] { long long s = 0; for (int x : v) s += x; return s; });
    }
    return 0;
}
