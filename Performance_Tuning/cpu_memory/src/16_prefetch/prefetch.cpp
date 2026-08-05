// Experiment 16: Prefetching.
//
// Random-order pointer chasing over one-cache-line nodes. Compares:
//   - no prefetch
//   - software prefetch (_mm_prefetch) at various distances
// Reports ns/element. Software prefetch may or may not help depending on
// the CPU; we do not assume a specific speedup.
//
// Reference: PDF 6.3.2 (Figure 6.7).

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "benchmark.h"
#include "cpu_info.h"

#include <xmmintrin.h>

struct alignas(64) Elem {
    Elem* next;
    char pad[64 - sizeof(Elem*)];
};
static_assert(sizeof(Elem) == 64, "one cache line per element");

static constexpr int kRounds = 5;

static std::vector<Elem> build(int n, std::mt19937& rng) {
    std::vector<Elem> pool((size_t)n);
    std::vector<Elem*> perm((size_t)n);
    for (int i = 0; i < n; ++i) perm[(size_t)i] = &pool[(size_t)i];
    std::shuffle(perm.begin(), perm.end(), rng);
    for (int i = 0; i < n; ++i) perm[(size_t)i]->next = perm[(size_t)((i + 1) % n)];
    return pool;
}

static double measure(const std::vector<Elem>& v, int dist) {
    int n = (int)v.size();
    const Elem* cur = &v[0];
    auto fn = [&] {
        uint64_t s = 0;
        for (int i = 0; i < n; ++i) {
            if (dist > 0 && i + dist < n) {
                // prefetch node dist ahead
                _mm_prefetch((const char*)cur->next, _MM_HINT_T0);
            }
            s += (uint64_t)(uintptr_t)cur;
            cur = cur->next;
        }
        bm::do_not_optimize(s);
    };
    fn();
    auto res = bm::time_rounds(kRounds, fn);
    return res.median_ms * 1e6 / (double)n;
}

int main() {
    long line = cpuinfo::l1d_line_size();
    std::printf("Experiment 16: prefetch (line=%ld)\n", line);

    for (int n : {1 << 16, 1 << 18, 1 << 20}) {
        std::mt19937 rng(42);
        auto pool = build(n, rng);
        std::printf("nodes=%d\n", n);
        std::printf("%-10s %-12s\n", "distance", "ns/elem");
        for (int dist : {0, 2, 4, 8, 16}) {
            double ns = measure(pool, dist);
            std::printf("%-10d %-12.2f\n", dist, ns);
        }
    }
    return 0;
}
