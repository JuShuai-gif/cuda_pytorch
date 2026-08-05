// Experiment 05: Cache capacity.
//
// Sweeps the working set size from 4 KiB to 128 MiB using sequential
// pointer-chasing over one-element-per-cache-line nodes. The average
// latency jumps when the working set no longer fits L1d / L2 / L3,
// exposing the cache hierarchy sizes.
//
// Reference: PDF 3.3.2 (Figures 3.10-3.12).

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "benchmark.h"

struct alignas(64) Elem {
    Elem* next;
    char pad[64 - sizeof(Elem*)];
};
static_assert(sizeof(Elem) == 64, "one cache line per element");

static constexpr int kRounds = 5;

static void build_seq(std::vector<Elem>& v) {
    for (size_t i = 0; i < v.size(); ++i) v[i].next = &v[(i + 1) % v.size()];
}

static void build_rand(std::vector<Elem>& v, std::mt19937& rng) {
    size_t n = v.size();
    std::vector<Elem*> perm(n);
    for (size_t i = 0; i < n; ++i) perm[i] = &v[i];
    std::shuffle(perm.begin(), perm.end(), rng);
    for (size_t i = 0; i < n; ++i) perm[i]->next = perm[(i + 1) % n];
}

// Returns ns/element (median of rounds).
static double measure(const std::vector<Elem>& v) {
    const Elem* cur = &v[0];
    size_t n = v.size();
    size_t reps = std::max<size_t>(4, (1u << 20) / std::max<size_t>(n, 1));
    auto fn = [&] {
        uint64_t s = 0;
        for (size_t r = 0; r < reps; ++r)
            for (size_t i = 0; i < n; ++i) {
                s += (uint64_t)(uintptr_t)cur;
                cur = cur->next;
            }
        bm::do_not_optimize(s);
    };
    fn();  // warmup
    auto res = bm::time_rounds(kRounds, fn);
    return res.median_ms * 1e6 / (double)(reps * n);
}

int main() {
    std::printf("Experiment 05: cache capacity (sequential pointer chase)\n");
    std::printf("%-14s %-12s\n", "workingset", "ns/elem");

    for (size_t size = 4u << 10; size <= (128u << 20); size <<= 1) {
        size_t n = size / sizeof(Elem);
        std::vector<Elem> pool(n);
        build_seq(pool);
        double ns = measure(pool);
        std::printf("%-14zu %-12.2f\n", size, ns);
    }
    return 0;
}
