// Experiment 01: Memory hierarchy latency via pointer chasing.
//
// A circular singly-linked list whose elements are laid out in a shuffled
// order. Each element is one cache line (64 bytes). Traversing the list
// forces a dependent load per element (pointer chasing), so hardware
// prefetching cannot hide the latency. Sweeping the working set size
// exposes the L1d / L2 / L3 / DRAM latency steps.
//
// Reference: PDF 3.3.2 (Measurements of Cache Effects), Figures 3.10-3.12.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <unistd.h>

#include "benchmark.h"

static constexpr size_t kMin = 1 << 10;      // 1 KiB
static constexpr size_t kMax = 1 << 27;      // 128 MiB
static constexpr int kRounds = 7;

struct alignas(64) Elem {
    Elem* next;
    char pad[64 - sizeof(Elem*)];
};
static_assert(sizeof(Elem) == 64, "elem must be one cache line");

static void build_list(std::vector<Elem>& pool, bool shuffled,
                       std::mt19937& rng) {
    size_t n = pool.size();
    for (size_t i = 0; i < n; ++i) {
        size_t j = shuffled ? (i + 1) % n : (i + 1) % n;
        (void)j;
        pool[i].next = &pool[(i + 1) % n];
    }
    if (shuffled) {
        // Shuffle the next pointers so traversal jumps randomly,
        // defeating the sequential prefetcher.
        std::vector<Elem*> perm(n);
        for (size_t i = 0; i < n; ++i) perm[i] = &pool[i];
        std::shuffle(perm.begin(), perm.end(), rng);
        for (size_t i = 0; i < n; ++i)
            perm[i]->next = perm[(i + 1) % n];
    }
}

static double run_traverse(const std::vector<Elem>& pool) {
    // Dependent load chain. ~3 passes over the list for stability.
    const Elem* cur = &pool[0];
    size_t n = pool.size();
    size_t reps = std::max<size_t>(3, (1u << 20) / std::max<size_t>(n, 1));
    uint64_t sum = 0;
    for (size_t r = 0; r < reps; ++r) {
        for (size_t i = 0; i < n; ++i) {
            sum += (uint64_t)(uintptr_t)cur;
            cur = cur->next;  // dependent load: next iteration needs this
        }
    }
    bm::do_not_optimize(sum);
    return static_cast<double>(reps) * static_cast<double>(n);
}

int main(int argc, char** argv) {
    bool shuffled = argc > 1 ? std::atoi(argv[1]) != 0 : true;
    std::printf("Experiment 01: memory latency (pointer chasing, %s order)\n",
                shuffled ? "random" : "sequential");
    std::printf("L1d line size: %ld bytes (detected)\n", ::sysconf(_SC_LEVEL1_DCACHE_LINESIZE));
    std::printf("%-14s %-12s %-12s\n", "workingset", "ns/elem", "checksum");
    std::fflush(stdout);

    for (size_t size = kMin; size <= kMax; size <<= 1) {
        size_t n = size / sizeof(Elem);
        std::vector<Elem> pool(n);
        std::mt19937 rng(42);
        build_list(pool, shuffled, rng);

        // Warmup
        for (int i = 0; i < 2; ++i) run_traverse(pool);

        auto res = bm::time_rounds(kRounds, [&] { run_traverse(pool); });

        double elements = static_cast<double>(n);
        // ns/element from median round time; the round does `reps` passes.
        // Recompute reps the same way as run_traverse.
        size_t reps = std::max<size_t>(3, (1u << 20) / std::max<size_t>(n, 1));
        double total = res.median_ms * 1e6;
        double per_elem = total / (double)(reps * n);

        std::printf("%-14zu %-12.2f %-12llu\n", size, per_elem,
                    (unsigned long long)(res.median_ms));
        std::fflush(stdout);
    }
    return 0;
}
