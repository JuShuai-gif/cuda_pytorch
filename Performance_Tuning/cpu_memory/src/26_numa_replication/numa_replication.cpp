// Experiment 26: NUMA replication.
//
// For read-only data accessed by threads on different nodes, replicating a
// per-node copy (via numa_alloc_onnode for each node) avoids inter-node
// reads. Compares access to a single shared copy vs per-node replicated
// copies on a multi-node machine. Single-node machines print notice.
//
// Reference: PDF 6.5.7 (Explicit NUMA optimizations), note/25.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <numa.h>

#include "benchmark.h"

static constexpr size_t MB = 1024 * 1024;
static constexpr size_t DATA = 128 * MB;

int main() {
    if (numa_available() < 0 || numa_max_node() < 1) {
        std::printf("Single node or NUMA unavailable; replication test skipped.\n");
        return 0;
    }
    int nodes = numa_max_node() + 1;
    std::printf("Experiment 26: NUMA replication (%d nodes)\n", nodes);

    // Shared single copy (allocated on current node).
    char* shared = (char*)numa_alloc(DATA);
    for (size_t i = 0; i < DATA; i += 4096) shared[i] = 1;

    auto read_shared = [&] {
        volatile char sink = 0;
        for (size_t r = 0; r < 4; ++r)
            for (size_t i = 0; i < DATA; i += 4096) sink += shared[i];
        bm::do_not_optimize(sink);
    };
    read_shared();
    auto r_s = bm::time_rounds(3, read_shared);

    // Replicated: one copy per node.
    std::vector<char*> copies;
    for (int n = 0; n < nodes; ++n) {
        if (numa_bitmask_isbitset(numa_nodes_ptr, n) == 0) continue;
        char* p = (char*)numa_alloc_onnode(DATA, n);
        for (size_t i = 0; i < DATA; i += 4096) p[i] = 1;
        copies.push_back(p);
    }
    auto read_replicated = [&] {
        volatile char sink = 0;
        for (int n = 0; n < (int)copies.size(); ++n)
            for (size_t i = 0; i < DATA; i += 4096) sink += copies[(size_t)n][i];
        bm::do_not_optimize(sink);
    };
    read_replicated();
    auto r_r = bm::time_rounds(3, read_replicated);

    std::printf("shared single copy : mean=%.3f ms\n", r_s.mean_ms);
    std::printf("per-node copies    : mean=%.3f ms\n", r_r.mean_ms);
    std::printf("NOTE: the single-copy thread ran on the allocation node;\n"
                "inter-node costs are best measured with pinned threads.\n");

    for (char* p : copies) numa_free(p, DATA);
    numa_free(shared, DATA);
    return 0;
}
