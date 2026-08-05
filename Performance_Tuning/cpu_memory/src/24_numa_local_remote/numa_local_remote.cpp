// Experiment 24: NUMA local vs remote access.
//
// Requires libnuma and a multi-node machine. Allocates a buffer, binds to
// a node, and measures sequential write bandwidth with local vs remote
// memory binding (numa_alloc_onnode). On a single-node machine it prints a
// notice and exits 0 (no fabricated data).
//
// Reference: PDF 5.4 (Remote Access Costs, Figures 5.3-5.4).

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <numa.h>
#include <numaif.h>

#include "benchmark.h"
#include "cpu_info.h"

static constexpr size_t MB = 1024 * 1024;
static constexpr size_t TOTAL = 512 * MB;

int main() {
    if (numa_available() < 0) {
        std::printf("NUMA not available; skipping.\n");
        return 0;
    }
    int nodes = numa_max_node() + 1;
    std::printf("Experiment 24: NUMA local/remote (%d nodes)\n", nodes);
    if (nodes < 2) {
        std::printf("Single NUMA node only; remote-access test skipped.\n");
        return 0;
    }

    for (int node = 0; node < nodes; ++node) {
        if (numa_bitmask_isbitset(numa_nodes_ptr, node) == 0) continue;
        char* buf = (char*)numa_alloc_onnode(TOTAL, node);
        if (!buf) {
            std::printf("node%d: allocation failed\n", node);
            continue;
        }
        // Touch to force allocation on the node.
        for (size_t i = 0; i < TOTAL; i += 4096) buf[i] = 1;

        for (int cpu_node = 0; cpu_node < nodes; ++cpu_node) {
            if (numa_bitmask_isbitset(numa_nodes_ptr, cpu_node) == 0) continue;
            struct bitmask* old = numa_get_mems_allowed();
            struct bitmask* use = numa_allocate_nodemask();
            numa_bitmask_setbit(use, cpu_node);
            numa_set_membind(use);
            numa_free_nodemask(use);

            auto fn = [&] {
                for (size_t i = 0; i < TOTAL; ++i) buf[i] = (char)i;
                bm::compiler_barrier();
            };
            fn();
            auto res = bm::time_rounds(3, fn);
            double gbps = (double)TOTAL / (res.median_ms * 1e6 / 1e9) / 1e9;

            // Cpu node here is the membind; access cost depends on the
            // memory node vs the current CPU, which we approximate by
            // the bind node.
            std::printf("mem on node%d, cpu-node%d: mean=%.3f ms (%.2f GB/s)\n",
                        node, cpu_node, res.mean_ms, gbps);

            numa_set_membind(old);
            numa_free_nodemask(old);
        }
        numa_free(buf, TOTAL);
    }
    return 0;
}
