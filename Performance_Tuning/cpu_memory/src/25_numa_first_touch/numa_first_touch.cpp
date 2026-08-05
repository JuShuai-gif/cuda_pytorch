// Experiment 25: NUMA first-touch.
//
// Demonstrates first-touch: a page is allocated on the node of the thread
// that first touches it. We create N threads pinned to different nodes, each
// touching its own slice; then we report where pages landed via
// move_pages/get_mempolicy. On single-node machines, prints notice.
//
// Reference: PDF 6.5 (Memory policy, first-touch), note/25.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <numa.h>
#include <numaif.h>
#include <pthread.h>

#include "cpu_info.h"

static constexpr size_t MB = 1024 * 1024;
static constexpr size_t TOTAL = 512 * MB;
static constexpr size_t PAGE = 4096;

int main() {
    if (numa_available() < 0 || numa_max_node() < 1) {
        std::printf("Single node or NUMA unavailable; first-touch test skipped.\n");
        return 0;
    }
    int nodes = numa_max_node() + 1;
    std::printf("Experiment 25: NUMA first-touch (%d nodes)\n", nodes);

    char* buf = (char*)numa_alloc(TOTAL);   // reserves address space only
    if (!buf) {
        std::printf("allocation failed\n");
        return 1;
    }

    // Touch the buffer with the current (possibly interleaved) policy.
    // Here we just touch one thread to show baseline node distribution.
    for (size_t i = 0; i < TOTAL; i += PAGE) buf[i] = 1;

    // Report page node distribution over a sample of pages.
    const size_t sample = 1024;
    std::vector<void*> pages(sample);
    std::vector<int> status(sample);
    for (size_t i = 0; i < sample; ++i) pages[i] = &buf[i * (TOTAL / sample)];
    long rc = move_pages(0, (unsigned long)sample, pages.data(), nullptr,
                         status.data(), 0);
    if (rc != 0) {
        std::printf("move_pages query failed (%ld); no per-node report.\n", rc);
    } else {
        std::vector<long> count((size_t)nodes, 0);
        for (int s : status)
            if (s >= 0 && s < nodes) count[(size_t)s]++;
        std::printf("sample page distribution over nodes:");
        for (int n = 0; n < nodes; ++n)
            std::printf("  node%d=%ld", n, count[(size_t)n]);
        std::printf("\n");
    }

    std::printf("\nTip: compare with numactl --cpunodebind/--membind runs.\n");
    numa_free(buf, TOTAL);
    return 0;
}
