// Experiment 19: Page size effects.
//
// Measures the cost of touching one byte per page for 4 KB pages, and
// compares with huge pages (2 MB) when available via mmap MAP_HUGETLB.
// If huge pages are not configured, prints a clear notice and skips.
//
// Reference: PDF 6.2.4 (TLB usage), 7.5 (page faults, Figure 7.9).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "benchmark.h"
#include "cpu_info.h"

#include <sys/mman.h>
#include <unistd.h>

static constexpr size_t MB = 1024 * 1024;

int main() {
    std::printf("Experiment 19: page size effects\n");
    long page = cpuinfo::page_size();
    long huge = cpuinfo::huge_page_size();
    std::printf("4K page size: %ld\n", page);
    std::printf("huge page size: %ld, preallocated: %ld\n",
                huge, cpuinfo::huge_pages_total());

    const size_t TOTAL = 256 * MB;

    // 4K pages: touch one byte per page.
    size_t np4 = TOTAL / (size_t)page;
    std::vector<char> buf4(TOTAL, 1);
    auto touch_4k = [&] {
        volatile char sink = 0;
        for (size_t i = 0; i < np4; ++i) sink += buf4[i * (size_t)page];
        bm::do_not_optimize(sink);
    };
    touch_4k();
    auto r4 = bm::time_rounds(5, touch_4k);
    std::printf("4K pages: %zu pages, mean=%.3f ms\n", np4, r4.mean_ms);

    // Huge pages (2 MB) if available.
    bool have_huge = huge > 0 && cpuinfo::huge_pages_total() > 0;
    if (have_huge) {
        size_t nhu = TOTAL / (size_t)huge;
        void* p = mmap(nullptr, TOTAL, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        if (p == MAP_FAILED) {
            std::printf("MAP_HUGETLB failed: %s (no huge pages free?)\n",
                        strerror(errno));
            have_huge = false;
        } else {
            char* buf = static_cast<char*>(p);
            auto touch_huge = [&] {
                volatile char sink = 0;
                for (size_t i = 0; i < nhu; ++i) sink += buf[i * (size_t)huge];
                bm::do_not_optimize(sink);
            };
            touch_huge();
            auto rh = bm::time_rounds(5, touch_huge);
            std::printf("2M pages: %zu pages, mean=%.3f ms\n", nhu, rh.mean_ms);
            munmap(p, TOTAL);
        }
    }
    if (!have_huge)
        std::printf("Skipping huge-page test: system has no preallocated huge\n"
                    "pages. See note/15 on enabling (needs root/hugetlbfs).\n");
    return 0;
}
