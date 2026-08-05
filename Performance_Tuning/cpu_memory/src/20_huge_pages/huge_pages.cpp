// Experiment 20: Huge pages (THP / hugetlbfs / HugeTLB).
//
// 1) Reports THP mode and huge page availability.
// 2) Allocates a large buffer with mmap + MADV_HUGEPAGE (transparent huge
//    pages) when the kernel supports it.
// 3) Optionally tries MAP_HUGETLB (explicit huge pages).
// Never assumes huge pages exist; prints clear notices when unavailable.
//
// Reference: PDF 6.2.4, 7.5, note/15.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "benchmark.h"
#include "cpu_info.h"

#include <sys/mman.h>
#include <unistd.h>

static constexpr size_t MB = 1024 * 1024;
static constexpr size_t TOTAL = 512 * MB;

int main() {
    std::printf("Experiment 20: huge pages\n");
    std::printf("THP enabled: %s", cpuinfo::thp_enabled().c_str());
    std::printf("Huge page size: %ld bytes, preallocated: %ld\n",
                cpuinfo::huge_page_size(), cpuinfo::huge_pages_total());

    // 1) Baseline: 4K anonymous pages, sequential write.
    std::vector<char> base(TOTAL, 0);
    auto write_all = [](char* p, size_t n) {
        for (size_t i = 0; i < n; ++i) p[i] = 1;
        bm::compiler_barrier();
    };
    write_all(base.data(), TOTAL);
    auto r_base = bm::time_rounds(3, [&] { write_all(base.data(), TOTAL); });
    std::printf("4K pages sequential write: mean=%.3f ms\n", r_base.mean_ms);

    // 2) THP via madvise(MADV_HUGEPAGE).
    std::string thp = cpuinfo::thp_enabled();
    bool thp_possible = thp.find("madvise") != std::string::npos ||
                        thp.find("always") != std::string::npos;
    if (thp_possible) {
        void* p = mmap(nullptr, TOTAL, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p != MAP_FAILED) {
            if (madvise(p, TOTAL, MADV_HUGEPAGE) == 0) {
                write_all(static_cast<char*>(p), TOTAL);
                auto r = bm::time_rounds(3, [&] {
                    write_all(static_cast<char*>(p), TOTAL);
                });
                std::printf("madvise(MADV_HUGEPAGE) write: mean=%.3f ms\n",
                            r.mean_ms);
            } else {
                std::printf("madvise(MADV_HUGEPAGE) failed\n");
            }
            munmap(p, TOTAL);
        }
    } else {
        std::printf("THP not usable (enabled=%s), skipping madvise test.\n",
                    thp.c_str());
    }

    // 3) Explicit HugeTLB via MAP_HUGETLB (only if pages preallocated).
    if (cpuinfo::huge_pages_total() > 0) {
        void* p = mmap(nullptr, TOTAL, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        if (p != MAP_FAILED) {
            write_all(static_cast<char*>(p), TOTAL);
            auto r = bm::time_rounds(3, [&] {
                write_all(static_cast<char*>(p), TOTAL);
            });
            std::printf("MAP_HUGETLB (explicit) write: mean=%.3f ms\n", r.mean_ms);
            munmap(p, TOTAL);
        } else {
            std::printf("MAP_HUGETLB failed: %s\n", strerror(errno));
        }
    } else {
        std::printf("No preallocated huge pages (HugePages_Total=0); explicit\n"
                    "HugeTLB test skipped. Enable with (root):\n"
                    "  sysctl vm.nr_hugepages=<N>\n"
                    "or configure THP=always/madvise.\n");
    }
    return 0;
}
