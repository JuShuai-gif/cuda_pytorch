// Pitfall P9: checking huge pages actually took effect.
//
// Requesting THP via madvise(MADV_HUGEPAGE) is only a hint: the kernel may
// not allocate a 2 MB page (fragmentation, size, madvise mode). Always
// verify with /proc/self/smaps (AnonHugePages) whether huge pages were
// really used. This experiment allocates a large buffer and reports the
// AnonHugePages for its range.
//
// Related PDF: 7.5 / note/15 (huge pages, "hint, not guarantee").

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <sys/mman.h>
#include <unistd.h>

#include "cpu_info.h"

static constexpr size_t MB = 1024 * 1024;
static constexpr size_t TOTAL = 512 * MB;

static void touch(char* p, size_t n) {
    for (size_t i = 0; i < n; i += 4096) p[i] = 1;
}

int main() {
    std::printf("Pitfall P9: did huge pages actually take effect?\n");
    std::string thp = cpuinfo::thp_enabled();
    std::printf("THP: %s", thp.c_str());

    void* p = mmap(nullptr, TOTAL, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
        std::perror("mmap");
        return 1;
    }
    if (madvise(p, TOTAL, MADV_HUGEPAGE) == 0)
        std::printf("madvise(MADV_HUGEPAGE) ok\n");
    else
        std::perror("madvise");

    touch((char*)p, TOTAL);

    // Check the smaps entry for this mapping.
    FILE* f = fopen("/proc/self/smaps", "r");
    if (!f) { std::perror("smaps"); return 1; }
    char line[512];
    long anon_hp = 0, rss = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strstr(line, "AnonHugePages:")) {
            long kb = 0;
            sscanf(line, "AnonHugePages: %ld kB", &kb);
            anon_hp += kb;
        }
        if (strncmp(line, "Rss:", 4) == 0) {
            long kb = 0;
            sscanf(line, "Rss: %ld kB", &kb);
            rss += kb;
        }
    }
    fclose(f);
    std::printf("RSS (whole proc)        : %ld kB\n", rss);
    std::printf("AnonHugePages (whole)   : %ld kB\n", anon_hp);
    std::printf("\nLesson: MADV_HUGEPAGE is a hint. Verify with\n"
                "/proc/<pid>/smaps AnonHugePages; if it is 0, huge pages\n"
                "were not used (madvise mode, fragmentation, size).\n");
    munmap(p, TOTAL);
    return 0;
}
