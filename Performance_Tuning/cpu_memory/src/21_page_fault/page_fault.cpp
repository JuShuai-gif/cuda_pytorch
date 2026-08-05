// Experiment 21: Page faults.
//
// Compares:
//   1. malloc + no touch        (zero page faults)
//   2. malloc + first touch     (one minor fault per page)
//   3. second touch             (no new faults)
//   4. mmap + MAP_POPULATE      (faults during mmap)
// Reports minor/major fault deltas via getrusage.
//
// Reference: PDF 7.5 (Page Fault Optimization).

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

static constexpr size_t MB = 1024 * 1024;
static constexpr size_t TOTAL = 128 * MB;

static long minflt() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_minflt;
}

int main() {
    std::printf("Experiment 21: page faults (buffer %zu MB)\n", TOTAL / MB);
    long page = sysconf(_SC_PAGESIZE);
    std::printf("page size: %ld\n", page);

    // 1) malloc without touch
    {
        long before = minflt();
        char* p = (char*)malloc(TOTAL);
        long after = minflt();
        std::printf("malloc no-touch  : minor faults=%ld\n", after - before);
        // leave un-touched; free later
        free(p);
    }

    // 2) malloc + first touch
    long first_faults = 0;
    {
        long before = minflt();
        char* p = (char*)malloc(TOTAL);
        for (size_t i = 0; i < TOTAL; i += page) p[i] = 1;
        first_faults = minflt() - before;
        std::printf("malloc first-touch: minor faults=%ld\n", first_faults);

        // 3) second touch (should add ~0)
        long b2 = minflt();
        for (size_t i = 0; i < TOTAL; i += page) p[i] = 2;
        std::printf("second touch      : minor faults=%ld\n", minflt() - b2);
        free(p);
    }

    // 4) mmap + MAP_POPULATE
    {
        long before = minflt();
        void* p = mmap(nullptr, TOTAL, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
        long during = minflt() - before;
        long b2 = minflt();
        for (size_t i = 0; i < TOTAL; i += page)
            ((char*)p)[i] = 3;   // pages already present
        std::printf("MAP_POPULATE     : minor faults in mmap=%ld, touch=%ld\n",
                    during, minflt() - b2);
        munmap(p, TOTAL);
    }
    return 0;
}
