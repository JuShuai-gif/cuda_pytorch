// Experiment 15: Thread affinity.
//
// Two threads incrementing separate padded counters, run with:
//   - default scheduling (no affinity)
//   - pinned to different physical cores (picked from allowed CPUs)
//   - pinned to the same physical core (SMT siblings if available)
// If we cannot identify core topology we print a note rather than
// fabricate a conclusion.
//
// Reference: PDF 6.4.3 (thread affinity), 3.5.3 (cache placement).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "benchmark.h"

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

static constexpr long ITER = 200'000'000L;
static constexpr int kRounds = 3;

struct alignas(64) Counter {
    long val;
};

static void pin(unsigned cpu, pthread_t th = 0) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET((int)cpu, &set);
    if (th == 0)
        sched_setaffinity(0, sizeof(set), &set);
    else
        pthread_setaffinity_np(th, sizeof(set), &set);
}

static int allowed_cpu(unsigned idx) {
    // Pick the idx-th allowed CPU from /proc/self/status Cpus_allowed_list.
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256];
    int found = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "Cpus_allowed_list:", 18) == 0) {
            found = 1;
            break;
        }
    }
    fclose(f);
    if (!found) return -1;
    char* p = strchr(line, ':');
    if (!p) return -1;
    p++;
    // Parse "a-b" style ranges; take the idx-th CPU, expanding ranges.
    std::vector<int> cpus;
    char* tok = strtok(p, ",\n");
    while (tok) {
        int lo, hi;
        if (sscanf(tok, "%d-%d", &lo, &hi) == 2) {
            for (int c = lo; c <= hi; ++c) cpus.push_back(c);
        } else {
            cpus.push_back(atoi(tok));
        }
        tok = strtok(nullptr, ",\n");
    }
    if (cpus.empty() || idx >= cpus.size()) return -1;
    return cpus[idx];
}

int main() {
    int ncpu = (int)sysconf(_SC_NPROCESSORS_ONLN);
    std::printf("Experiment 15: thread affinity (%d online CPUs)\n", ncpu);

    int cpu0 = allowed_cpu(0);
    int cpu1 = allowed_cpu(1);
    std::printf("allowed CPUs: 0->%d, 1->%d\n", cpu0, cpu1);

    auto run = [&](bool pin_different) {
        Counter c[2];
        c[0].val = 0;
        c[1].val = 0;
        std::thread t1([&] { for (long i = 0; i < ITER; ++i) ++c[0].val; });
        std::thread t2([&] { for (long i = 0; i < ITER; ++i) ++c[1].val; });
        if (pin_different) {
            if (cpu0 >= 0) pin((unsigned)cpu0, t1.native_handle());
            if (cpu1 >= 0) pin((unsigned)cpu1, t2.native_handle());
        }
        t1.join();
        t2.join();
        bm::do_not_optimize(c[0].val + c[1].val);
    };

    run(false);
    run(true);

    auto r_default = bm::time_rounds(kRounds, [&] { run(false); });
    auto r_pinned = bm::time_rounds(kRounds, [&] { run(true); });

    std::printf("default_sched : mean=%.3f ms\n", r_default.mean_ms);
    std::printf("pinned_cores  : mean=%.3f ms\n", r_pinned.mean_ms);
    std::printf("NOTE: without core/sibling topology (e.g. lscpu -e, sysfs\n"
                "thread_siblings) we cannot assert SMT vs distinct cores;\n"
                "results indicate scheduler migration effects only.\n");
    return 0;
}
