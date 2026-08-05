// Experiment 23: Memory bandwidth (STREAM-like).
//
// Implements Copy, Scale, Add, Triad on arrays that exceed the last-level
// cache. Reports time and effective bandwidth GB/s. Also runs with
// 1..N threads to reveal the memory-bandwidth wall.
//
// Reference: PDF 3.5.1, note/11.

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include "benchmark.h"

static constexpr size_t N = 1 << 26;   // 64M doubles = 512 MB each
static constexpr int kRounds = 3;

static double* a;
static double* b;
static double* c;

static void init() {
    a = new double[N];
    b = new double[N];
    c = new double[N];
    for (size_t i = 0; i < N; ++i) {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 0.0;
    }
}

// Work on [lo, hi), each thread its own slice.
static void do_copy(size_t lo, size_t hi) {
    for (size_t i = lo; i < hi; ++i) c[i] = a[i];
}
static void do_scale(size_t lo, size_t hi) {
    for (size_t i = lo; i < hi; ++i) b[i] = 3.0 * a[i];
}
static void do_add(size_t lo, size_t hi) {
    for (size_t i = lo; i < hi; ++i) c[i] = a[i] + b[i];
}
static void do_triad(size_t lo, size_t hi) {
    for (size_t i = lo; i < hi; ++i) a[i] = b[i] + 3.0 * c[i];
}

static double run(unsigned threads, int op) {
    std::vector<std::thread> pool;
    std::atomic<long long> sink{0};
    size_t per = N / threads;
    for (unsigned t = 0; t < threads; ++t) {
        size_t lo = per * t, hi = (t + 1 == threads) ? N : per * (t + 1);
        pool.emplace_back([&, lo, hi, op] {
            switch (op) {
                case 0: do_copy(lo, hi); break;
                case 1: do_scale(lo, hi); break;
                case 2: do_add(lo, hi); break;
                default: do_triad(lo, hi); break;
            }
            sink.fetch_add(1, std::memory_order_relaxed);
        });
    }
    for (auto& th : pool) th.join();
    bm::do_not_optimize(sink.load());
    return 0;
}

int main() {
    init();
    unsigned max_t = std::thread::hardware_concurrency();
    if (max_t > 8) max_t = 8;

    std::printf("Experiment 23: memory bandwidth (arrays %zu MB each)\n",
                N * sizeof(double) / (1024 * 1024));

    const char* names[] = {"Copy", "Scale", "Add", "Triad"};
    double bytes_per_elem[] = {16.0, 24.0, 24.0, 32.0};  // read+write bytes

    for (unsigned t = 1; t <= max_t; t *= 2) {
        std::printf("--- threads=%u ---\n", t);
        for (int op = 0; op < 4; ++op) {
            // warmup
            run(t, op);
            auto res = bm::time_rounds(kRounds, [&] { run(t, op); });
            double gbps = (double)N * bytes_per_elem[op] /
                          (res.median_ms * 1e6 / 1e9) / 1e9;
            std::printf("%-6s median=%-10.3f ms  %-8.2f GB/s\n", names[op],
                        res.median_ms, gbps);
        }
    }
    return 0;
}
