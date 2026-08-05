// Pitfall P2: dead-code elimination in benchmarks.
//
// A "benchmark" loop whose result is never observed may be optimized away
// entirely, making the measurement useless (looks instant and "too fast").
// Also, a simple sum can be replaced by a closed-form (Gauss) formula by the
// compiler, so even a "kept" result may hide all the work. Both are reasons
// to write the workload so its work cannot be elided (write to a buffer).
//
// Related PDF: quality requirements (this project), benchmark correctness.

#include <cstdint>
#include <cstdio>
#include <vector>

#include "benchmark.h"

static uint64_t loop_optimized_away(uint64_t n) {
    uint64_t s = 0;
    for (uint64_t i = 0; i < n; ++i) s += i * 3;
    return s;  // return value never used at call site
}

static uint64_t loop_closed_form(uint64_t n) {
    uint64_t s = 0;
    for (uint64_t i = 0; i < n; ++i) s += i * 3;
    bm::do_not_optimize(s);  // result observed, but compiler may use closed form
    return s;
}

// Write into a buffer so the work cannot be replaced by a closed form.
static uint64_t loop_real_work(uint64_t n, int* buf) {
    uint64_t s = 0;
    for (uint64_t i = 0; i < n; ++i) buf[i] = static_cast<int>(i & 0x7fffffff);
    for (uint64_t i = 0; i < n; ++i) s += (uint64_t)buf[i];
    return s;
}

int main() {
    constexpr uint64_t N = 1u << 26;
    constexpr int kRounds = 3;
    std::vector<int> buf((size_t)N);

    std::printf("Pitfall P2: dead-code elimination in benchmarks\n");

    auto r_gone = bm::time_rounds(kRounds, [] { loop_optimized_away(N); });
    auto r_form = bm::time_rounds(kRounds, [] { loop_closed_form(N); });
    auto r_real = bm::time_rounds(kRounds, [&] {
        bm::do_not_optimize(loop_real_work(N, buf.data()));
    });

    std::printf("loop (result unused)          : mean=%.4f ms (elided?)\n",
                r_gone.mean_ms);
    std::printf("loop (closed form possible)   : mean=%.4f ms (suspicious)\n",
                r_form.mean_ms);
    std::printf("loop (real memory work)       : mean=%.2f ms\n", r_real.mean_ms);
    std::printf("\nLesson: a benchmark that is suspiciously fast probably got\n"
                "optimized away or replaced by a closed form. Keep real,\n"
                "observable side effects (buffer writes + checksum).\n");
    return 0;
}
