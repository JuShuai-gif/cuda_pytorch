// Experiment 08: Write-back behavior.
//
// Compares "scattered writes" (touch only part of each cache line, forcing
// write-allocate of the full line) vs "full-line writes". Also compares a
// read-modify-write pattern against a pure streaming write. On a write-back
// cache, full-line sequential writes are expected to be faster.
//
// Reference: PDF 3.3.3 (Write Behavior).

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

#include "benchmark.h"

static constexpr size_t N = 1 << 26;   // 64M ints = 256 MB
static constexpr int kRounds = 5;

int main() {
    std::printf("Experiment 08: write-back behavior (array %zu MB)\n",
                N * sizeof(int) / (1024 * 1024));
    std::vector<int> data(N, 0);

    // Mode 1: write every element sequentially (full line use).
    auto full_write = [&] {
        for (size_t i = 0; i < N; ++i) data[i] = 1;
        bm::compiler_barrier();
    };
    // Mode 2: write every other element (partial line, write-allocate).
    auto scattered_write = [&] {
        for (size_t i = 0; i < N; i += 2) data[i] = 1;
        bm::compiler_barrier();
    };
    // Mode 3: read-modify-write (read + write same element).
    auto rmw = [&] {
        for (size_t i = 0; i < N; ++i) data[i] += 1;
        bm::compiler_barrier();
    };

    struct Mode {
        const char* name;
        std::function<void()> fn;
    };
    Mode modes[] = {
        {"full_seq_write", full_write},
        {"scattered_write", scattered_write},
        {"read_modify_write", rmw},
    };

    std::printf("%-20s %-12s %-14s\n", "mode", "time_ms", "GB/s(effective)");
    for (auto& m : modes) {
        m.fn();  // warmup
        auto res = bm::time_rounds(kRounds, m.fn);
        double bytes = (double)N * sizeof(int);
        double gbps = bytes / (res.median_ms * 1e6 / 1e9) / 1e9;
        std::printf("%-20s %-12.3f %-14.3f\n", m.name, res.median_ms, gbps);
    }
    return 0;
}
