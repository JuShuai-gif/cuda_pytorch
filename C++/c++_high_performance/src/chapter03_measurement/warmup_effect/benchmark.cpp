// Warmup effect: the first iterations of a benchmark run slower than the
// steady state (cache cold, frequency scaling, lazy allocation, page faults).
// This is why the benchmark tool warms up before measuring (book PDF p.97,
// performance testing best practices: measure with realistic data, multiple
// rounds).

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "benchmark.hpp"

namespace {

constexpr std::size_t kRounds = 10;
constexpr std::size_t kIterations = 100;

// Fold the running checksum in so the compiler cannot eliminate the loop.
std::uint64_t sum_pass(std::vector<int>& v, std::uint64_t seed) {
    std::uint64_t s = seed;
    for (int x : v) {
        s += static_cast<std::uint64_t>(x) ^ (s >> 32);
    }
    return s;
}

}  // namespace

int main() {
    std::printf("== warmup_effect ==\n\n");

    std::vector<int> data(4'000'000, 3);
    std::uint64_t total_acc = 0;

    // Measure every round WITHOUT warming up: print each round's time so the
    // first (cold) rounds are visible.
    std::printf("round-by-round (no warmup):\n");
    for (std::size_t r = 0; r < kRounds; ++r) {
        std::uint64_t acc = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < kIterations; ++i) {
            acc += sum_pass(data, acc);
            chp::compiler_barrier();
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::printf("  round %2zu: %8.2f ms\n", r, ms);
        total_acc ^= acc;  // consume the checksum so the loop cannot be DCE'd
    }
    std::printf("  checksum: %llu\n",
                static_cast<unsigned long long>(total_acc));

    // Compare a benchmark WITH warmup (what we normally use).
    const auto r_warmed = chp::benchmark(kIterations, kRounds, 3,
        [&](std::uint64_t& acc) { acc += sum_pass(data, acc); });
    std::printf("\nwith warmup (mean): %.2f ms/iter (checksum %llu)\n",
                r_warmed.mean_ns / 1e6,
                static_cast<unsigned long long>(r_warmed.checksum));
    return 0;
}
