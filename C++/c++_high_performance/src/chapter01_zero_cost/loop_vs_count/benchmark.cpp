#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "baseline.hpp"
#include "benchmark.hpp"
#include "optimized.hpp"

namespace {

constexpr std::size_t kCount = 4'000'000;
constexpr std::size_t kIterations = 3;
constexpr std::size_t kRounds = 7;
constexpr std::size_t kWarmup = 2;

}  // namespace

int main() {
    std::printf("== loop_vs_count benchmark ==\n");

    std::mt19937 gen(42u);
    std::vector<int> values(kCount);
    std::uniform_int_distribution<int> dist(0, 9);
    for (std::size_t i = 0; i < kCount; ++i) {
        values[i] = dist(gen);
    }
    const int needle = 5;

    const auto r_loop = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            acc += static_cast<std::uint64_t>(
                chp::lvc::count_loop(values, needle));
        });
    const auto r_algo = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            acc += static_cast<std::uint64_t>(
                chp::lvc::count_algorithm(values, needle));
        });

    std::printf("Data: %zu random ints in [0,9], needle=%d, hits=%zu\n\n",
                kCount, needle, chp::lvc::count_loop(values, needle));

    chp::print_result("hand-written for-loop", r_loop);
    chp::print_result("std::count", r_algo);

    if (r_loop.checksum == r_algo.checksum) {
        std::printf("Checksums identical.\n");
        return 0;
    }
    std::printf("ERROR: checksums differ!\n");
    return 1;
}
