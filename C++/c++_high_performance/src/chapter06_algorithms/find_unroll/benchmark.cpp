#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "benchmark.hpp"
#include "find.hpp"

namespace {

constexpr std::size_t kCount = 10'000'000;
constexpr std::size_t kIterations = 5;
constexpr std::size_t kRounds = 7;
constexpr std::size_t kWarmup = 2;

}  // namespace

int main() {
    std::printf("== find_unroll benchmark ==\n");
    std::printf("std::find over %zu ints, value present (book PDF p.159).\n\n",
                kCount);

    std::mt19937 gen(7u);
    std::vector<int> data(kCount);
    for (std::size_t i = 0; i < kCount; ++i) {
        data[i] = static_cast<int>(gen() % 1000);
    }
    const int needle = static_cast<int>(gen() % 1000);
    const auto expected = std::find(data.begin(), data.end(), needle);

    const auto r_slow = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            acc += static_cast<std::uint64_t>(
                std::distance(data.begin(), chp::fu::find_slow(
                    data.begin(), data.end(), needle)));
        });
    const auto r_fast = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            acc += static_cast<std::uint64_t>(
                std::distance(data.begin(), chp::fu::find_fast(
                    data.begin(), data.end(), needle)));
        });
    const auto r_std = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            acc += static_cast<std::uint64_t>(std::distance(
                data.begin(), std::find(data.begin(), data.end(), needle)));
        });

    chp::print_result("find_slow (naive loop)", r_slow);
    chp::print_result("find_fast (unrolled x4)", r_fast);
    chp::print_result("std::find (libstdc++)", r_std);

    std::printf("expected distance: %td\n",
                std::distance(data.begin(), expected));
    return 0;
}
