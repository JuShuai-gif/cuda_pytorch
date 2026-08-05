#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "benchmark.hpp"

namespace {

constexpr std::size_t kCount = 100'000;
constexpr std::size_t kIterations = 30;
constexpr std::size_t kRounds = 7;
constexpr std::size_t kWarmup = 2;

// Build a vector of kCount ints without reserving: triggers reallocations.
std::uint64_t build_no_reserve() {
    std::vector<int> v;
    for (std::size_t i = 0; i < kCount; ++i) {
        v.push_back(static_cast<int>(i));
    }
    return v.back();
}

// Build a vector of kCount ints after reserving: no reallocations.
std::uint64_t build_with_reserve() {
    std::vector<int> v;
    v.reserve(kCount);
    for (std::size_t i = 0; i < kCount; ++i) {
        v.push_back(static_cast<int>(i));
    }
    return v.back();
}

}  // namespace

int main() {
    std::printf("== vector_growth benchmark ==\n");
    std::printf("Building a %zu-element vector of ints.\n\n", kCount);

    const auto r_nr = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) { acc += build_no_reserve(); });
    const auto r_rs = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) { acc += build_with_reserve(); });

    chp::print_result("push_back without reserve (reallocations happen)", r_nr);
    chp::print_result("push_back with reserve (no reallocations)", r_rs);

    const double ratio = r_nr.mean_ns / r_rs.mean_ns;
    std::printf("no-reserve/reserve time ratio: %.2fx\n", ratio);
    return 0;
}
