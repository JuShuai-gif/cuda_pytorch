// Sorting only what you need: sort vs partial_sort vs nth_element.
//
// The book (PDF p.160-162) shows that for "find the median" or "find the top
// m elements" you do not need a full sort:
//   std::sort          O(n log n)
//   std::partial_sort  O(n log m)
//   std::nth_element   O(n)

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "benchmark.hpp"

namespace {

constexpr std::size_t kCount = 2'000'000;
constexpr std::size_t kM = 100'000;
constexpr std::size_t kIterations = 5;
constexpr std::size_t kRounds = 7;
constexpr std::size_t kWarmup = 2;

void reset(std::vector<int>& v) {
    static std::mt19937 gen(2024u);
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] = static_cast<int>(gen() % 1'000'000);
    }
}

}  // namespace

int main() {
    std::printf("== partial_sorting benchmark ==\n");
    std::printf("%zu elements, need the top %zu (book PDF p.160-162).\n\n",
                kCount, kM);

    std::vector<int> data(kCount);
    reset(data);

    const auto r_sort = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            reset(data);
            std::sort(data.begin(), data.end());
            acc += static_cast<std::uint64_t>(data[kM - 1]);
        });

    reset(data);
    const auto r_partial = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            reset(data);
            std::partial_sort(data.begin(), data.begin() + kM, data.end());
            acc += static_cast<std::uint64_t>(data[kM - 1]);
        });

    reset(data);
    const auto r_nth = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            reset(data);
            std::nth_element(data.begin(), data.begin() + kM, data.end());
            acc += static_cast<std::uint64_t>(data[kM]);
        });

    chp::print_result("std::sort        (O(n log n))", r_sort);
    chp::print_result("std::partial_sort (O(n log m))", r_partial);
    chp::print_result("std::nth_element  (O(n))", r_nth);

    std::printf("sort/partial time ratio: %.2fx\n",
                r_sort.mean_ns / r_partial.mean_ns);
    std::printf("sort/nth     time ratio: %.2fx\n",
                r_sort.mean_ns / r_nth.mean_ns);
    return 0;
}
