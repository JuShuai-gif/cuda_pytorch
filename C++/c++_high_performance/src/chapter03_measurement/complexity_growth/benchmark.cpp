#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "baseline.hpp"
#include "benchmark.hpp"

namespace {

constexpr std::size_t kRounds = 5;
constexpr std::size_t kWarmup = 2;

// The sizes tested in the book's table (PDF p.89).
constexpr std::size_t kSizes[] = {10, 1'000, 100'000};

// Keep the total measured work roughly constant across sizes.
std::size_t iterations_for(std::size_t n) {
    std::size_t it = 20'000'000 / (n == 0 ? 1 : n);
    return it < 1 ? 1 : it;
}

}  // namespace

int main() {
    std::printf("== complexity_growth benchmark ==\n");
    std::printf("Searching for a key that is NOT present (worst case).\n");
    std::printf("Time per single search, in ns.\n\n");

    std::printf("%-10s %14s %14s %14s\n", "n", "linear/int", "linear/Point",
                "binary/int");
    for (const std::size_t n : kSizes) {
        std::vector<int> ints(n);
        std::vector<chp::cg::Point> points(n);
        for (std::size_t i = 0; i < n; ++i) {
            ints[i] = static_cast<int>(i);
            points[i].x = static_cast<int>(i);
            points[i].y = static_cast<int>(i);
        }
        const std::size_t iters = iterations_for(n);

        const auto r_int = chp::benchmark(iters, kRounds, kWarmup,
            [&](std::uint64_t& acc) {
                acc += static_cast<std::uint64_t>(
                    chp::cg::linear_search_int(ints, -1));
            });
        const auto r_pt = chp::benchmark(iters, kRounds, kWarmup,
            [&](std::uint64_t& acc) {
                acc += static_cast<std::uint64_t>(
                    chp::cg::linear_search_point(points, chp::cg::Point{-1, -1}));
            });
        const auto r_bin = chp::benchmark(iters, kRounds, kWarmup,
            [&](std::uint64_t& acc) {
                acc += static_cast<std::uint64_t>(
                    chp::cg::binary_search_int(ints, -1));
            });

        std::printf("%-10zu %14.2f %14.2f %14.2f\n", n, r_int.mean_ns,
                    r_pt.mean_ns, r_bin.mean_ns);
    }
    return 0;
}
