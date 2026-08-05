#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <vector>

#include "benchmark.hpp"

namespace {

constexpr std::size_t kIterations = 2'000'000;
constexpr std::size_t kRounds = 7;
constexpr std::size_t kWarmup = 2;

struct BigState {
    double payload[8];  // larger than std::function's SBO on libstdc++
};

}  // namespace

int main() {
    std::printf("== std_function_cost benchmark ==\n");
    std::printf("Comparing call overhead of a std::function that fits the\n");
    std::printf("Small Buffer Optimization vs one that heap allocates.\n\n");

    const auto r_small = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            std::function<int(int)> f = [](int v) { return v + 1; };
            acc += static_cast<std::uint64_t>(f(1));
        });
    const auto r_big = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            BigState state{};
            std::function<int(int)> f =
                [state](int v) { return v + static_cast<int>(state.payload[0]); };
            acc += static_cast<std::uint64_t>(f(1));
        });
    const auto r_lambda = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            auto f = [](int v) { return v + 1; };
            acc += static_cast<std::uint64_t>(f(1));
        });

    chp::print_result("std::function, SBO-friendly capture (created per call)", r_small);
    chp::print_result("std::function, large capture (heap-allocated per call)", r_big);
    chp::print_result("plain lambda (created per call)", r_lambda);

    return 0;
}
