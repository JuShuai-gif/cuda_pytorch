#include <any>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "benchmark.hpp"

namespace {

constexpr std::size_t kIterations = 2'000'000;
constexpr std::size_t kRounds = 7;
constexpr std::size_t kWarmup = 2;

}  // namespace

int main() {
    std::printf("== any_demo benchmark ==\n");
    std::printf("Reading a stored value back: direct int vs std::any + any_cast.\n\n");

    const auto r_direct = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            int v = 42;
            acc += static_cast<std::uint64_t>(v);
        });
    const auto r_any = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            std::any a = 42;
            acc += static_cast<std::uint64_t>(std::any_cast<int>(a));
        });

    chp::print_result("direct int value", r_direct);
    chp::print_result("std::any + std::any_cast", r_any);

    const double ratio = r_any.mean_ns / r_direct.mean_ns;
    std::printf("any/direct time ratio: %.2fx\n", ratio);
    return 0;
}
