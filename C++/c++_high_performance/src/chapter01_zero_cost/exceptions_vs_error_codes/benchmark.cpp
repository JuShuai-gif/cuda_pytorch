#include <cstdint>
#include <cstdio>

#include "baseline.hpp"
#include "benchmark.hpp"
#include "optimized.hpp"

namespace {

constexpr std::size_t kIterations = 5'000'000;
constexpr std::size_t kRounds = 7;
constexpr std::size_t kWarmup = 2;

}  // namespace

int main() {
    std::printf("== exceptions_vs_error_codes benchmark ==\n");
    std::printf("Success path only (no error occurs): measures whether an\n");
    std::printf("unthrown exception adds any runtime cost vs error codes.\n\n");

    const auto r_codes = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            int out = 0;
            acc += static_cast<std::uint64_t>(
                chp::evc::divide_checked(1000, 7, out));
            acc += static_cast<std::uint64_t>(out);
        });
    const auto r_except = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            acc += static_cast<std::uint64_t>(chp::evc::divide_throwing(1000, 7));
        });

    chp::print_result("error-code style (success path)", r_codes);
    chp::print_result("exception style (success path)", r_except);

    const double ratio = r_codes.mean_ns / r_except.mean_ns;
    std::printf("error-code/exception time ratio: %.2fx\n", ratio);
    return 0;
}
