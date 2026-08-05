#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <vector>

#include "benchmark.hpp"

namespace {

constexpr std::size_t kCount = 1'000'000;
constexpr std::size_t kIterations = 10;
constexpr std::size_t kRounds = 7;
constexpr std::size_t kWarmup = 2;

}  // namespace

int main() {
    std::printf("== optional_demo benchmark ==\n");
    std::printf("Accessing 1M int values: direct vector vs vector<optional>.\n\n");

    std::vector<int> plain(kCount, 7);
    std::vector<std::optional<int>> opts(kCount, std::optional<int>{7});

    const auto r_plain = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            long sum = 0;
            for (std::size_t i = 0; i < kCount; ++i) {
                sum += plain[i];
            }
            acc += static_cast<std::uint64_t>(sum);
        });
    const auto r_opt = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            long sum = 0;
            for (std::size_t i = 0; i < kCount; ++i) {
                if (opts[i].has_value()) {
                    sum += *opts[i];
                }
            }
            acc += static_cast<std::uint64_t>(sum);
        });

    chp::print_result("vector<int> direct access", r_plain);
    chp::print_result("vector<optional<int>> with has_value check", r_opt);

    const double ratio = r_opt.mean_ns / r_plain.mean_ns;
    std::printf("optional/direct time ratio: %.2fx (extra is the bool check)\n",
                ratio);
    return 0;
}
