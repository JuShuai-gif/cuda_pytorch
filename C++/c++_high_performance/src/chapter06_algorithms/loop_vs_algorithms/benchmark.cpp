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
constexpr std::size_t kIterations = 5;
constexpr std::size_t kRounds = 7;
constexpr std::size_t kWarmup = 2;

}  // namespace

int main() {
    std::printf("== loop_vs_algorithms benchmark ==\n");
    std::printf("%zu ints (book PDF p.153-159).\n\n", kCount);

    std::mt19937 gen(11u);
    std::vector<int> data(kCount);
    for (std::size_t i = 0; i < kCount; ++i) {
        data[i] = static_cast<int>(gen() % 100);
    }
    const int needle = 42;

    // count
    const auto c_loop = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            acc += static_cast<std::uint64_t>(
                chp::lva::count_loop(data, needle));
        });
    const auto c_algo = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            acc += static_cast<std::uint64_t>(
                chp::lva::count_algo(data, needle));
        });

    // accumulate
    const auto a_loop = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            acc += static_cast<std::uint64_t>(
                chp::lva::accumulate_loop(data));
        });
    const auto a_algo = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            acc += static_cast<std::uint64_t>(
                chp::lva::accumulate_algo(data));
        });

    // transform (returns a vector)
    const auto t_loop = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            acc += static_cast<std::uint64_t>(
                chp::lva::transform_loop(data).back());
        });
    const auto t_algo = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            acc += static_cast<std::uint64_t>(
                chp::lva::transform_algo(data).back());
        });

    // copy_if
    const auto ci_loop = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            acc += static_cast<std::uint64_t>(
                chp::lva::copy_if_loop(data).size());
        });
    const auto ci_algo = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            acc += static_cast<std::uint64_t>(
                chp::lva::copy_if_algo(data).size());
        });

    chp::print_result("count     : hand loop", c_loop);
    chp::print_result("count     : std::count", c_algo);
    chp::print_result("accumulate: hand loop", a_loop);
    chp::print_result("accumulate: std::accumulate", a_algo);
    chp::print_result("transform : hand loop", t_loop);
    chp::print_result("transform : std::transform", t_algo);
    chp::print_result("copy_if   : hand loop", ci_loop);
    chp::print_result("copy_if   : std::copy_if", ci_algo);

    if (c_loop.checksum == c_algo.checksum &&
        a_loop.checksum == a_algo.checksum) {
        std::printf("checksums match.\n");
        return 0;
    }
    std::printf("ERROR: checksums differ!\n");
    return 1;
}
