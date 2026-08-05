#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "baseline.hpp"
#include "benchmark.hpp"

namespace {

constexpr std::size_t kIterations = 300;
constexpr std::size_t kRounds = 7;
constexpr std::size_t kWarmup = 2;

// Building a vector by many push_back calls forces several reallocations.
template <typename T>
std::uint64_t build_vec(std::size_t seed) {
    std::vector<T> v;
    for (std::size_t i = 0; i < 100'000; ++i) {
        v.emplace_back(static_cast<int>(i + seed));
    }
    return v.back().value;
}

}  // namespace

int main() {
    std::printf("== noexcept_move benchmark ==\n");
    std::printf("Building a 100k-element vector with many reallocations.\n\n");

    const auto r_ne = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            acc += build_vec<chp::nomv::MoveNoexcept>(0);
        });
    const auto r_th = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            acc += build_vec<chp::nomv::MoveThrowing>(0);
        });

    chp::print_result("MoveNoexcept (noexcept move)  -> moves during growth",
                      r_ne);
    chp::print_result("MoveThrowing (throwing move)  -> copies during growth",
                      r_th);

    const double ratio = r_th.mean_ns / r_ne.mean_ns;
    std::printf("throwing/noexcept time ratio: %.2fx\n", ratio);
    return 0;
}
