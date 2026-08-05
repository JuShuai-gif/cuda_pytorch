#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <vector>

#include "benchmark.hpp"

namespace {

constexpr std::size_t kCount = 1'000'000;
constexpr std::size_t kIterations = 3;
constexpr std::size_t kRounds = 5;
constexpr std::size_t kWarmup = 2;

// A lambda object stored directly in a vector: fully inlined at -O2/-O3.
// (C++17 forbids lambdas in unevaluated contexts, so we name one first.)
auto lbd = [](int v) { return v * 3; };
using LambdaType = decltype(lbd);

// An equivalent hand-written function object.
struct TimesThree {
    int operator()(int v) const { return v * 3; }
};

}  // namespace

int main() {
    std::printf("== callable_overhead benchmark ==\n");
    std::printf("One million callable objects in a vector, invoked in a loop.\n\n");

    std::vector<LambdaType> lambdas(kCount, lbd);
    std::vector<TimesThree> functors(kCount, TimesThree{});
    std::vector<std::function<int(int)>> funcs(kCount, TimesThree{});

    const auto r_lambda = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            int res = 0;
            for (std::size_t i = 0; i < kCount; ++i) {
                res = lambdas[i](res);
                res ^= static_cast<int>(i);  // prevent loop constant-folding
            }
            acc += static_cast<std::uint64_t>(res);
        });
    const auto r_functor = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            int res = 0;
            for (std::size_t i = 0; i < kCount; ++i) {
                res = functors[i](res);
                res ^= static_cast<int>(i);  // prevent loop constant-folding
            }
            acc += static_cast<std::uint64_t>(res);
        });
    const auto r_function = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            int res = 0;
            for (std::size_t i = 0; i < kCount; ++i) {
                res = funcs[i](res);
                res ^= static_cast<int>(i);  // prevent loop constant-folding
            }
            acc += static_cast<std::uint64_t>(res);
        });

    chp::print_result("vector of lambdas (direct)", r_lambda);
    chp::print_result("vector of functor objects", r_functor);
    chp::print_result("vector of std::function", r_function);

    if (r_lambda.checksum == r_functor.checksum &&
        r_functor.checksum == r_function.checksum) {
        std::printf("Checksums identical.\n");
        return 0;
    }
    std::printf("ERROR: checksums differ!\n");
    return 1;
}
