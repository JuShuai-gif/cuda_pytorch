// Performance: naive chunking vs divide-and-conquer across chunk sizes.
//
// Uses a cost function that grows with the input value (PDF p.307): a fixed
// chunk count leaves one slow chunk, while many small chunks balance load.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>

#include "benchmark.hpp"

#include "parallel.hpp"

namespace {

// Cost grows with v, so the naive equal-size chunks are not equal cost.
float cost_varying(float v) {
    float sum = v;
    const auto i_max = static_cast<std::size_t>(v / 100.0F);
    for (std::size_t i = 0; i < i_max; ++i) {
        sum += static_cast<float>(i * i * i) * sum;
    }
    return sum;
}

constexpr std::size_t kCount = 400'000;
constexpr std::size_t kIterations = 10;
constexpr std::size_t kRounds = 5;
constexpr std::size_t kWarmup = 1;

}  // namespace

int main() {
    std::printf("== par_transform benchmark ==\n");

    std::vector<float> src(kCount);
    std::iota(src.begin(), src.end(), 0.0F);

    const auto run = [&](auto transform_fn) {
        std::vector<float> dst(kCount);
        const auto res = chp::benchmark(kIterations, kRounds, kWarmup,
            [&](std::uint64_t& acc) {
                transform_fn(src, dst);
                acc += static_cast<std::uint64_t>(dst[kCount / 2] * 1e6);
            });
        return res;
    };

    const auto r_serial = run([](const std::vector<float>& s,
                                 std::vector<float>& d) {
        std::transform(s.begin(), s.end(), d.begin(), cost_varying);
    });

    const auto r_naive = run([](const std::vector<float>& s,
                                std::vector<float>& d) {
        chp11::par_transform_naive(s.begin(), s.end(), d.begin(), cost_varying);
    });

    const auto r_dac = run([](const std::vector<float>& s,
                              std::vector<float>& d) {
        chp11::par_transform(s.begin(), s.end(), d.begin(), cost_varying, 10'000);
    });

    std::printf("Data: %zu floats, cost grows with value; serial baseline\n\n",
                kCount);

    chp::print_result("serial transform", r_serial);
    chp::print_result("par_transform_naive (hw conc chunks)", r_naive);
    chp::print_result("par_transform divide&conquer chunk=10000", r_dac);

    std::printf("naive/serial ratio: %.2fx\n",
                r_serial.mean_ns / r_naive.mean_ns);
    std::printf("dac/serial ratio:   %.2fx\n",
                r_serial.mean_ns / r_dac.mean_ns);

    return 0;
}
