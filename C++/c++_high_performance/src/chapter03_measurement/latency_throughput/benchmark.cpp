// Latency vs throughput (book PDF p.96).
//
// Latency: time between request and response of a single operation. Here we
// measure a serial dependency chain: each step depends on the previous one,
// so the CPU cannot overlap steps (limited by latency, not throughput).
//
// Throughput: number of operations per time unit. Here we process a batch of
// independent elements; independent operations can be pipelined/vectorized,
// so more work per unit time is possible.

#include <cstddef>
#include <cstdio>
#include <vector>

#include "benchmark.hpp"

namespace {

// Serial dependency chain: result of step i feeds step i+1. This is a
// latency-bound workload. The seed prevents constant-folding.
std::uint64_t latency_chain(std::size_t n, std::uint64_t seed) {
    std::uint64_t x = seed;
    for (std::size_t i = 0; i < n; ++i) {
        x = (x * 1103515245U) + 12345U;  // LCG: read-after-write dependency
    }
    return x;
}

// Independent elements summed. Each element is independent, so the loop can
// be pipelined and vectorized: a throughput-bound workload.
std::uint64_t throughput_batch(std::vector<std::uint64_t>& v,
                               std::uint64_t seed) {
    std::uint64_t sum = seed;
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] = (v[i] * 1103515245U) + 12345U;
        sum ^= v[i];
    }
    return sum;
}

}  // namespace

int main() {
    std::printf("== latency_throughput ==\n\n");

    constexpr std::size_t kIterations = 5;
    constexpr std::size_t kRounds = 7;
    constexpr std::size_t kWarmup = 2;

    // Latency: one chain step per iteration, 10M steps.
    const std::size_t steps = 10'000'000;
    const auto r_lat = chp::benchmark(1, kRounds, kWarmup,
        [&](std::uint64_t& acc) { acc += latency_chain(steps, acc); });
    // r_lat measures the whole chain; per-step latency:
    const double ns_per_step = r_lat.mean_ns / static_cast<double>(steps);

    // Throughput: one batch of 10M independent elements per iteration.
    std::vector<std::uint64_t> v(10'000'000, 7);
    const auto r_tpt = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) { acc += throughput_batch(v, acc); });
    const double ns_per_batch_elem =
        r_tpt.mean_ns / static_cast<double>(v.size());

    chp::print_result("latency: serial dependency chain (10M steps)", r_lat);
    chp::print_result("throughput: 10M independent elements", r_tpt);
    std::printf("\nper-step latency:    %8.2f ns/op\n", ns_per_step);
    std::printf("per-element batch:   %8.2f ns/op\n", ns_per_batch_elem);
    std::printf("throughput:          %8.2f ops/ns (batch)\n",
                1.0 / ns_per_batch_elem);
    std::printf("\nThe chain is latency-bound: its per-op cost cannot be\n");
    std::printf("hidden because every step depends on the previous one.\n");
    return 0;
}
