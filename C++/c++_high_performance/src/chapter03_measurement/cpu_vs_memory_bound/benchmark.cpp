// CPU-bound vs memory-bound (book PDF p.96).
//
// A task is CPU-bound if it would run faster with a faster CPU; it is
// memory-bound if the main memory speed/bandwidth is the bottleneck.
//
// Here we build two loops over the same data:
//   - cpu_loop: repeated arithmetic on values already in registers/cache
//   - memory_loop: a single pass over a large array (cache misses dominate)

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "benchmark.hpp"

namespace {

// Pure arithmetic, touches a small buffer that fits in L1 cache.
// Takes the running checksum as a seed so the compiler cannot constant-fold
// the loop (a pure function of a compile-time constant would be).
std::uint64_t cpu_loop(std::size_t n, std::uint64_t seed) {
    std::uint64_t x = seed;
    for (std::size_t i = 0; i < n; ++i) {
        x = (x * 6364136223846793005ULL) + 1442695040888963407ULL;
    }
    return x;
}

// One pass over a large array: the reads must come from DRAM. The running
// checksum is folded into each element so consecutive passes differ and the
// loop cannot be eliminated.
std::uint64_t memory_loop(std::vector<std::uint64_t>& v, std::uint64_t seed) {
    std::uint64_t sum = seed;
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] = (v[i] * 3U) + 1U;
        sum ^= v[i];
    }
    return sum;
}

}  // namespace

int main() {
    std::printf("== cpu_vs_memory_bound ==\n\n");

    constexpr std::size_t kIterations = 5;
    constexpr std::size_t kRounds = 7;
    constexpr std::size_t kWarmup = 2;

    const std::size_t work = 100'000'000;
    const auto r_cpu = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) { acc += cpu_loop(work, acc); });
    std::printf("CPU loop: %zu arithmetic iterations -> %.2f ms\n", work,
                r_cpu.mean_ns / 1e6);
    std::printf("  checksum: %llu\n",
                static_cast<unsigned long long>(r_cpu.checksum));

    // A large buffer far exceeds L2/L3 so the pass is memory-bound.
    const std::size_t mb = 256;
    std::vector<std::uint64_t> v(mb * 1024 * 1024 / sizeof(std::uint64_t), 1);
    const auto r_mem = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) { acc += memory_loop(v, acc); });
    std::printf("Memory loop: %zu MiB read+write once -> %.2f ms\n", mb,
                r_mem.mean_ns / 1e6);
    std::printf("  checksum: %llu\n",
                static_cast<unsigned long long>(r_mem.checksum));

    // Effective per-byte throughput.
    const double bytes = static_cast<double>(v.size()) * sizeof(std::uint64_t);
    const double gb_per_s = bytes / (r_mem.mean_ns / 1e9) / 1e9;
    std::printf("effective read bandwidth: %.1f GiB/s\n", gb_per_s);
    std::printf("\nUse: perf stat <binary> to see cache-misses\n");
    std::printf("  ./scripts/perf_stat.sh ./build/chapter03_measurement/"
                "ch03_cpu_mem_benchmark\n");
    return 0;
}
