// A deliberately unbalanced program: one function dominates the runtime,
// demonstrating how to find hot spots with a profiler (book PDF p.97-102).
//
// 80% of the work is done by compute_heavy(); the rest is split across two
// cheap helpers. Run with perf/gprof to see that:
//
//   ./scripts/perf_record.sh ./build/chapter03_measurement/ch03_hotspot_profiling
//   perf report
//
//   gprof ./build/chapter03_measurement/ch03_hotspot_profiling gmon.out

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

// The hot spot: sorting 100k-element vectors, repeated.
std::uint64_t compute_heavy(std::size_t iterations) {
    std::vector<int> data(100'000);
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<int>((i * 2654435761U) % 1000003);
    }
    std::uint64_t checksum = 0;
    for (std::size_t r = 0; r < iterations; ++r) {
        std::sort(data.begin(), data.end());
        checksum += static_cast<std::uint64_t>(data[r % data.size()]);
        for (std::size_t i = 0; i < data.size(); ++i) {
            data[i] = static_cast<int>((data[i] * 31U) % 1000003);
        }
    }
    return checksum;
}

// A cheap helper called many times.
std::uint64_t helper_a(std::uint64_t x) { return (x * 3U) + 1U; }

// Another cheap helper called many times.
std::uint64_t helper_b(std::uint64_t x) { return (x >> 1) ^ x; }

}  // namespace

int main() {
    const std::size_t heavy_iters = 8;
    std::uint64_t checksum = 0;
    for (std::size_t i = 0; i < 100; ++i) {
        checksum = helper_a(checksum);
        checksum = helper_b(checksum);
    }
    checksum += compute_heavy(heavy_iters);
    std::printf("hotspot_profiling checksum: %llu\n",
                static_cast<unsigned long long>(checksum));
    return 0;
}
