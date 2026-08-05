#include "benchmark.hpp"

#include <cinttypes>
#include <cstdio>

#include "system_info.hpp"

namespace chp {

void print_result(const char* name, const BenchmarkResult& res) {
    std::printf("Benchmark: %s\n", name);
    std::printf("  mean:   %10.1f ns/iter\n", res.mean_ns);
    std::printf("  median: %10.1f ns/iter\n", res.median_ns);
    std::printf("  min:    %10.1f ns/iter\n", res.min_ns);
    std::printf("  max:    %10.1f ns/iter\n", res.max_ns);
    std::printf("  stddev: %10.1f ns/iter\n", res.stddev_ns);
    std::printf("  iterations: %" PRIuMAX "\n",
                static_cast<std::uintmax_t>(res.iterations));
    std::printf("  checksum:   %" PRIu64 "\n", res.checksum);
    std::printf("  system: ");
    print_system_info();
    std::printf("\n");
}

}  // namespace chp
