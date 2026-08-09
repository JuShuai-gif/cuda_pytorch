// Benchmark: TaskSystem parallel reduce vs single-threaded sum.

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <future>
#include <numeric>
#include <thread>
#include <vector>

#include "task_system.hpp"

namespace {

constexpr std::size_t kCount = 50'000'000;

template <typename Fn>
long measure_with_checksum(Fn fn) {
    long checksum = 0;
    const auto t0 = std::chrono::steady_clock::now();
    checksum = fn();
    const auto t1 = std::chrono::steady_clock::now();
    std::printf("  checksum=%ld\n", checksum);
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0)
        .count();
}

}  // namespace

int main() {
    std::printf("== task_system benchmark ==\n");

    std::vector<long> data(kCount);
    std::iota(data.begin(), data.end(), 0L);

    // Single thread.
    const long serial_ns = measure_with_checksum([&] {
        long sum = 0;
        for (const long v : data) {
            sum += v;
        }
        return sum;
    });

    // Thread pool, one task per worker batch.
    chp::TaskSystem pool;
    const auto workers = pool.size();
    const std::size_t chunk = (kCount + workers - 1) / workers;

    const long par_ns = measure_with_checksum([&] {
        std::vector<std::future<long>> futures;
        futures.reserve(workers);
        for (std::size_t start = 0; start < kCount; start += chunk) {
            const auto stop = std::min(start + chunk, kCount);
            futures.push_back(pool.submit([&data, start, stop] {
                long sum = 0;
                for (std::size_t i = start; i < stop; ++i) {
                    sum += data[i];
                }
                return sum;
            }));
        }
        long sum = 0;
        for (auto& f : futures) {
            sum += f.get();
        }
        return sum;
    });

    const double serial_s = static_cast<double>(serial_ns) / 1e9;
    const double par_s = static_cast<double>(par_ns) / 1e9;
    std::printf("data: %zu longs\n", kCount);
    std::printf("serial:      %7.3f s\n", serial_s);
    std::printf("pool (%zu):  %7.3f s\n", workers, par_s);
    std::printf("speedup:     %5.2fx\n", serial_s / par_s);

    return 0;
}
