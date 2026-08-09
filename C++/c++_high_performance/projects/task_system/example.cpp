// Task system demo: parallel sum of squares over a vector.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

#include "task_system.hpp"

namespace {

constexpr std::size_t kCount = 10'000'000;

}  // namespace

int main() {
    std::printf("== task_system ==\n");

    chp::TaskSystem pool;
    std::printf("worker threads: %zu\n", pool.size());

    std::vector<long> data(kCount);
    for (std::size_t i = 0; i < kCount; ++i) {
        data[i] = static_cast<long>(i);
    }

    // Split the range across tasks; each task sums a chunk of squares.
    const std::size_t num_tasks = pool.size() * 4;
    const std::size_t chunk = (kCount + num_tasks - 1) / num_tasks;

    std::vector<std::future<long>> futures;
    futures.reserve(num_tasks);
    for (std::size_t start = 0; start < kCount; start += chunk) {
        const auto stop = std::min(start + chunk, kCount);
        futures.push_back(pool.submit([&data, start, stop] {
            long sum = 0;
            for (std::size_t i = start; i < stop; ++i) {
                sum += data[i] * data[i];
            }
            return sum;
        }));
    }

    long total = 0;
    for (auto& f : futures) {
        total += f.get();
    }

    // Reference: n(n-1)(2n-1)/6 for sum of squares 0..n-1.
    const long expect =
        static_cast<long>(kCount) * static_cast<long>(kCount - 1) *
        static_cast<long>(2 * kCount - 1) / 6;
    std::printf("parallel sum of squares = %ld (expected %ld) %s\n", total,
                expect, total == expect ? "OK" : "MISMATCH");

    return 0;
}
