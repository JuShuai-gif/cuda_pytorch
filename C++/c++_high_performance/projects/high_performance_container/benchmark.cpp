// Benchmark: open-addressing HashSet vs std::unordered_set lookups.

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <string>
#include <unordered_set>
#include <vector>

#include "hash_set.hpp"

namespace {

constexpr std::size_t kCount = 200'000;
constexpr std::size_t kLookups = 400'000;

template <typename Fn>
double measure(Fn fn) {
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

}  // namespace

int main() {
    std::printf("== high_performance_container benchmark ==\n");

    std::vector<std::string> keys;
    keys.reserve(kCount);
    for (std::size_t i = 0; i < kCount; ++i) {
        keys.push_back("key" + std::to_string(i));
    }

    // Random-ish lookup probes: half present, half absent.
    std::vector<std::string> probes;
    probes.reserve(kLookups);
    for (std::size_t i = 0; i < kLookups; ++i) {
        const std::size_t idx = (i * 7919) % kCount;  // prime stride
        probes.push_back((i % 2 == 0) ? keys[idx] : keys[idx] + "_nope");
    }

    chp::HashSet my_set;
    for (const auto& k : keys) {
        my_set.insert(k);
    }

    const double my_s = measure([&] {
        std::size_t hits = 0;
        for (const auto& p : probes) {
            hits += my_set.contains(p) ? 1 : 0;
        }
        std::printf("  (my_set hits=%zu)\n", hits);
    });

    std::unordered_set<std::string> std_set;
    for (const auto& k : keys) {
        std_set.insert(k);
    }

    const double std_s = measure([&] {
        std::size_t hits = 0;
        for (const auto& p : probes) {
            hits += std_set.count(p) ? 1 : 0;
        }
        std::printf("  (std::unordered_set hits=%zu)\n", hits);
    });

    std::printf("data: %zu keys, %zu lookups\n", kCount, kLookups);
    std::printf("hash_set:          %7.3f s\n", my_s);
    std::printf("std::unordered_set: %7.3f s\n", std_s);
    std::printf("ratio (std/mine):  %5.2fx\n", std_s / my_s);

    return 0;
}
