// Associative containers: ordered (tree) vs unordered (hash table).
//
// The book (PDF p.112-115): tree-based containers (set/map) offer O(log n)
// insert/search/delete; hash-based containers (unordered_set/map) offer
// O(1) on average. The difference only matters for large containers.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "benchmark.hpp"

namespace {

constexpr std::size_t kCount = 1'000'000;
constexpr std::size_t kIterations = 10;
constexpr std::size_t kRounds = 7;
constexpr std::size_t kWarmup = 2;

}  // namespace

int main() {
    std::printf("== associative_containers ==\n");
    std::printf("Lookup of %zu integer keys.\n\n", kCount);

    std::map<int, int> om;
    std::unordered_map<int, int> um;
    for (std::size_t i = 0; i < kCount; ++i) {
        const int key = static_cast<int>(i);
        om.emplace(key, key);
        um.emplace(key, key);
    }

    const auto r_map = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            std::uint64_t sum = 0;
            for (std::size_t i = 0; i < kCount; i += 7) {
                const int key = static_cast<int>(i);
                sum += static_cast<std::uint64_t>(om.at(key));
            }
            acc += sum;
        });
    const auto r_umap = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            std::uint64_t sum = 0;
            for (std::size_t i = 0; i < kCount; i += 7) {
                const int key = static_cast<int>(i);
                sum += static_cast<std::uint64_t>(um.at(key));
            }
            acc += sum;
        });

    chp::print_result("std::map (balanced tree, O(log n))", r_map);
    chp::print_result("std::unordered_map (hash table, O(1))", r_umap);

    const double ratio = r_map.mean_ns / r_umap.mean_ns;
    std::printf("map/unordered_map time ratio: %.2fx\n", ratio);
    return 0;
}
