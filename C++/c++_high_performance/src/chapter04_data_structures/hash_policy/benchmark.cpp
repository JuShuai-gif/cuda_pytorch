// Hash policy of std::unordered_* containers.
//
// The book (PDF p.114-117): hash tables use separate chaining. A good hash
// distributes keys evenly; the load_factor (elements/buckets) controls
// collisions; rehash happens when max_load_factor is reached. We measure:
//  - bucket count growth during insertions;
//  - the effect of load_factor on lookup performance;
//  - the effect of reserve() on rehash frequency.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <unordered_map>

#include "benchmark.hpp"

namespace {

constexpr std::size_t kCount = 1'000'000;
constexpr std::size_t kIterations = 10;
constexpr std::size_t kRounds = 7;
constexpr std::size_t kWarmup = 2;

// A deliberately terrible hash: all keys land in one bucket.
struct BadHash {
    std::size_t operator()(int) const { return 0; }
};

}  // namespace

int main() {
    std::printf("== hash_policy ==\n\n");

    // --- bucket growth and rehash ---
    {
        std::unordered_map<int, int> m;
        std::size_t prev_buckets = m.bucket_count();
        std::size_t rehashes = 0;
        for (std::size_t i = 0; i < 100'000; ++i) {
            m.emplace(static_cast<int>(i), 1);
            if (m.bucket_count() != prev_buckets) {
                ++rehashes;
                prev_buckets = m.bucket_count();
            }
        }
        std::printf("100k inserts: buckets=%zu max_load_factor=%.1f "
                    "rehashes=%zu\n",
                    m.bucket_count(), m.max_load_factor(), rehashes);
    }

    // --- good hash: default std::hash<int> ---
    {
        std::unordered_map<int, int> m;
        m.reserve(kCount);
        for (std::size_t i = 0; i < kCount; ++i) {
            m.emplace(static_cast<int>(i), 1);
        }
        std::size_t max_bucket = 0;
        for (std::size_t b = 0; b < m.bucket_count(); ++b) {
            const std::size_t n = m.bucket_size(b);
            if (n > max_bucket) {
                max_bucket = n;
            }
        }
        std::printf("good hash: %zu keys, %zu buckets, max bucket size %zu\n",
                    m.size(), m.bucket_count(), max_bucket);
    }

    // --- bad hash: everything in one bucket, O(n) lookup ---
    {
        // Fewer elements: an O(n) scan per lookup is expensive.
        constexpr std::size_t kBadCount = 100'000;
        std::unordered_map<int, int, BadHash> m;
        m.reserve(kBadCount);
        for (std::size_t i = 0; i < kBadCount; ++i) {
            m.emplace(static_cast<int>(i), 1);
        }
        std::printf("bad hash: %zu keys, %zu buckets, bucket[0] size %zu\n",
                    m.size(), m.bucket_count(), m.bucket_size(0));

        const auto r_bad = chp::benchmark(kIterations, kRounds, kWarmup,
            [&](std::uint64_t& acc) {
                std::uint64_t sum = 0;
                for (std::size_t i = 0; i < kBadCount; i += 100) {
                    sum += static_cast<std::uint64_t>(
                        m.at(static_cast<int>(i)));
                }
                acc += sum;
            });
        chp::print_result("bad hash (one bucket, O(n) scan)", r_bad);

        std::unordered_map<int, int> good;
        good.reserve(kBadCount);
        for (std::size_t i = 0; i < kBadCount; ++i) {
            good.emplace(static_cast<int>(i), 1);
        }
        const auto r_good = chp::benchmark(kIterations, kRounds, kWarmup,
            [&](std::uint64_t& acc) {
                std::uint64_t sum = 0;
                for (std::size_t i = 0; i < kBadCount; i += 100) {
                    sum += static_cast<std::uint64_t>(
                        good.at(static_cast<int>(i)));
                }
                acc += sum;
            });
        chp::print_result("good hash (even distribution)", r_good);
        std::printf("bad/good time ratio: %.2fx\n",
                    r_bad.mean_ns / r_good.mean_ns);
    }

    return 0;
}
