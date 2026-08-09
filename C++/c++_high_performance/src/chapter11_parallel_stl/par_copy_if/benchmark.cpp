// Performance: copy_if strategies under light and heavy predicates.
//
// The atomic-index strategy (sync) suffers false sharing when threads write
// adjacent destinations; with a cheap predicate it can be slower than the
// serial version. The split strategy avoids shared writes entirely.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "benchmark.hpp"

#include "../par_transform/parallel.hpp"

namespace {

bool is_odd(unsigned v) { return (v % 2) == 1; }

bool is_prime(unsigned v) {
    if (v < 2) {
        return false;
    }
    if (v == 2) {
        return true;
    }
    if (v % 2 == 0) {
        return false;
    }
    for (unsigned i = 3; i * i <= v; i += 2) {
        if (v % i == 0) {
            return false;
        }
    }
    return true;
}

constexpr std::size_t kCount = 4'000'000;
constexpr std::size_t kChunk = 100'000;
constexpr std::size_t kIterations = 5;
constexpr std::size_t kRounds = 5;
constexpr std::size_t kWarmup = 1;

}  // namespace

int main() {
    std::printf("== par_copy_if benchmark ==\n");

    std::vector<unsigned> src(kCount);
    for (std::size_t i = 0; i < kCount; ++i) {
        src[i] = static_cast<unsigned>(i);
    }

    for (const char* name : {"is_odd", "is_prime"}) {
        const auto pred = (name[3] == 'o') ? &is_odd : &is_prime;

        std::vector<unsigned> dst(kCount);

        const auto serial = chp::benchmark(kIterations, kRounds, kWarmup,
            [&](std::uint64_t& acc) {
                const auto end = std::copy_if(src.begin(), src.end(),
                                              dst.begin(), pred);
                acc += static_cast<std::uint64_t>(std::distance(dst.begin(), end));
            });
        const auto split = chp::benchmark(kIterations, kRounds, kWarmup,
            [&](std::uint64_t& acc) {
                const auto end = chp11::par_copy_if_split(
                    src.begin(), src.end(), dst.begin(), pred, kChunk);
                acc += static_cast<std::uint64_t>(std::distance(dst.begin(), end));
            });
        const auto sync = chp::benchmark(kIterations, kRounds, kWarmup,
            [&](std::uint64_t& acc) {
                const auto end = chp11::par_copy_if_sync(
                    src.begin(), src.end(), dst.begin(), pred, kChunk);
                acc += static_cast<std::uint64_t>(std::distance(dst.begin(), end));
            });

        std::printf("\nPredicate: %s, %zu elements, chunk=%zu\n",
                    name, kCount, kChunk);
        chp::print_result("serial copy_if", serial);
        chp::print_result("par_copy_if_split (parallel + compact)", split);
        chp::print_result("par_copy_if_sync (atomic index)", sync);
        std::printf("split/serial: %.2fx  sync/serial: %.2fx\n",
                    serial.mean_ns / split.mean_ns, serial.mean_ns / sync.mean_ns);
    }

    return 0;
}
