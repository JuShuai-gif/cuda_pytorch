// Parallel std::copy_if, two strategies (PDF p.327-331).
//
// Strategy 1 (sync): an atomic write index. Correct but trashes the cache
// because every thread writes adjacent destinations -> false sharing.
// Strategy 2 (split): parallel conditional copy into a sparse range, then a
// serial std::move compaction. No shared writes, scales with the predicate.

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <vector>

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

}  // namespace

int main() {
    std::printf("== par_copy_if ==\n");

    constexpr std::size_t n = 1'000'000;
    constexpr std::size_t chunk = 100'000;

    std::vector<unsigned> src(n);
    for (std::size_t i = 0; i < n; ++i) {
        src[i] = static_cast<unsigned>(i);
    }

    for (const char* name : {"is_odd", "is_prime"}) {
        const auto pred = (name[3] == 'o') ? &is_odd : &is_prime;

        std::vector<unsigned> serial(n);
        const auto s_end = std::copy_if(src.begin(), src.end(), serial.begin(), pred);

        std::vector<unsigned> split(n);
        const auto p_end = chp11::par_copy_if_split(src.begin(), src.end(),
                                                    split.begin(), pred, chunk);

        std::vector<unsigned> sync(n);
        const auto y_end = chp11::par_copy_if_sync(src.begin(), src.end(),
                                                   sync.begin(), pred, chunk);

        const auto split_len = std::distance(split.begin(), p_end);
        const auto sync_len = std::distance(sync.begin(), y_end);
        const auto serial_len = std::distance(serial.begin(), s_end);

        // split preserves order; sync does not (slots are claimed by
        // scheduling order), so compare sync as a multiset.
        std::vector<unsigned> sorted_sync(sync.begin(), y_end);
        std::sort(sorted_sync.begin(), sorted_sync.end());
        std::vector<unsigned> sorted_serial(serial.begin(), s_end);
        std::sort(sorted_serial.begin(), sorted_serial.end());

        std::printf("%s: serial=%td split=%td sync=%td  split==serial: %d  sync content==serial: %d\n",
                    name, serial_len, split_len, sync_len,
                    std::equal(serial.begin(), s_end, split.begin()),
                    sorted_sync == sorted_serial);
    }

    return 0;
}
