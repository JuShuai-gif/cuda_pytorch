// Correctness checks for the two parallel copy_if strategies.

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <iterator>
#include <vector>

#include "test_utils.hpp"

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
    constexpr std::size_t n = 200'000;
    constexpr std::size_t chunk = 20'000;

    std::vector<unsigned> src(n);
    for (std::size_t i = 0; i < n; ++i) {
        src[i] = static_cast<unsigned>(i);
    }

    for (const auto pred : {static_cast<bool (*)(unsigned)>(&is_odd),
                            static_cast<bool (*)(unsigned)>(&is_prime)}) {
        std::vector<unsigned> serial(n), split(n), sync(n);
        const auto s_end = std::copy_if(src.begin(), src.end(), serial.begin(), pred);
        const auto p_end = chp11::par_copy_if_split(src.begin(), src.end(),
                                                    split.begin(), pred, chunk);
        const auto y_end = chp11::par_copy_if_sync(src.begin(), src.end(),
                                                   sync.begin(), pred, chunk);

        const auto serial_len = std::distance(serial.begin(), s_end);
        const auto split_len = std::distance(split.begin(), p_end);
        const auto sync_len = std::distance(sync.begin(), y_end);

        // split preserves order and content.
        CHP_CHECK(serial_len == split_len);
        CHP_CHECK(std::equal(serial.begin(), s_end, split.begin()));

        // sync assigns destination slots by scheduling order, so order may
        // differ; content must be identical. Compare sorted copies.
        CHP_CHECK(serial_len == sync_len);
        std::vector<unsigned> sorted_serial(serial.begin(), s_end);
        std::sort(sorted_serial.begin(), sorted_serial.end());
        std::vector<unsigned> sorted_sync(sync.begin(), y_end);
        std::sort(sorted_sync.begin(), sorted_sync.end());
        CHP_CHECK(sorted_serial == sorted_sync);
    }

    return chp::test_summary("par_copy_if");
}
