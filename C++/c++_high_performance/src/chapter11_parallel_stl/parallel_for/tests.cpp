// Correctness checks for parallel_for.

#include <algorithm>
#include <cstdio>
#include <execution>
#include <numeric>
#include <vector>

#include "test_utils.hpp"

#include "../../chapter05_iterators/linear_range/linear_range.hpp"

namespace {

template <typename Policy, typename Index, typename F>
void parallel_for(Policy policy, Index first, Index last, F f) {
    auto r = chp::lr::make_linear_range<Index>(first, last,
                                               static_cast<std::size_t>(last - first));
    std::for_each(policy, r.begin(), r.end(), std::move(f));
}

}  // namespace

int main() {
    constexpr std::size_t n = 100'000;

    // Fill dst[i] = i*i in parallel; matches the serial version.
    std::vector<long> par(n), ser(n);
    parallel_for(std::execution::par, std::size_t{0}, n, [&](std::size_t i) {
        par[i] = static_cast<long>(i) * static_cast<long>(i);
    });
    for (std::size_t i = 0; i < n; ++i) {
        ser[i] = static_cast<long>(i) * static_cast<long>(i);
    }
    CHP_CHECK(par == ser);

    // Mutating elements through the index works too.
    std::vector<int> v(n);
    std::iota(v.begin(), v.end(), 0);
    parallel_for(std::execution::par, std::size_t{0}, v.size(), [&](std::size_t i) {
        v[i] *= 3;
    });
    CHP_CHECK(v[0] == 0);
    CHP_CHECK(v[1] == 3);
    CHP_CHECK(v[n - 1] == static_cast<int>((n - 1) * 3));

    return chp::test_summary("parallel_for");
}
