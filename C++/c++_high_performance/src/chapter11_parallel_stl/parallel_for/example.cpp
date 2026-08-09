// Parallelizing an index-based for-loop (PDF p.338-339).
//
// Index loops have no direct STL equivalent; combine a LinearRange of
// indices with std::for_each + an execution policy, wrapped as parallel_for.

#include <algorithm>
#include <cstdio>
#include <execution>
#include <string>
#include <vector>

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
    std::printf("== parallel_for ==\n");

    std::vector<std::string> mice{"Mickey", "Minnie", "Jerry", "Donald"};
    parallel_for(std::execution::par, std::size_t{0}, mice.size(), [&](std::size_t i) {
        if (i == 0) {
            mice[i] += " is first.";
        } else if (i + 1 == mice.size()) {
            mice[i] += " is last.";
        } else {
            mice[i] += ".";
        }
    });

    for (const auto& m : mice) {
        std::printf("%s ", m.c_str());
    }
    std::printf("\n");

    // Parallel index loop over a numeric computation: fill dst[i] = i*i*i.
    constexpr std::size_t n = 10'000'000;
    std::vector<long> dst(n);
    parallel_for(std::execution::par_unseq, std::size_t{0}, n, [&](std::size_t i) {
        dst[i] = static_cast<long>(i) * static_cast<long>(i) * static_cast<long>(i);
    });
    std::printf("dst[1]=%ld dst[9]=%ld dst[9999999]=%ld\n",
                dst[1], dst[9], dst[n - 1]);

    return 0;
}
