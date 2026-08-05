#include <cstdio>
#include <random>
#include <vector>

#include "baseline.hpp"
#include "optimized.hpp"
#include "test_utils.hpp"

int main() {
    const std::vector<int> a = {1, 2, 3, 2, 5, 2, 7};
    CHP_CHECK(chp::lvc::count_loop(a, 2) == 3);
    CHP_CHECK(chp::lvc::count_algorithm(a, 2) == 3);
    CHP_CHECK(chp::lvc::count_loop(a, 9) == 0);
    CHP_CHECK(chp::lvc::count_algorithm(a, 9) == 0);
    CHP_CHECK(chp::lvc::count_loop(a, 1) == 1);
    CHP_CHECK(chp::lvc::count_algorithm(a, 7) == 1);

    const std::vector<int> empty;
    CHP_CHECK(chp::lvc::count_loop(empty, 0) == 0);
    CHP_CHECK(chp::lvc::count_algorithm(empty, 0) == 0);

    // Both implementations must agree on a large random input.
    std::mt19937 gen(7u);
    std::vector<int> big(100000);
    std::uniform_int_distribution<int> dist(0, 99);
    for (std::size_t i = 0; i < big.size(); ++i) {
        big[i] = dist(gen);
    }
    for (int needle = 0; needle < 100; ++needle) {
        CHP_CHECK(chp::lvc::count_loop(big, needle) ==
                  chp::lvc::count_algorithm(big, needle));
    }

    return chp::test_summary("loop_vs_count");
}
