// Correctness checks for execution policies.

#include <algorithm>
#include <cstdio>
#include <execution>
#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

namespace {

bool is_even(int v) { return (v % 2) == 0; }

}  // namespace

int main() {
    std::vector<int> v(100'000);
    std::iota(v.begin(), v.end(), 0);

    // Parallel reduce matches serial reduce for commutative addition.
    const auto seq = std::reduce(std::execution::seq, v.begin(), v.end(), 0);
    const auto par = std::reduce(std::execution::par, v.begin(), v.end(), 0);
    const auto pun = std::reduce(std::execution::par_unseq, v.begin(), v.end(), 0);
    const auto acc = std::accumulate(v.begin(), v.end(), 0);
    CHP_CHECK(seq == acc);
    CHP_CHECK(par == acc);
    CHP_CHECK(pun == acc);

    // transform_reduce with par policy.
    const std::vector<std::string> mice{"Mickey", "Minnie", "Jerry"};
    const auto n = std::transform_reduce(
        std::execution::par, mice.begin(), mice.end(), std::size_t{0},
        [](std::size_t a, std::size_t b) { return a + b; },
        [](const std::string& s) { return s.size(); });
    CHP_CHECK(n == 6 + 6 + 5);  // "Mickey"=6, "Minnie"=6, "Jerry"=5

    // par find / count_if / transform agree with serial versions.
    {
        const auto par_find =
            std::find(std::execution::par, v.begin(), v.end(), 42);
        const auto ser_find = std::find(v.begin(), v.end(), 42);
        CHP_CHECK(par_find == ser_find);

        const auto par_cnt = std::count_if(std::execution::par, v.begin(),
                                           v.end(), is_even);
        const auto ser_cnt = std::count_if(v.begin(), v.end(), is_even);
        CHP_CHECK(par_cnt == ser_cnt);

        std::vector<int> par_out(v.size()), ser_out(v.size());
        std::transform(std::execution::par, v.begin(), v.end(), par_out.begin(),
                       [](int x) { return x * x; });
        std::transform(v.begin(), v.end(), ser_out.begin(),
                       [](int x) { return x * x; });
        CHP_CHECK(par_out == ser_out);
    }

    return chp::test_summary("execution_policies");
}
