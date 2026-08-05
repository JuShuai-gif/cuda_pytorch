#include <algorithm>
#include <cstdio>
#include <vector>

#include "baseline.hpp"
#include "optimized.hpp"
#include "test_utils.hpp"

int main() {
    const std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    CHP_CHECK(chp::lva::count_loop(v, 3) == chp::lva::count_algo(v, 3));
    CHP_CHECK(chp::lva::count_loop(v, 0) == chp::lva::count_algo(v, 0));
    CHP_CHECK(chp::lva::find_loop(v, 7) == chp::lva::find_algo(v, 7));
    CHP_CHECK(chp::lva::find_loop(v, 99) == chp::lva::find_algo(v, 99));
    CHP_CHECK(chp::lva::count_if_loop(v) == chp::lva::count_if_algo(v));
    CHP_CHECK(chp::lva::accumulate_loop(v) == chp::lva::accumulate_algo(v));
    CHP_CHECK(chp::lva::transform_loop(v) == chp::lva::transform_algo(v));
    CHP_CHECK(chp::lva::copy_if_loop(v) == chp::lva::copy_if_algo(v));

    // Empty vector.
    const std::vector<int> empty;
    CHP_CHECK(chp::lva::count_loop(empty, 1) == 0);
    CHP_CHECK(chp::lva::find_algo(empty, 1) == false);
    CHP_CHECK(chp::lva::accumulate_algo(empty) == 0);

    return chp::test_summary("loop_vs_algorithms");
}
