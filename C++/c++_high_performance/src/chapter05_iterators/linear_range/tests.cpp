#include <algorithm>
#include <cstdio>
#include <cmath>
#include <iterator>
#include <set>
#include <vector>

#include "linear_range.hpp"
#include "test_utils.hpp"

using chp::lr::LinearRange;
using chp::lr::LinearRangeIterator;
using chp::lr::make_linear_range;

int main() {
    // Values match start + step*idx within float tolerance.
    {
        std::vector<float> out;
        for (auto t : make_linear_range(0.0F, 1.0F, 11)) {
            out.push_back(t);
        }
        CHP_CHECK(out.size() == 11);
        CHP_CHECK(std::fabs(out.front() - 0.0F) < 1e-6F);
        CHP_CHECK(std::fabs(out.back() - 1.0F) < 1e-6F);
        // The key property: the LAST value is exactly reachable.
        CHP_CHECK(std::fabs(out[10] - 1.0F) < 1e-6F);
        // Adjacent values are step_size apart (within tolerance).
        CHP_CHECK(std::fabs(out[1] - out[0] - 0.1F) < 1e-6F);
    }

    // 4 values: 0, 1/3, 2/3, 1.
    {
        std::vector<double> out;
        for (auto t : make_linear_range(0.0, 1.0, 4)) {
            out.push_back(t);
        }
        const std::vector<double> expected = {0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0};
        CHP_CHECK(out.size() == expected.size());
        for (std::size_t i = 0; i < expected.size(); ++i) {
            CHP_CHECK(std::fabs(out[i] - expected[i]) < 1e-9);
        }
    }

    // Reverse range.
    {
        std::vector<float> out;
        for (auto t : make_linear_range(1.0F, 0.0F, 4)) {
            out.push_back(t);
        }
        CHP_CHECK(std::fabs(out.front() - 1.0F) < 1e-6F);
        CHP_CHECK(std::fabs(out.back() - 0.0F) < 1e-6F);
    }

    // STL algorithm compatibility.
    {
        std::set<float> s;
        const float start = 0.0F;
        const float stop = 1.0F;
        const std::size_t num = 6;
        const float step = chp::lr::get_step_size(start, stop, num);
        LinearRangeIterator<float> first{start, step, 0};
        LinearRangeIterator<float> last{start, step, num};
        std::copy(first, last, std::inserter(s, s.end()));
        CHP_CHECK(s.size() == 6);
        CHP_CHECK(s.count(1.0F) == 1);
    }

    // Bidirectional: -- works.
    {
        LinearRange<float> r{0.0F, 1.0F, 4};
        auto it = r.end();
        --it;
        CHP_CHECK(std::fabs(*it - 1.0F) < 1e-6F);
    }

    return chp::test_summary("linear_range");
}
