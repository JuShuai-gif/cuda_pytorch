#include "optimized.hpp"

#include <algorithm>
#include <numeric>

namespace chp {
namespace lva {

std::size_t count_algo(const std::vector<int>& v, int needle) {
    return static_cast<std::size_t>(std::count(v.begin(), v.end(), needle));
}

bool find_algo(const std::vector<int>& v, int needle) {
    return std::find(v.begin(), v.end(), needle) != v.end();
}

std::size_t count_if_algo(const std::vector<int>& v) {
    return static_cast<std::size_t>(std::count_if(
        v.begin(), v.end(), [](int x) { return x % 7 == 0; }));
}

std::vector<int> transform_algo(const std::vector<int>& v) {
    std::vector<int> out(v.size());
    std::transform(v.begin(), v.end(), out.begin(),
                   [](int x) { return x * 3 + 1; });
    return out;
}

std::vector<int> copy_if_algo(const std::vector<int>& v) {
    std::vector<int> out;
    out.reserve(v.size());
    std::copy_if(v.begin(), v.end(), std::back_inserter(out),
                 [](int x) { return x > 50; });
    return out;
}

int accumulate_algo(const std::vector<int>& v) {
    return std::accumulate(v.begin(), v.end(), 0);
}

}  // namespace lva
}  // namespace chp
