#include "optimized.hpp"

#include <algorithm>

namespace chp {
namespace lvc {

std::size_t count_algorithm(const std::vector<int>& values, int needle) {
    return static_cast<std::size_t>(
        std::count(values.begin(), values.end(), needle));
}

}  // namespace lvc
}  // namespace chp
