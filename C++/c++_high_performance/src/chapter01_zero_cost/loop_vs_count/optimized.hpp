#pragma once

#include <cstddef>
#include <vector>

namespace chp {
namespace lvc {

// Counts occurrences of `needle` using std::count from <algorithm>.
std::size_t count_algorithm(const std::vector<int>& values, int needle);

}  // namespace lvc
}  // namespace chp
