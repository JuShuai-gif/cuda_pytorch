#pragma once

#include <cstddef>
#include <vector>

namespace chp {
namespace lvc {

// Counts occurrences of `needle` with a hand-written for-loop.
std::size_t count_loop(const std::vector<int>& values, int needle);

}  // namespace lvc
}  // namespace chp
