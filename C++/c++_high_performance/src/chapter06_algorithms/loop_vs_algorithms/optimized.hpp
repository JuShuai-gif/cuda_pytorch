#pragma once

#include <cstddef>
#include <vector>

namespace chp {
namespace lva {

std::size_t count_algo(const std::vector<int>& v, int needle);
bool find_algo(const std::vector<int>& v, int needle);
std::size_t count_if_algo(const std::vector<int>& v);
std::vector<int> transform_algo(const std::vector<int>& v);
std::vector<int> copy_if_algo(const std::vector<int>& v);
int accumulate_algo(const std::vector<int>& v);

}  // namespace lva
}  // namespace chp
