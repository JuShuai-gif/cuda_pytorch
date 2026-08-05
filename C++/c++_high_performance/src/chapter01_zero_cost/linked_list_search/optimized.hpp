#pragma once

#include <cstddef>
#include <list>
#include <string>
#include <vector>

namespace chp {
namespace lls {

// Counts occurrences of `needle` in a std::vector using std::count.
std::size_t count_title_stl_vector(const std::vector<std::string>& books,
                                   const std::string& needle);

// Counts occurrences of `needle` in a std::list using std::count.
std::size_t count_title_stl_list(const std::list<std::string>& books,
                                 const std::string& needle);

}  // namespace lls
}  // namespace chp
