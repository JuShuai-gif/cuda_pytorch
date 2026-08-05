#include "optimized.hpp"

#include <algorithm>

namespace chp {
namespace lls {

std::size_t count_title_stl_vector(const std::vector<std::string>& books,
                                   const std::string& needle) {
    return static_cast<std::size_t>(
        std::count(books.begin(), books.end(), needle));
}

std::size_t count_title_stl_list(const std::list<std::string>& books,
                                 const std::string& needle) {
    return static_cast<std::size_t>(
        std::count(books.begin(), books.end(), needle));
}

}  // namespace lls
}  // namespace chp
