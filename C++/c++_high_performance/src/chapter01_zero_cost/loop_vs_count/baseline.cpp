#include "baseline.hpp"

namespace chp {
namespace lvc {

std::size_t count_loop(const std::vector<int>& values, int needle) {
    std::size_t count = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (values[i] == needle) {
            ++count;
        }
    }
    return count;
}

}  // namespace lvc
}  // namespace chp
