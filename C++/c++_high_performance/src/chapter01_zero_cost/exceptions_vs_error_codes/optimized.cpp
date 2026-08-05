#include "optimized.hpp"

#include <stdexcept>

namespace chp {
namespace evc {

int divide_throwing(int a, int b) {
    if (b == 0) {
        throw std::runtime_error("division by zero");
    }
    return a / b;
}

}  // namespace evc
}  // namespace chp
