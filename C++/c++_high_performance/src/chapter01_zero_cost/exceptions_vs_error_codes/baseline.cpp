#include "baseline.hpp"

namespace chp {
namespace evc {

DivideError divide_checked(int a, int b, int& out) {
    if (b == 0) {
        return DivideError::division_by_zero;
    }
    out = a / b;
    return DivideError::none;
}

}  // namespace evc
}  // namespace chp
