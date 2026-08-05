#include "optimized.hpp"

#include <stdexcept>

namespace chp {
namespace raii {

int use_raii(ResourceGuard& guard, int value) {
    (void)guard;
    // If throwing_step throws, the guard's destructor releases the resource.
    return throwing_step(value);
}

}  // namespace raii
}  // namespace chp
