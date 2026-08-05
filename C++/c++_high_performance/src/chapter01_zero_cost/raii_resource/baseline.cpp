#include "baseline.hpp"

#include <stdexcept>

namespace chp {
namespace raii {

int throwing_step(int value) {
    if (value == 0) {
        throw std::runtime_error("invalid value");
    }
    return value * 2;
}

int use_manual(Resource*& out, int value) {
    // Manual resource acquisition...
    Resource* resource = new Resource();
    out = resource;  // ...and manual ownership transfer to the caller.
    // If throwing_step throws, `resource` is never deleted: a leak.
    return throwing_step(value);
}

}  // namespace raii
}  // namespace chp
