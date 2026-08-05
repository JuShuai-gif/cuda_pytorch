#pragma once

#include <cstddef>

namespace chp {
namespace raii {

// A tracked resource: every construction/destruction is counted so tests can
// verify that the release path actually ran.
struct Resource {
    static inline int constructed = 0;
    static inline int destroyed = 0;

    Resource() { ++constructed; }
    Resource(const Resource&) { ++constructed; }
    Resource& operator=(const Resource&) = default;
    ~Resource() { ++destroyed; }
};

// Manual resource handling: the caller must remember to delete the resource
// on every exit path. If `throwing_step` throws, `out` leaks.
int use_manual(Resource*& out, int value);

// A step that throws when value == 0, after acquiring the resource.
int throwing_step(int value);

}  // namespace raii
}  // namespace chp
