#pragma once

#include "baseline.hpp"

namespace chp {
namespace raii {

// RAII-based handling: the resource is released automatically when the guard
// goes out of scope, including on exception paths.
class ResourceGuard {
public:
    ResourceGuard() : resource_(new Resource()) {}
    ~ResourceGuard() { delete resource_; }

    ResourceGuard(const ResourceGuard&) = delete;
    ResourceGuard& operator=(const ResourceGuard&) = delete;

    Resource& get() { return *resource_; }

private:
    Resource* resource_;
};

// Same computation as use_manual(), but resource ownership is encapsulated.
int use_raii(ResourceGuard& guard, int value);

}  // namespace raii
}  // namespace chp
