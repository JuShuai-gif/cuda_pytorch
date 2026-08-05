#pragma once

#include <cstddef>
#include <memory>
#include <vector>

namespace chp {
namespace cvp {

struct Particle {
    float x = 0.0F;
    float y = 0.0F;
    float z = 0.0F;
    float mass = 1.0F;
};

// Iterates over heap-allocated particles through unique_ptr indirection.
double sum_mass_ptr(const std::vector<std::unique_ptr<Particle>>& particles);

}  // namespace cvp
}  // namespace chp
