#pragma once

#include <cstddef>
#include <vector>

#include "baseline.hpp"

namespace chp {
namespace cvp {

// Iterates over particles stored contiguously inside the vector.
double sum_mass_soa(const std::vector<Particle>& particles);

}  // namespace cvp
}  // namespace chp
