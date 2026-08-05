#include "optimized.hpp"

namespace chp {
namespace cvp {

double sum_mass_soa(const std::vector<Particle>& particles) {
    double sum = 0.0;
    for (const auto& p : particles) {
        sum += static_cast<double>(p.mass);
    }
    return sum;
}

}  // namespace cvp
}  // namespace chp
