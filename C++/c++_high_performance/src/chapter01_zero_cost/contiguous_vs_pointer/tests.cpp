#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include "baseline.hpp"
#include "optimized.hpp"
#include "test_utils.hpp"

int main() {
    const std::vector<double> masses = {1.0, 2.5, 3.25, 0.5, 10.0};
    double expected = 0.0;

    std::vector<chp::cvp::Particle> soa;
    std::vector<std::unique_ptr<chp::cvp::Particle>> ptr;
    for (double m : masses) {
        expected += m;
        chp::cvp::Particle p;
        p.mass = static_cast<float>(m);
        soa.push_back(p);
        ptr.push_back(std::make_unique<chp::cvp::Particle>(p));
    }

    CHP_CHECK(chp::cvp::sum_mass_soa(soa) == expected);
    CHP_CHECK(chp::cvp::sum_mass_ptr(ptr) == expected);

    // Both implementations must agree on a larger random input.
    std::mt19937 gen(9u);
    std::uniform_real_distribution<float> dist(0.0F, 1.0F);
    const std::size_t n = 100000;
    std::vector<chp::cvp::Particle> big_soa;
    std::vector<std::unique_ptr<chp::cvp::Particle>> big_ptr;
    big_soa.reserve(n);
    big_ptr.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        chp::cvp::Particle p;
        p.mass = dist(gen);
        big_soa.push_back(p);
        big_ptr.push_back(std::make_unique<chp::cvp::Particle>(p));
    }
    CHP_CHECK(chp::cvp::sum_mass_soa(big_soa) ==
              chp::cvp::sum_mass_ptr(big_ptr));

    // Empty containers.
    const std::vector<chp::cvp::Particle> empty;
    std::vector<std::unique_ptr<chp::cvp::Particle>> empty_ptr;
    CHP_CHECK(chp::cvp::sum_mass_soa(empty) == 0.0);
    CHP_CHECK(chp::cvp::sum_mass_ptr(empty_ptr) == 0.0);

    return chp::test_summary("contiguous_vs_pointer");
}
