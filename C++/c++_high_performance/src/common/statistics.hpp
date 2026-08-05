#pragma once

#include <cstddef>
#include <vector>

namespace chp {

struct Statistics {
    double mean = 0.0;
    double median = 0.0;
    double min = 0.0;
    double max = 0.0;
    double stddev = 0.0;
};

// Computes descriptive statistics over a set of time samples (in ns).
// Requires at least one sample; empty input yields all-zero statistics.
Statistics compute_statistics(const std::vector<double>& samples);

}  // namespace chp
