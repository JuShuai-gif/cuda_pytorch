#include "statistics.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace chp {

Statistics compute_statistics(const std::vector<double>& samples) {
    Statistics stats{};
    if (samples.empty()) {
        return stats;
    }

    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    stats.min = sorted.front();
    stats.max = sorted.back();

    const std::size_t n = sorted.size();

    double sum = 0.0;
    for (double s : samples) {
        sum += s;
    }
    stats.mean = sum / static_cast<double>(n);

    if (n % 2 == 0) {
        stats.median = (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    } else {
        stats.median = sorted[n / 2];
    }

    double sq_sum = 0.0;
    for (double s : samples) {
        const double d = s - stats.mean;
        sq_sum += d * d;
    }
    stats.stddev = std::sqrt(sq_sum / static_cast<double>(n));

    return stats;
}

}  // namespace chp
