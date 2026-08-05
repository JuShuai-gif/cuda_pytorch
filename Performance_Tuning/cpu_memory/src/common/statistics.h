#pragma once

// Minimal statistics helpers (kept in common for reuse).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace stats {

template <typename T>
double mean(const std::vector<T>& v) {
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (const auto& x : v) s += static_cast<double>(x);
    return s / static_cast<double>(v.size());
}

template <typename T>
double median(std::vector<T> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return static_cast<double>(v[v.size() / 2]);
}

template <typename T>
double stddev(const std::vector<T>& v) {
    if (v.size() < 2) return 0.0;
    double m = mean(v);
    double s = 0.0;
    for (const auto& x : v) {
        double d = static_cast<double>(x) - m;
        s += d * d;
    }
    return std::sqrt(s / static_cast<double>(v.size() - 1));
}

}  // namespace stats
