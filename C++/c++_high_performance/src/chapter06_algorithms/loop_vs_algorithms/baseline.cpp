#include "baseline.hpp"

#include <numeric>

namespace chp {
namespace lva {

std::size_t count_loop(const std::vector<int>& v, int needle) {
    std::size_t n = 0;
    for (int x : v) {
        if (x == needle) {
            ++n;
        }
    }
    return n;
}

bool find_loop(const std::vector<int>& v, int needle) {
    for (int x : v) {
        if (x == needle) {
            return true;
        }
    }
    return false;
}

std::size_t count_if_loop(const std::vector<int>& v) {
    std::size_t n = 0;
    for (int x : v) {
        if (x % 7 == 0) {
            ++n;
        }
    }
    return n;
}

std::vector<int> transform_loop(const std::vector<int>& v) {
    std::vector<int> out;
    out.reserve(v.size());
    for (int x : v) {
        out.push_back(x * 3 + 1);
    }
    return out;
}

std::vector<int> copy_if_loop(const std::vector<int>& v) {
    std::vector<int> out;
    out.reserve(v.size());
    for (int x : v) {
        if (x > 50) {
            out.push_back(x);
        }
    }
    return out;
}

int accumulate_loop(const std::vector<int>& v) {
    int sum = 0;
    for (int x : v) {
        sum += x;
    }
    return sum;
}

}  // namespace lva
}  // namespace chp
