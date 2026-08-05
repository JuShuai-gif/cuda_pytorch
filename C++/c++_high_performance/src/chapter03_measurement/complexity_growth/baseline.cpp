#include "baseline.hpp"

namespace chp {
namespace cg {

bool linear_search_int(const std::vector<int>& vals, int key) {
    for (const int v : vals) {
        if (v == key) {
            return true;
        }
    }
    return false;
}

bool linear_search_point(const std::vector<Point>& pts, const Point& key) {
    for (std::size_t i = 0; i < pts.size(); ++i) {
        if (pts[i].x == key.x && pts[i].y == key.y) {
            return true;
        }
    }
    return false;
}

// Note: the book's version (PDF p.88) uses a closed range with unsigned
// size_t; when the key is smaller than every element, `high = mid - 1`
// underflows to SIZE_MAX and the loop runs out of bounds. We use a half-open
// range instead, which is correct for unsigned indices.
bool binary_search_int(const std::vector<int>& vals, int key) {
    std::size_t low = 0;
    std::size_t high = vals.size();
    while (low < high) {
        const std::size_t mid = low + ((high - low) / 2);
        if (vals[mid] < key) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low < vals.size() && vals[low] == key;
}

}  // namespace cg
}  // namespace chp
