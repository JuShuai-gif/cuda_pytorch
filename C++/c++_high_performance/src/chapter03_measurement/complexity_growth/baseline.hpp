#pragma once

#include <cstddef>
#include <vector>

namespace chp {
namespace cg {

struct Point {
    int x = 0;
    int y = 0;
};

// Linear search over ints (book PDF p.86).
bool linear_search_int(const std::vector<int>& vals, int key);

// Linear search over Points (book PDF p.87).
bool linear_search_point(const std::vector<Point>& pts, const Point& key);

// Binary search over sorted ints (book PDF p.88).
bool binary_search_int(const std::vector<int>& vals, int key);

}  // namespace cg
}  // namespace chp
