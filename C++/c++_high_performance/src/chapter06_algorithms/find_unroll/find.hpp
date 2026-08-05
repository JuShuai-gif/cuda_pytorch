#pragma once

#include <cstddef>
#include <iterator>

namespace chp {
namespace fu {

// A naive std::find implementation (book PDF p.158).
template <typename It, typename Value>
It find_slow(It first, It last, const Value& value) {
    for (auto it = first; it != last; ++it) {
        if (*it == value) {
            return it;
        }
    }
    return last;
}

// The libstdc++ trick: unroll the loop in chunks of four and compare the
// trip count against zero (book PDF p.158-159).
template <typename It, typename Value>
It find_fast(It first, It last, const Value& value) {
    const auto num_trips = (last - first) / 4;
    for (auto trip_count = num_trips; trip_count > 0; --trip_count) {
        if (*first == value) { return first; }
        ++first;
        if (*first == value) { return first; }
        ++first;
        if (*first == value) { return first; }
        ++first;
        if (*first == value) { return first; }
        ++first;
    }
    switch (last - first) {
        case 3:
            if (*first == value) { return first; }
            ++first;
            [[fallthrough]];
        case 2:
            if (*first == value) { return first; }
            ++first;
            [[fallthrough]];
        case 1:
            if (*first == value) { return first; }
            ++first;
            [[fallthrough]];
        case 0:
        default:
            return last;
    }
}

}  // namespace fu
}  // namespace chp
