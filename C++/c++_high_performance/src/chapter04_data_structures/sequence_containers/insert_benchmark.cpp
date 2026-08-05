// Insertion patterns for sequence containers.
//
// The book (PDF p.108-111) notes that the choice depends on usage pattern:
//  - vector: O(1) back insert (amortized), O(n) middle insert;
//  - deque:  O(1) back AND front insert;
//  - list:   O(1) insert anywhere IF you already have the iterator (O(n) to find).

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <list>
#include <vector>

#include "benchmark.hpp"

namespace {

constexpr std::size_t kCount = 200'000;
constexpr std::size_t kIterations = 30;
constexpr std::size_t kRounds = 5;
constexpr std::size_t kWarmup = 2;

// Push to the back.
std::uint64_t back_insert(std::vector<int>& v, std::deque<int>& d,
                          std::list<int>& l) {
    for (std::size_t i = 0; i < kCount; ++i) {
        v.push_back(static_cast<int>(i));
        d.push_back(static_cast<int>(i));
        l.push_back(static_cast<int>(i));
    }
    return v.back() + d.back() + l.back();
}

// Insert in the middle.
std::uint64_t middle_insert(std::vector<int>& v, std::deque<int>& d,
                            std::list<int>& l) {
    const std::size_t mid_v = v.size() / 2;
    const std::size_t mid_d = d.size() / 2;
    auto mid_l = l.begin();
    for (std::size_t i = 0; i < l.size() / 2; ++i) {
        ++mid_l;
    }
    v.insert(v.begin() + static_cast<std::ptrdiff_t>(mid_v), 1);
    d.insert(d.begin() + static_cast<std::ptrdiff_t>(mid_d), 1);
    l.insert(mid_l, 1);  // O(1) once the iterator is there
    return v[mid_v] + d[mid_d] + *mid_l;
}

}  // namespace

int main() {
    std::printf("== sequence_containers insertion ==\n\n");

    std::vector<int> v(kCount, 0);
    std::deque<int> d(kCount, 0);
    std::list<int> l(kCount, 0);

    const auto r_back = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) { acc += back_insert(v, d, l); });
    chp::print_result("push_back: vector + deque + list", r_back);

    const auto r_mid = chp::benchmark(1, kRounds, kWarmup,
        [&](std::uint64_t& acc) { acc += middle_insert(v, d, l); });
    chp::print_result("single middle insert: vector + deque + list", r_mid);

    return 0;
}
