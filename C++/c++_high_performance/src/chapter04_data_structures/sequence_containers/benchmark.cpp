// Sequence containers: memory layout and access characteristics.
//
// The book (PDF p.108-111) explains:
//  - std::vector/std::array: contiguous -> cache friendly traversal;
//  - std::deque: fixed-size blocks -> O(1) index, but not fully contiguous;
//  - std::list/std::forward_list: node-based, pointer per element -> cache
//    misses while iterating; forward_list uses less memory (one pointer).

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <forward_list>
#include <list>
#include <vector>

#include "benchmark.hpp"

namespace {

constexpr std::size_t kCount = 2'000'000;
constexpr std::size_t kIterations = 3;
constexpr std::size_t kRounds = 5;
constexpr std::size_t kWarmup = 2;

struct Item {
    int value;
};

std::int64_t sum_vec(const std::vector<Item>& v) {
    std::int64_t sum = 0;
    for (const auto& it : v) {
        sum += it.value;
    }
    return sum;
}

std::int64_t sum_deque(const std::deque<Item>& d) {
    std::int64_t sum = 0;
    for (const auto& it : d) {
        sum += it.value;
    }
    return sum;
}

std::int64_t sum_list(const std::list<Item>& l) {
    std::int64_t sum = 0;
    for (const auto& it : l) {
        sum += it.value;
    }
    return sum;
}

std::int64_t sum_flist(const std::forward_list<Item>& l) {
    std::int64_t sum = 0;
    for (const auto& it : l) {
        sum += it.value;
    }
    return sum;
}

}  // namespace

int main() {
    std::printf("== sequence_containers ==\n");
    std::printf("Iterating %zu elements in each container.\n\n", kCount);

    std::vector<Item> vec(kCount, Item{1});
    std::deque<Item> deq(kCount, Item{1});
    std::list<Item> list(kCount, Item{1});
    std::forward_list<Item> flist(kCount, Item{1});

    std::printf("sizes: vector=%zu deque=%zu list=%zu forward_list=%zu bytes\n",
                sizeof(std::vector<Item>), sizeof(std::deque<Item>),
                sizeof(std::list<Item>), sizeof(std::forward_list<Item>));

    const auto r_vec = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            acc += static_cast<std::uint64_t>(sum_vec(vec));
        });
    const auto r_deq = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            acc += static_cast<std::uint64_t>(sum_deque(deq));
        });
    const auto r_list = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            acc += static_cast<std::uint64_t>(sum_list(list));
        });
    const auto r_flist = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            acc += static_cast<std::uint64_t>(sum_flist(flist));
        });

    chp::print_result("std::vector (contiguous)", r_vec);
    chp::print_result("std::deque (block-based)", r_deq);
    chp::print_result("std::list (doubly linked)", r_list);
    chp::print_result("std::forward_list (singly linked)", r_flist);

    if (r_vec.checksum == r_deq.checksum && r_deq.checksum == r_list.checksum &&
        r_list.checksum == r_flist.checksum) {
        std::printf("Checksums identical.\n");
        return 0;
    }
    std::printf("ERROR: checksums differ!\n");
    return 1;
}
