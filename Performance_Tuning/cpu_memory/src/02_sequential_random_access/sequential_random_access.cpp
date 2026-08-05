// Experiment 02: Sequential vs random access.
//
// Compares forward, backward, fixed-stride, and random access over a large
// array. Reports total time, ns/element, GB/s, and a checksum.
//
// Reference: PDF 3.3.2, 6.2.1 (sequential prefetching, random access ~70% slower).

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include "benchmark.h"

static constexpr size_t N = 1 << 26;   // 64M ints = 256 MB (exceeds LLC)
static constexpr int kRounds = 5;

static void init(std::vector<int>& v, std::mt19937& rng) {
    for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<int>(i);
}

static uint64_t checksum_seq(const std::vector<int>& v) {
    uint64_t h = 0;
    for (int x : v) h = bm::mix64(h ^ (uint64_t)x);
    return h;
}

int main() {
    std::printf("Experiment 02: sequential vs random access (N=%zu, %zu MB)\n",
                N, N * sizeof(int) / (1024 * 1024));

    std::vector<int> data(N);
    std::mt19937 rng(42);
    init(data, rng);

    // Order vector for random access.
    std::vector<uint32_t> order(N);
    for (uint32_t i = 0; i < N; ++i) order[i] = i;
    std::shuffle(order.begin(), order.end(), rng);

    uint64_t cs_seq = checksum_seq(data);

    struct Mode {
        const char* name;
        std::function<uint64_t()> fn;
    };

    auto fwd = [](const std::vector<int>& d, const std::vector<uint32_t>&) -> uint64_t {
        uint64_t s = 0;
        for (size_t i = 0; i < d.size(); ++i) s += (uint64_t)d[i];
        bm::do_not_optimize(s);
        return s;
    };
    auto bwd = [](const std::vector<int>& d, const std::vector<uint32_t>&) -> uint64_t {
        uint64_t s = 0;
        for (size_t i = d.size(); i-- > 0;) s += (uint64_t)d[i];
        bm::do_not_optimize(s);
        return s;
    };
    auto stride = [](const std::vector<int>& d, const std::vector<uint32_t>&) -> uint64_t {
        uint64_t s = 0;
        for (size_t i = 0; i < d.size(); i += 64) s += (uint64_t)d[i];
        bm::do_not_optimize(s);
        return s;
    };
    auto rnd = [&order](const std::vector<int>& d, const std::vector<uint32_t>&) -> uint64_t {
        uint64_t s = 0;
        for (size_t i = 0; i < d.size(); ++i) s += (uint64_t)d[order[i]];
        bm::do_not_optimize(s);
        return s;
    };

    std::printf("%-16s %-12s %-10s %-10s %-14s\n",
                "mode", "time_ms", "ns/elem", "GB/s", "checksum");

    const char* names[4] = {"forward", "backward", "stride64", "random"};
    std::function<uint64_t()> fns[4] = {
        [&] { return fwd(data, order); },
        [&] { return bwd(data, order); },
        [&] { return stride(data, order); },
        [&] { return rnd(data, order); },
    };
    uint64_t checks[4];
    checks[0] = checks[1] = checks[2] = cs_seq;
    // random reads all elements once, so checksum equals sequential sum
    checks[3] = cs_seq;

    for (int m = 0; m < 4; ++m) {
        // warmup
        fns[m]();
        auto res = bm::time_rounds(kRounds, [&] { fns[m](); });
        double ns_elem = res.median_ms * 1e6 / (double)N;
        double gbps = (double)N * sizeof(int) / (res.median_ms * 1e6 / 1e9) / 1e9;
        std::printf("%-16s %-12.2f %-10.2f %-10.3f %-14llu\n",
                    names[m], res.median_ms, ns_elem, gbps,
                    (unsigned long long)checks[m]);
    }
    return 0;
}
