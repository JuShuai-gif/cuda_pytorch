#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <variant>
#include <vector>

#include "benchmark.hpp"

namespace {

constexpr std::size_t kCount = 1'000'000;
constexpr std::size_t kIterations = 10;
constexpr std::size_t kRounds = 7;
constexpr std::size_t kWarmup = 2;

// Sum over kCount values, "guaranteed present" in this workload, but each
// representation still has to be inspected for absence.
std::int64_t sum_sentinel(const std::vector<int>& v) {
    std::int64_t sum = 0;
    for (std::size_t i = 0; i < v.size(); ++i) {
        const int x = v[i];
        if (x != -1) {  // sentinel check
            sum += x;
        }
    }
    return sum;
}

std::int64_t sum_pointer(const std::vector<const int*>& v) {
    std::int64_t sum = 0;
    for (std::size_t i = 0; i < v.size(); ++i) {
        const int* p = v[i];
        if (p != nullptr) {
            sum += *p;
        }
    }
    return sum;
}

std::int64_t sum_optional(const std::vector<std::optional<int>>& v) {
    std::int64_t sum = 0;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (v[i].has_value()) {
            sum += *v[i];
        }
    }
    return sum;
}

std::int64_t sum_variant(const std::vector<std::variant<std::monostate, int>>& v) {
    std::int64_t sum = 0;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (std::holds_alternative<int>(v[i])) {
            sum += std::get<int>(v[i]);
        }
    }
    return sum;
}

}  // namespace

int main() {
    std::printf("== optional_variant_pointer benchmark ==\n");
    std::printf("Reading 1M values via four 'maybe-absent' representations.\n\n");

    std::vector<int> plain(kCount, 42);
    std::vector<const int*> ptrs;
    std::vector<std::optional<int>> opts;
    std::vector<std::variant<std::monostate, int>> vars;
    ptrs.reserve(kCount);
    opts.reserve(kCount);
    vars.reserve(kCount);
    for (std::size_t i = 0; i < kCount; ++i) {
        ptrs.push_back(&plain[i]);
        opts.push_back(std::optional<int>{plain[i]});
        vars.emplace_back(std::in_place_index<1>, plain[i]);
    }

    const auto r_sentinel = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            acc += static_cast<std::uint64_t>(sum_sentinel(plain));
        });
    const auto r_ptr = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            acc += static_cast<std::uint64_t>(sum_pointer(ptrs));
        });
    const auto r_opt = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            acc += static_cast<std::uint64_t>(sum_optional(opts));
        });
    const auto r_var = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            acc += static_cast<std::uint64_t>(sum_variant(vars));
        });

    chp::print_result("sentinel value (-1 check)", r_sentinel);
    chp::print_result("pointer (nullptr check)", r_ptr);
    chp::print_result("std::optional (has_value)", r_opt);
    chp::print_result("std::variant (holds_alternative)", r_var);

    if (r_sentinel.checksum == r_ptr.checksum &&
        r_ptr.checksum == r_opt.checksum && r_opt.checksum == r_var.checksum) {
        std::printf("Checksums identical.\n");
        return 0;
    }
    std::printf("ERROR: checksums differ!\n");
    return 1;
}
