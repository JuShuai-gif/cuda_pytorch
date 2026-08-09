// Performance: comparing concatenated strings with and without a proxy.
//
// The book (PDF p.263) reports ~10.7x on 100 million comparisons on an
// Intel i7-7700k. This benchmark reproduces the experiment locally and
// reports the measured ratio for this machine.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "benchmark.hpp"

namespace {

struct StringWithProxy {
    std::string str;
};

// --- Baseline: (a + b) == c, allocates a temporary string each time ---
bool naive_equal(const StringWithProxy& a, const StringWithProxy& b,
                 const StringWithProxy& c) {
    return (a.str + b.str) == c.str;
}

struct ConcatProxy {
    const std::string& a;
    const std::string& b;
};

bool is_concat_equal(const std::string& a, const std::string& b,
                     const std::string& c) {
    return a.size() + b.size() == c.size() &&
           std::equal(a.begin(), a.end(), c.begin()) &&
           std::equal(b.begin(), b.end(), c.begin() + a.size());
}

auto operator+(const StringWithProxy& a, const StringWithProxy& b) {
    return ConcatProxy{a.str, b.str};
}

auto operator==(ConcatProxy&& concat, const StringWithProxy& rhs) -> bool {
    return is_concat_equal(concat.a, concat.b, rhs.str);
}

constexpr std::size_t kCount = 100'000;
constexpr std::size_t kIterations = 50;
constexpr std::size_t kRounds = 7;
constexpr std::size_t kWarmup = 2;

}  // namespace

int main() {
    std::printf("== concat_proxy benchmark ==\n");

    std::mt19937 gen(42u);
    std::uniform_int_distribution<int> len(5, 40);

    auto random_string = [&] {
        const int n = len(gen);
        std::string s;
        s.reserve(static_cast<std::size_t>(n) + 1);
        for (int i = 0; i < n; ++i) {
            s.push_back(static_cast<char>('a' + gen() % 26));
        }
        return s;
    };

    // c[i] is intentionally a+b so roughly half the comparisons are equal.
    std::vector<StringWithProxy> a(kCount), b(kCount), c(kCount);
    for (std::size_t i = 0; i < kCount; ++i) {
        a[i].str = random_string();
        b[i].str = random_string();
        c[i].str = a[i].str + b[i].str;
        if (i % 2 == 0) {
            a[i].str.push_back('x');  // make this pair unequal
        }
    }

    const auto r_naive = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            for (std::size_t i = 0; i < kCount; ++i) {
                acc += naive_equal(a[i], b[i], c[i]) ? 1u : 0u;
            }
        });
    const auto r_proxy = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            for (std::size_t i = 0; i < kCount; ++i) {
                acc += ((a[i] + b[i]) == c[i]) ? 1u : 0u;
            }
        });

    std::printf("Data: %zu string triples, %zu full passes per iteration\n\n",
                kCount, kIterations);

    chp::print_result("naive (a + b) == c", r_naive);
    chp::print_result("proxy  (a + b) == c", r_proxy);

    const double ratio = r_naive.mean_ns / r_proxy.mean_ns;
    std::printf("naive/proxy time ratio: %.2fx\n", ratio);
    std::printf("checksums match: %s\n",
                r_naive.checksum == r_proxy.checksum ? "yes" : "NO");

    return 0;
}
