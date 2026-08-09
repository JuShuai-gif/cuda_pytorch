// Performance: nearest-point search with and without DistProxy.
//
// The book (PDF p.273) reports ~2x on an Intel i7-7700k. Reproduced locally;
// the reported ratio is machine specific.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "benchmark.hpp"

namespace chp9 {

class DistProxy {
public:
    DistProxy(float x0, float y0, float x1, float y1)
        : dist_sqrd_{std::pow(x0 - x1, 2.0F) + std::pow(y0 - y1, 2.0F)} {}
    auto operator<(const DistProxy& other) const {
        return dist_sqrd_ < other.dist_sqrd_;
    }
    operator float() const&& { return std::sqrt(dist_sqrd_); }

private:
    float dist_sqrd_{};
};

class Point {
public:
    Point(float x, float y) : x_(x), y_(y) {}
    auto distance(const Point& p) const {
        return DistProxy{x_, y_, p.x_, p.y_};
    }
    float x() const { return x_; }
    float y() const { return y_; }

private:
    float x_{};
    float y_{};
};

}  // namespace chp9

namespace {

float distance_sqrt(const chp9::Point& a, const chp9::Point& b) {
    const float dx = a.x() - b.x();
    const float dy = a.y() - b.y();
    return std::sqrt(dx * dx + dy * dy);
}

constexpr std::size_t kPoints = 10'000;
constexpr std::size_t kIterations = 200;
constexpr std::size_t kRounds = 7;
constexpr std::size_t kWarmup = 2;

}  // namespace

int main() {
    std::printf("== distance_proxy benchmark ==\n");

    std::mt19937 gen(7u);
    std::uniform_real_distribution<float> dist(0.0F, 1000.0F);

    std::vector<chp9::Point> points;
    points.reserve(kPoints);
    for (std::size_t i = 0; i < kPoints; ++i) {
        points.emplace_back(dist(gen), dist(gen));
    }
    const chp9::Point needle{135.0F, 246.0F};

    // Baseline: sqrt on every comparison.
    const auto r_naive = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            float best = 0.0F;
            for (const auto& p : points) {
                const float d = distance_sqrt(p, needle);
                best = (best == 0.0F || d < best) ? d : best;
            }
            acc += static_cast<std::uint64_t>(best * 1e6);
        });

    // Proxy: comparisons avoid sqrt; only the final distance is computed.
    const auto r_proxy = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            const chp9::Point* best = nullptr;
            for (const auto& p : points) {
                if (best == nullptr || p.distance(needle) < best->distance(needle)) {
                    best = &p;
                }
            }
            acc += static_cast<std::uint64_t>(best->distance(needle) * 1e6);
        });

    std::printf("Data: %zu random points, one nearest-neighbor search per "
                "iteration\n\n", kPoints);

    chp::print_result("naive (sqrt each compare)", r_naive);
    chp::print_result("proxy  (sqrt once)", r_proxy);

    const double ratio = r_naive.mean_ns / r_proxy.mean_ns;
    std::printf("naive/proxy time ratio: %.2fx\n", ratio);

    return 0;
}
