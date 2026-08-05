#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include "baseline.hpp"
#include "benchmark.hpp"
#include "optimized.hpp"

namespace {

constexpr std::size_t kCount = 2'000'000;
constexpr std::size_t kIterations = 5;
constexpr std::size_t kRounds = 7;
constexpr std::size_t kWarmup = 2;

}  // namespace

int main() {
    std::printf("== contiguous_vs_pointer benchmark ==\n");

    std::mt19937 gen(2024u);
    std::uniform_real_distribution<float> dist(0.0F, 100.0F);

    // Identical data for both layouts: one mass per particle index.
    std::vector<float> masses(kCount);
    for (std::size_t i = 0; i < kCount; ++i) {
        masses[i] = dist(gen);
    }

    // Contiguous storage.
    std::vector<chp::cvp::Particle> soa(kCount);
    for (std::size_t i = 0; i < kCount; ++i) {
        soa[i].mass = masses[i];
    }

    // Pointer storage: every particle is a separate heap allocation.
    std::vector<std::unique_ptr<chp::cvp::Particle>> ptr;
    ptr.reserve(kCount);
    for (std::size_t i = 0; i < kCount; ++i) {
        auto p = std::make_unique<chp::cvp::Particle>();
        p->mass = masses[i];
        ptr.push_back(std::move(p));
    }

    const auto r_ptr = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            acc += static_cast<std::uint64_t>(
                chp::cvp::sum_mass_ptr(ptr) * 1e9);
        });
    const auto r_soa = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            acc += static_cast<std::uint64_t>(
                chp::cvp::sum_mass_soa(soa) * 1e9);
        });

    std::printf("Data: %zu particles, stored contiguously vs as %zu separate "
                "heap allocations\n\n", kCount, kCount);

    chp::print_result("vector<Particle> (contiguous)", r_soa);
    chp::print_result("vector<unique_ptr<Particle>> (pointer indirection)", r_ptr);

    const double ratio = r_ptr.mean_ns / r_soa.mean_ns;
    std::printf("pointer/contiguous time ratio: %.2fx\n", ratio);

    return 0;
}
