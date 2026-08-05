#include <cstdint>
#include <cstdio>

#include "baseline.hpp"
#include "benchmark.hpp"
#include "optimized.hpp"

namespace {

constexpr std::size_t kIterations = 200'000;
constexpr std::size_t kRounds = 7;
constexpr std::size_t kWarmup = 2;

}  // namespace

int main() {
    std::printf("== raii_resource benchmark ==\n");
    std::printf("Both implementations allocate+release the same heap resource; "
                "the RAII guard adds no extra runtime cost of its own.\n\n");

    const auto r_manual = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            chp::raii::Resource* r = nullptr;
            acc += static_cast<std::uint64_t>(chp::raii::use_manual(r, 7));
            delete r;  // manual release
        });
    const auto r_raii = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            chp::raii::ResourceGuard guard;
            acc += static_cast<std::uint64_t>(chp::raii::use_raii(guard, 7));
        });

    chp::print_result("manual new/delete per call", r_manual);
    chp::print_result("RAII guard per call", r_raii);

    const double ratio = r_manual.mean_ns / r_raii.mean_ns;
    std::printf("manual/raii time ratio: %.2fx\n", ratio);
    return 0;
}
