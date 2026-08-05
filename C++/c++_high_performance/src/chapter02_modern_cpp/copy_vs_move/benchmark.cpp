#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "baseline.hpp"
#include "benchmark.hpp"

namespace {

constexpr std::size_t kSize = 4096;
constexpr std::size_t kIterations = 50'000;
constexpr std::size_t kRounds = 7;
constexpr std::size_t kWarmup = 2;

}  // namespace

int main() {
    std::printf("== copy_vs_move benchmark ==\n");
    std::printf("Copy/move of a Buffer owning %zu doubles.\n\n", kSize);

    chp::cvm::Buffer source(kSize);

    const auto r_copy = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            chp::cvm::Buffer copy(source);  // copy: allocate + copy data
            acc += static_cast<std::uint64_t>(copy.size());
        });
    const auto r_move = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            chp::cvm::Buffer copy(std::move(source));  // move: steal pointer
            source = std::move(copy);                  // restore for next iter
            acc += static_cast<std::uint64_t>(copy.size());
        });

    chp::print_result("copy-construct (allocate + copy data)", r_copy);
    chp::print_result("move-construct (steal pointer)", r_move);

    const double ratio = r_copy.mean_ns / r_move.mean_ns;
    std::printf("copy/move time ratio: %.2fx\n", ratio);
    return 0;
}
