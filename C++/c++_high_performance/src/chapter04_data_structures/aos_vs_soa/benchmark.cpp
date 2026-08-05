// AoS vs SoA / parallel arrays (book PDF p.120-127).
//
// The book measures summing a field over 1M users:
//   BigUser (128 bytes)  -> ~11 ms
//   SmallUser (40 bytes) -> ~4 ms
//   levels as vector<short> (parallel array) -> ~0.7 ms
//   playing as vector<bool> (bit array)      -> ~0.03 ms

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

constexpr std::size_t kCount = 1'000'000;
constexpr std::size_t kIterations = 5;
constexpr std::size_t kRounds = 7;
constexpr std::size_t kWarmup = 2;

}  // namespace

int main() {
    std::printf("== aos_vs_soa ==\n");
    std::printf("sizeof(BigUser)=%zu  sizeof(SmallUser)=%zu\n",
                sizeof(chp::avs::BigUser), sizeof(chp::avs::SmallUser));

    std::mt19937 gen(2024u);
    std::uniform_int_distribution<int> level_dist(1, 20);
    std::uniform_int_distribution<int> bool_dist(0, 1);

    std::vector<chp::avs::BigUser> big_users(kCount);
    std::vector<chp::avs::SmallUser> small_users(kCount);
    std::vector<short> levels(kCount);
    std::vector<bool> playing(kCount);

    for (std::size_t i = 0; i < kCount; ++i) {
        big_users[i].level = static_cast<short>(level_dist(gen));
        big_users[i].is_playing = bool_dist(gen) == 1;
        small_users[i].level = big_users[i].level;
        small_users[i].is_playing = big_users[i].is_playing;
        levels[i] = big_users[i].level;
        playing[i] = big_users[i].is_playing;
    }

    const short target = 5;

    const auto r_big_level = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            acc += static_cast<std::uint64_t>(
                chp::avs::num_users_at_level(big_users, target));
        });
    const auto r_small_level = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            acc += static_cast<std::uint64_t>(
                chp::avs::num_users_at_level(small_users, target));
        });
    const auto r_soa_level = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            acc += static_cast<std::uint64_t>(
                chp::avs::num_users_at_level(levels, target));
        });

    const auto r_big_play = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            acc += static_cast<std::uint64_t>(
                chp::avs::num_playing_users(big_users));
        });
    const auto r_small_play = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            acc += static_cast<std::uint64_t>(
                chp::avs::num_playing_users(small_users));
        });
    const auto r_soa_play = chp::benchmark(kIterations, kRounds, kWarmup,
        [&](std::uint64_t& acc) {
            acc += static_cast<std::uint64_t>(
                chp::avs::num_playing_users(playing));
        });

    std::printf("\n-- num_users_at_level (1M users) --\n");
    chp::print_result("BigUser   (128 B, AoS)", r_big_level);
    chp::print_result("SmallUser (40 B, AoS)", r_small_level);
    chp::print_result("vector<short> levels (SoA)", r_soa_level);

    std::printf("\n-- num_playing_users (1M users) --\n");
    chp::print_result("BigUser   (128 B, AoS)", r_big_play);
    chp::print_result("SmallUser (40 B, AoS)", r_small_play);
    chp::print_result("vector<bool> playing (bit array)", r_soa_play);

    return 0;
}
