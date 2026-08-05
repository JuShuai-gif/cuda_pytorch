#include <cstdio>
#include <vector>

#include "baseline.hpp"
#include "optimized.hpp"
#include "test_utils.hpp"

int main() {
    // Build small data and verify all representations agree.
    std::vector<chp::avs::BigUser> big(10);
    std::vector<chp::avs::SmallUser> small(10);
    std::vector<short> levels(10);
    std::vector<bool> playing(10);

    for (int i = 0; i < 10; ++i) {
        big[i].level = static_cast<short>(i % 3);
        big[i].is_playing = (i % 2) == 0;
        small[i].level = big[i].level;
        small[i].is_playing = big[i].is_playing;
        levels[i] = big[i].level;
        playing[i] = big[i].is_playing;
    }

    for (short level = 0; level < 3; ++level) {
        const std::size_t a = chp::avs::num_users_at_level(big, level);
        const std::size_t b = chp::avs::num_users_at_level(small, level);
        const std::size_t c = chp::avs::num_users_at_level(levels, level);
        CHP_CHECK(a == b && b == c);
    }

    const std::size_t pb = chp::avs::num_playing_users(big);
    const std::size_t ps = chp::avs::num_playing_users(small);
    const std::size_t po = chp::avs::num_playing_users(playing);
    CHP_CHECK(pb == ps && ps == po);
    CHP_CHECK(pb == 5);

    // Empty containers.
    CHP_CHECK(chp::avs::num_users_at_level(std::vector<short>{}, 1) == 0);
    CHP_CHECK(chp::avs::num_playing_users(std::vector<bool>{}) == 0);

    return chp::test_summary("aos_vs_soa");
}
