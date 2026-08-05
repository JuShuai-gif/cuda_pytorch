#include "optimized.hpp"

#include <algorithm>

namespace chp {
namespace avs {

std::size_t num_users_at_level(const std::vector<short>& levels, short level) {
    return static_cast<std::size_t>(
        std::count(levels.begin(), levels.end(), level));
}

std::size_t num_playing_users(const std::vector<bool>& playing) {
    return static_cast<std::size_t>(
        std::count(playing.begin(), playing.end(), true));
}

}  // namespace avs
}  // namespace chp
