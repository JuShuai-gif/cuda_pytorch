#include "baseline.hpp"

#include <algorithm>

namespace chp {
namespace avs {

std::size_t num_users_at_level(const std::vector<BigUser>& users, short level) {
    return static_cast<std::size_t>(
        std::count_if(users.begin(), users.end(),
                      [level](const BigUser& u) { return u.level == level; }));
}

std::size_t num_users_at_level(const std::vector<SmallUser>& users,
                               short level) {
    return static_cast<std::size_t>(
        std::count_if(users.begin(), users.end(),
                      [level](const SmallUser& u) { return u.level == level; }));
}

std::size_t num_playing_users(const std::vector<BigUser>& users) {
    return static_cast<std::size_t>(
        std::count_if(users.begin(), users.end(),
                      [](const BigUser& u) { return u.is_playing; }));
}

std::size_t num_playing_users(const std::vector<SmallUser>& users) {
    return static_cast<std::size_t>(
        std::count_if(users.begin(), users.end(),
                      [](const SmallUser& u) { return u.is_playing; }));
}

}  // namespace avs
}  // namespace chp
