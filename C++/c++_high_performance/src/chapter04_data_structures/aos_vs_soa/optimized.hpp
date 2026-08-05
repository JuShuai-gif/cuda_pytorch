#pragma once

#include <cstddef>
#include <vector>

namespace chp {
namespace avs {

// Parallel arrays: only the field we iterate is stored contiguously.
// (book PDF p.124-127)
std::size_t num_users_at_level(const std::vector<short>& levels, short level);

// vector<bool> is a bit array; count is very fast (book PDF p.127).
std::size_t num_playing_users(const std::vector<bool>& playing);

}  // namespace avs
}  // namespace chp
