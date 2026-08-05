#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace chp {
namespace avs {

// The "fat" User: everything inline (book PDF p.122-123). ~128 bytes on
// x86-64 because of the five std::string members.
struct BigUser {
    std::string name;
    std::string username;
    std::string password;
    std::string security_question;
    std::string security_answer;
    short level = 0;
    bool is_playing = false;
};

// Authentication info split out (book PDF p.123-124). The User shrinks to
// ~40 bytes: name + a pointer + level + flag.
struct AuthInfo {
    std::string username;
    std::string password;
    std::string security_question;
    std::string security_answer;
};

struct SmallUser {
    std::string name;
    std::unique_ptr<AuthInfo> auth;
    short level = 0;
    bool is_playing = false;
};

// Count users at a given level (book PDF p.123).
std::size_t num_users_at_level(const std::vector<BigUser>& users, short level);
std::size_t num_users_at_level(const std::vector<SmallUser>& users, short level);

// Count playing users (book PDF p.123).
std::size_t num_playing_users(const std::vector<BigUser>& users);
std::size_t num_playing_users(const std::vector<SmallUser>& users);

}  // namespace avs
}  // namespace chp
