#include <cstdio>
#include <type_traits>
#include <vector>

#include "test_utils.hpp"

namespace {

class Person {
public:
    int age() const { return age_; }
    void set_age(int age) { age_ = age; }

private:
    int age_ = 0;
};

class Team {
public:
    const Person& leader() const { return leader_; }
    Person& leader() { return leader_; }

private:
    Person leader_;
};

int total_age(const std::vector<Team>& teams) {
    int total = 0;
    for (const auto& team : teams) {
        total += team.leader().age();
    }
    return total;
}

}  // namespace

int main() {
    std::vector<Team> teams(3);
    teams[0].leader().set_age(30);
    teams[1].leader().set_age(40);
    teams[2].leader().set_age(50);

    CHP_CHECK(total_age(teams) == 120);

    const auto& const_teams = teams;
    CHP_CHECK(const_teams[0].leader().age() == 30);
    CHP_CHECK(total_age(const_teams) == 120);

    // The const overload must not expose mutation.
    static_assert(std::is_same<decltype(teams[0].leader()), Person&>::value,
                  "mutable overload returns Person&");
    static_assert(
        std::is_same<decltype(const_teams[0].leader()), const Person&>::value,
        "const overload returns const Person&");

    return chp::test_summary("const_correctness");
}
