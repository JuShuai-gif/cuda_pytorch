#include <cstdio>
#include <vector>

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

// Explicitly const: the compiler rejects any attempt to mutate `teams`.
int total_age(const std::vector<Team>& teams) {
    int total = 0;
    for (const auto& team : teams) {
        total += team.leader().age();
    }
    return total;
}

// Mutable: allowed to call the non-const overload of leader().
void make_all_20(std::vector<Team>& teams) {
    for (auto& team : teams) {
        team.leader().set_age(20);
    }
}

}  // namespace

int main() {
    std::printf("== const correctness ==\n");

    std::vector<Team> teams(3);
    make_all_20(teams);
    teams[0].leader().set_age(30);
    teams[1].leader().set_age(40);

    std::printf("total age (mutable vector): %d\n", total_age(teams));

    // const teams -> only const members accessible, no mutation possible.
    const auto& const_ref = teams;
    std::printf("total age (const ref):     %d\n", total_age(const_ref));
    return 0;
}
