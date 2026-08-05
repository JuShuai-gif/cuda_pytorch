#include <cstdio>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "test_utils.hpp"

int main() {
    std::map<int, int> om;
    std::unordered_map<int, int> um;
    std::set<int> os;
    std::unordered_set<int> us;

    for (int i = 0; i < 100; ++i) {
        om[i] = i * 2;
        um[i] = i * 2;
        os.insert(i);
        us.insert(i);
    }

    CHP_CHECK(om.size() == 100);
    CHP_CHECK(um.size() == 100);
    CHP_CHECK(os.size() == 100);
    CHP_CHECK(us.size() == 100);

    CHP_CHECK(om.at(42) == 84);
    CHP_CHECK(um.at(42) == 84);
    CHP_CHECK(os.count(42) == 1);
    CHP_CHECK(us.count(42) == 1);

    // Inserting duplicates does not grow the container.
    os.insert(42);
    us.insert(42);
    CHP_CHECK(os.size() == 100);
    CHP_CHECK(us.size() == 100);

    // Missing keys.
    CHP_CHECK(om.count(500) == 0);
    CHP_CHECK(um.count(500) == 0);

    return chp::test_summary("associative_containers");
}
