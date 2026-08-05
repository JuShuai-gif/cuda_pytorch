#include <cstdio>
#include <forward_list>
#include <list>
#include <vector>

#include "test_utils.hpp"

int main() {
    // All containers hold the same elements; iteration sums must match.
    std::vector<int> vec = {1, 2, 3, 4, 5};
    std::list<int> list = {1, 2, 3, 4, 5};
    std::forward_list<int> flist = {1, 2, 3, 4, 5};

    int sv = 0, sl = 0, sf = 0;
    for (int x : vec) sv += x;
    for (int x : list) sl += x;
    for (int x : flist) sf += x;
    CHP_CHECK(sv == sl && sl == sf && sv == 15);

    // forward_list has no back(): forward iteration only.
    int count = 0;
    for (auto it = flist.begin(); it != flist.end(); ++it) {
        ++count;
    }
    CHP_CHECK(count == 5);

    // list is bidirectional.
    auto rit = list.rbegin();
    CHP_CHECK(*rit == 5);

    return chp::test_summary("sequence_containers");
}
