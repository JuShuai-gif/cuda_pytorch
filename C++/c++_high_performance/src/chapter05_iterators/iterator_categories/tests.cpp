#include <cstddef>
#include <iterator>
#include <list>
#include <type_traits>
#include <vector>

#include "test_utils.hpp"

int main() {
    using VecIt = std::vector<int>::iterator;
    using ListIt = std::list<int>::iterator;

    // Random access iterators support arithmetic: +, -, +=, subscript.
    std::vector<int> v = {1, 2, 3, 4, 5};
    VecIt it = v.begin();
    CHP_CHECK(*(it + 2) == 3);
    CHP_CHECK(it[4] == 5);
    CHP_CHECK((v.end() - v.begin()) == 5);

    // Bidirectional iterators support ++ and -- only.
    std::list<int> l = {1, 2, 3};
    ListIt lit = l.begin();
    ++lit;
    --lit;
    CHP_CHECK(*lit == 1);

    // Categories are ordered by capability.
    static_assert(std::is_base_of_v<std::forward_iterator_tag,
                                    std::bidirectional_iterator_tag>,
                  "bidirectional is a forward iterator");
    static_assert(std::is_base_of_v<std::bidirectional_iterator_tag,
                                    std::random_access_iterator_tag>,
                  "random access is a bidirectional iterator");

    return chp::test_summary("iterator_categories");
}
