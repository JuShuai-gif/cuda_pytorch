// iterator_distance: choose an implementation based on the iterator category.
//
// The book (PDF p.135) shows how std::iterator_traits lets a template function
// branch at compile time: random access iterators can subtract directly
// (O(1)); all other categories must count steps (O(n)).

#include <cstddef>
#include <cstdio>
#include <iterator>
#include <list>
#include <type_traits>
#include <vector>

namespace {

template <typename Iterator>
auto iterator_distance(Iterator a, Iterator b) {
    using Traits = std::iterator_traits<Iterator>;
    using Category = typename Traits::iterator_category;
    using Difference = typename Traits::difference_type;
    if constexpr (std::is_same_v<Category, std::random_access_iterator_tag>) {
        return b - a;  // O(1)
    } else {
        Difference steps = 0;
        while (a != b) {
            ++steps;
            ++a;
        }
        return steps;  // O(n)
    }
}

}  // namespace

int main() {
    std::printf("== iterator_distance ==\n");

    std::vector<int> vec = {0, 1, 2, 3, 4};
    std::list<int> lst = {0, 1, 2, 3, 4};

    const auto d_vec = iterator_distance(vec.begin(), vec.end());
    const auto d_lst = iterator_distance(lst.begin(), lst.end());
    const auto d_ptr = iterator_distance(&vec[0], &vec[0] + vec.size());

    std::printf("vector distance: %td\n", d_vec);
    std::printf("list   distance: %td\n", d_lst);
    std::printf("pointer distance: %td\n", d_ptr);

    return d_vec == 5 && d_lst == 5 && d_ptr == 5 ? 0 : 1;
}
