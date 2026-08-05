// Iterator categories and iterator_traits (book PDF p.129-135).
//
// Categories (in order of capability):
//   input_iterator / output_iterator
//   forward_iterator
//   bidirectional_iterator
//   random_access_iterator
//   contiguous_iterator (C++20)
//
// std::iterator_traits exposes the five associated types of any iterator
// (including raw pointers). The category tag drives compile-time dispatch.

#include <cstdio>
#include <iterator>
#include <list>
#include <map>
#include <type_traits>
#include <vector>

namespace {

template <typename Iterator>
void report(const char* name) {
    using Category = typename std::iterator_traits<Iterator>::iterator_category;
    std::printf("%-18s category = ", name);
    if (std::is_same_v<Category, std::random_access_iterator_tag>) {
        std::printf("random_access\n");
    } else if (std::is_same_v<Category, std::bidirectional_iterator_tag>) {
        std::printf("bidirectional\n");
    } else if (std::is_same_v<Category, std::forward_iterator_tag>) {
        std::printf("forward\n");
    } else if (std::is_same_v<Category, std::input_iterator_tag>) {
        std::printf("input\n");
    } else {
        std::printf("other\n");
    }
}

}  // namespace

int main() {
    std::printf("== iterator_categories ==\n\n");

    report<int*>("int* (raw pointer)");
    report<std::vector<int>::iterator>("vector<int>::iterator");
    report<std::vector<int>::const_iterator>("vector<int>::const_iterator");
    report<std::list<int>::iterator>("list<int>::iterator");
    report<std::map<int, int>::iterator>("map<int,int>::iterator");

    // Raw pointers satisfy the same traits interface.
    static_assert(
        std::is_same_v<std::iterator_traits<int*>::value_type, int>,
        "pointer value_type is int");
    static_assert(std::is_same_v<std::iterator_traits<int*>::difference_type,
                                 std::ptrdiff_t>,
                  "pointer difference_type is ptrdiff_t");

    // The vector iterator is a random access iterator, list is not.
    static_assert(
        std::is_base_of_v<std::random_access_iterator_tag,
                          std::iterator_traits<std::vector<int>::iterator>::
                              iterator_category>,
        "vector iterator is random access");
    static_assert(
        !std::is_base_of_v<std::random_access_iterator_tag,
                           std::iterator_traits<std::list<int>::iterator>::
                               iterator_category>,
        "list iterator is NOT random access");

    std::printf("\ncategory checks passed\n");
    return 0;
}
