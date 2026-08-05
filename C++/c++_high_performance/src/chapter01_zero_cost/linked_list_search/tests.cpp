#include <cstdio>
#include <list>
#include <string>
#include <vector>

#include "baseline.hpp"
#include "optimized.hpp"
#include "test_utils.hpp"

int main() {
    // Two equivalent book lists with a known number of "Hamlet" entries.
    const std::vector<std::string> books = {"Hamlet", "Macbeth", "Hamlet",
                                            "King Lear", "Hamlet"};
    const std::list<std::string> book_list(books.begin(), books.end());

    // Build an equivalent C-style linked list.
    std::vector<chp::lls::CNode> nodes(books.size());
    for (std::size_t i = 0; i < books.size(); ++i) {
        nodes[i].title = books[i].c_str();
        nodes[i].next = (i + 1 < books.size()) ? &nodes[i + 1] : nullptr;
    }

    const std::string hamlet = "Hamlet";
    const std::size_t expected = 3;

    CHP_CHECK(chp::lls::count_title_c_style(&nodes[0], hamlet.c_str()) ==
              expected);
    CHP_CHECK(chp::lls::count_title_stl_vector(books, hamlet) == expected);
    CHP_CHECK(chp::lls::count_title_stl_list(book_list, hamlet) == expected);

    CHP_CHECK(chp::lls::count_title_c_style(&nodes[0], "Macbeth") == 1);
    CHP_CHECK(chp::lls::count_title_stl_vector(books, "Missing") == 0);

    // Empty containers / empty list edge cases.
    const std::vector<std::string> empty;
    CHP_CHECK(chp::lls::count_title_stl_vector(empty, hamlet) == 0);

    return chp::test_summary("linked_list_search");
}
