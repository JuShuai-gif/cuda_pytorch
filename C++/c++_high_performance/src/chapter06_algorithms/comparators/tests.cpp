#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "test_utils.hpp"

namespace preds {

auto less_by_size = [](const std::string& a, const std::string& b) {
    return a.size() < b.size();
};

auto equal_by_size = [](std::size_t size) {
    return [size](const auto& v) { return v.size() == size; };
};

auto equal_case_insensitive = [](const std::string& needle) {
    return [&needle](const std::string& s) {
        if (needle.size() != s.size()) {
            return false;
        }
        for (std::size_t i = 0; i < s.size(); ++i) {
            if (std::tolower(static_cast<unsigned char>(s[i])) !=
                std::tolower(static_cast<unsigned char>(needle[i]))) {
                return false;
            }
        }
        return true;
    };
};

}  // namespace preds

int main() {
    std::vector<std::string> names = {"Apu", "Lisa", "Bart", "Ralph"};

    // Custom comparator sorts by length.
    std::sort(names.begin(), names.end(), preds::less_by_size);
    CHP_CHECK(names.front().size() <= names.back().size());

    // equal_by_size predicate.
    auto it = std::find_if(names.begin(), names.end(), preds::equal_by_size(3));
    CHP_CHECK(it != names.end());
    CHP_CHECK(*it == "Apu");

    // Case-insensitive match.
    const std::size_t n = static_cast<std::size_t>(std::count_if(
        names.begin(), names.end(), preds::equal_case_insensitive("APU")));
    CHP_CHECK(n == 1);
    const std::size_t m = static_cast<std::size_t>(std::count_if(
        names.begin(), names.end(), preds::equal_case_insensitive("nope")));
    CHP_CHECK(m == 0);

    return chp::test_summary("comparators");
}
