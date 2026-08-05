// Custom comparators and general-purpose predicates.
//
// The book (PDF p.149-151): algorithms default to operator== / operator<,
// but a custom comparator or predicate can be passed explicitly. Building a
// small namespace of named predicates keeps the calling code readable.

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <string>
#include <vector>

namespace {

namespace preds {

// Compare strings by length (book PDF p.150).
auto less_by_size = [](const std::string& a, const std::string& b) {
    return a.size() < b.size();
};

// Factory returning a predicate "size == n" (book PDF p.150).
auto equal_by_size = [](std::size_t size) {
    return [size](const auto& v) { return v.size() == size; };
};

// Case-insensitive string equality (book PDF p.151).
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

}  // namespace

int main() {
    std::printf("== comparators ==\n");

    std::vector<std::string> names = {"Ralph", "Lisa", "Homer",
                                      "Maggie", "Apu", "Bart"};

    // Sort by length (book PDF p.150).
    std::sort(names.begin(), names.end(), preds::less_by_size);
    std::printf("sorted by size: ");
    for (const auto& n : names) {
        std::printf("%s ", n.c_str());
    }
    std::printf("\n");

    // find_if by length.
    auto it = std::find_if(names.begin(), names.end(), preds::equal_by_size(3));
    std::printf("first name of size 3: %s\n",
                it != names.end() ? it->c_str() : "(none)");

    // Case-insensitive count (book PDF p.151).
    const auto n_maggies = std::count_if(
        names.begin(), names.end(), preds::equal_case_insensitive("maggie"));
    std::printf("case-insensitive 'maggie' count: %zu\n", n_maggies);

    return 0;
}
