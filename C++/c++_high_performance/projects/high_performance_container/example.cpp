// Hash set demo: dedupe a stream of words.

#include <cstdio>
#include <string>
#include <vector>

#include "hash_set.hpp"

namespace {

const std::vector<std::string> kWords = {
    "the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog",
    "the", "quick", "brown", "fox",
};

}  // namespace

int main() {
    std::printf("== high_performance_container ==\n");

    chp::HashSet set;
    std::vector<std::string> inserted;
    for (const auto& w : kWords) {
        if (set.insert(w)) {
            inserted.push_back(w);
        }
    }

    std::printf("unique words (%zu): ", set.size());
    for (const auto& w : set.collect()) {
        std::printf("%s ", w.c_str());
    }
    std::printf("\n");

    std::printf("contains(\"fox\")=%d contains(\"cat\")=%d\n",
                set.contains("fox"), set.contains("cat"));
    std::printf("size after dup inserts: %zu\n", set.size());

    return 0;
}
