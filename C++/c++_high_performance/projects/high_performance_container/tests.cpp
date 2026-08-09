// Correctness checks for HashSet.

#include <cstdio>
#include <string>
#include <vector>

#include "hash_set.hpp"

namespace chp {
inline int g_test_failures = 0;
}  // namespace chp

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "[FAIL] %s:%d: %s\n", __FILE__, __LINE__,   \
                         #cond);                                             \
            ++chp::g_test_failures;                                          \
        }                                                                    \
    } while (0)

int main() {

    chp::HashSet set;
    CHECK(set.empty());
    CHECK(set.size() == 0);

    // Insert returns true on new, false on duplicate.
    CHECK(set.insert("a"));
    CHECK(set.insert("b"));
    CHECK(!set.insert("a"));
    CHECK(set.size() == 2);
    CHECK(!set.empty());

    // Contains.
    CHECK(set.contains("a"));
    CHECK(set.contains("b"));
    CHECK(!set.contains("c"));

    // Rehash: insert enough to force growth, all still findable.
    for (int i = 0; i < 1000; ++i) {
        set.insert("key" + std::to_string(i));
    }
    CHECK(set.contains("key0"));
    CHECK(set.contains("key999"));
    CHECK(set.contains("a"));   // survivors of rehash
    CHECK(!set.contains("key-1"));
    CHECK(set.size() == 1002);

    // collect() returns exactly size() unique entries.
    const auto entries = set.collect();
    CHECK(entries.size() == set.size());
    for (const auto& e : entries) {
        CHECK(set.contains(e));
    }

    // Large batch: every element present after many rehashes.
    chp::HashSet big;
    for (int i = 0; i < 50'000; ++i) {
        big.insert(std::to_string(i));
    }
    CHECK(big.size() == 50'000);
    for (int i = 0; i < 50'000; i += 997) {
        CHECK(big.contains(std::to_string(i)));
    }
    CHECK(!big.contains(std::to_string(50'000)));

    if (chp::g_test_failures == 0) {
        std::printf("[PASS] high_performance_container: all checks passed\n");
        return 0;
    }
    std::printf("[FAIL] high_performance_container: %d check(s) failed\n",
                chp::g_test_failures);
    return 1;
}
