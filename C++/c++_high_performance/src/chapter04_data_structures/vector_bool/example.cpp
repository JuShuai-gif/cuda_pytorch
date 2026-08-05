// std::vector<bool> is a specialized bit array, not a vector of bools.
//
// The book (PDF p.127) notes that vector<bool> packs bits, enabling count/find
// to process 64 bits at a time, and that its future is uncertain (a dynamic
// bitset may replace it).

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <type_traits>
#include <vector>

#include "test_utils.hpp"

int main() {
    std::printf("== vector_bool ==\n");

    // A normal vector<bool> is a bitset internally.
    std::vector<bool> bits(1000, false);
    bits[0] = true;
    bits[999] = true;
    CHP_CHECK(bits[0] == true);
    CHP_CHECK(bits[1] == false);

    // count processes 64 bits at a time (fast).
    const auto n_true = std::count(bits.begin(), bits.end(), true);
    CHP_CHECK(n_true == 2);

    // It is NOT a standard vector: sizeof differs and references are proxies.
    std::printf("sizeof(vector<bool>) = %zu\n", sizeof(std::vector<bool>));
    std::printf("sizeof(vector<char>) = %zu\n", sizeof(std::vector<char>));

    // The reference type is a proxy object, not a real bool&.
    auto proxy = bits[0];
    static_assert(!std::is_same<decltype(proxy), bool&>::value,
                  "vector<bool> returns a proxy, not bool&");

    // flip() inverts all bits.
    bits.flip();
    CHP_CHECK(bits[0] == false);
    CHP_CHECK(bits[1] == true);

    std::printf("vector_bool: all checks passed\n");
    return chp::test_summary("vector_bool");
}
