// Assembly-inspection example.
//
// This file defines near-equivalent operations in "C style" and "STL style".
// The functions are marked noinline so their bodies appear in the emitted
// assembly instead of being inlined at the call site. Compare the generated
// machine code to see whether the C++ abstraction has any runtime cost:
//
//   g++ -std=c++17 -O3 -S example.cpp
//   clang++ -std=c++17 -O3 -S example.cpp
//
// Look for the labels of count_loop / count_algo and count_c_style /
// count_strings. On -O3 the two members of each pair typically compile to
// the same (or equivalent) loop; the STL versions get fully inlined.

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

[[gnu::noinline]] std::size_t count_loop(const std::vector<int>& v,
                                         int needle) {
    std::size_t n = 0;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (v[i] == needle) {
            ++n;
        }
    }
    return n;
}

[[gnu::noinline]] std::size_t count_algo(const std::vector<int>& v,
                                         int needle) {
    return static_cast<std::size_t>(std::count(v.begin(), v.end(), needle));
}

[[gnu::noinline]] std::size_t count_c_style(const char* const* strs,
                                            std::size_t n,
                                            const char* needle) {
    std::size_t count = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (std::strcmp(strs[i], needle) == 0) {
            ++count;
        }
    }
    return count;
}

[[gnu::noinline]] std::size_t count_strings(const std::vector<std::string>& v,
                                            const std::string& needle) {
    return static_cast<std::size_t>(std::count(v.begin(), v.end(), needle));
}

}  // namespace

int main() {
    std::vector<int> values = {1, 5, 2, 5, 3, 5, 4};
    const char* raw[] = {"Hamlet", "Macbeth", "Hamlet"};
    std::vector<std::string> strings = {"Hamlet", "Macbeth", "Hamlet"};

    const std::size_t a = count_loop(values, 5);
    const std::size_t b = count_algo(values, 5);
    const std::size_t c = count_c_style(raw, 3, "Hamlet");
    const std::size_t d = count_strings(strings, "Hamlet");

    std::printf("loop=%zu algo=%zu cstyle=%zu stl=%zu\n", a, b, c, d);
    return 0;
}
