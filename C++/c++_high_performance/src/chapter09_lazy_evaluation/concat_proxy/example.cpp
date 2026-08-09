// Comparing concatenated strings without building a temporary (PDF p.261-265).
//
// Naive:  (a + b) == c   allocates a temporary std::string.
// Proxy:  operator+ returns a ConcatProxy holding references; operator==
//         compares the two parts in place, no allocation, same syntax.

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <string>
#include <utility>

namespace chp9 {

class String {
public:
    String() = default;
    String(std::string str) : str_(std::move(str)) {}
    const std::string& str() const { return str_; }
    std::size_t size() const { return str_.size(); }

private:
    std::string str_;
};

// Proxy object holding the two operands of a pending concatenation.
// The references keep the strings alive only while the proxy is a temporary.
struct ConcatProxy {
    const std::string& a;
    const std::string& b;

    // Assigning the concatenation: converts on a temporary only.
    operator String() const&& { return String{a + b}; }
};

// Direct comparison of a+b against c, no temporary string (PDF p.261).
inline bool is_concat_equal(const std::string& a, const std::string& b,
                            const std::string& c) {
    return a.size() + b.size() == c.size() &&
           std::equal(a.begin(), a.end(), c.begin()) &&
           std::equal(b.begin(), b.end(), c.begin() + a.size());
}

// operator+ does no work: it just packages the operands into a proxy.
auto operator+(const String& a, const String& b) {
    return ConcatProxy{a.str(), b.str()};
}

// r-value only: storing the proxy in a variable then comparing would leave
// the referenced temporaries dangling (PDF p.264).
auto operator==(ConcatProxy&& concat, const String& rhs) -> bool {
    return is_concat_equal(concat.a, concat.b, rhs.str());
}

auto operator!=(ConcatProxy&& concat, const String& rhs) -> bool {
    return !is_concat_equal(concat.a, concat.b, rhs.str());
}

}  // namespace chp9

int main() {
    std::printf("== concat_proxy ==\n");

    const chp9::String a{"Cole"};
    const chp9::String b{"Porter"};
    const chp9::String c{"ColePorter"};

    // Same syntax as (a + b) == c, but no temporary string is allocated.
    std::printf("(a + b) == c: %d\n", (a + b) == c);
    std::printf("(a + b) == a: %d\n", (a + b) == a);

    // Assigning the concatenation requires an explicit String target
    // (auto would deduce ConcatProxy, PDF p.265).
    const chp9::String chagall = chp9::String{"Marc"} + chp9::String{"Chagall"};
    std::printf("String = a + b: %s\n", chagall.str().c_str());

    return 0;
}
